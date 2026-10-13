"""Experimental async execution over joblib's reusable loky pool."""

from __future__ import annotations

import asyncio
import functools
import multiprocessing
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

import joblib
from joblib.executor import get_memmapping_executor
from joblib.externals.loky.process_executor import _ExceptionWithTraceback
from joblib.parallel import BatchedCalls
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Sequence

T = TypeVar("T")


@dataclass(frozen=True)
class Job(Generic[T]):
    """Describe a worker call and its metadata without reserving function keywords."""

    func: Callable[..., Any]
    args: tuple[Any, ...] | Callable[[], tuple[Any, ...]] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    resource: Literal["cpu", "gpu"] = "cpu"
    pass_context: bool = False


@dataclass(frozen=True)
class TaskConfig:
    """Scheduling requirements declared by an action before execution."""

    kind: Literal["worker", "workflow"]
    resource: Literal["cpu", "gpu"] | None


@dataclass(frozen=True, eq=False)
class Task(Job[T]):
    """An early-bound, awaitable action with callee-owned scheduling requirements.

    Worker tasks execute their callable in the shared process pool. Workflow
    tasks await an async callable on the coordinating event loop, where it can
    compose further tasks. Constructing either kind performs no computation.
    Arguments are bound by reference, as with ``joblib.delayed``.
    """

    func: Callable[..., T | Awaitable[T]]
    kind: Literal["worker", "workflow"] = "worker"
    # Awaitables represent executions, not values. Override the inherited Job
    # equality/hash as well: gather deduplicates awaitables by identity.
    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def config(self) -> TaskConfig:
        """Inspect scheduling without running the action."""
        return TaskConfig(self.kind, self.resource if self.kind == "worker" else None)

    def __await__(self) -> Generator[Any, None, T]:
        return self._execute().__await__()

    async def _execute(self) -> T:
        if self.kind == "worker":
            return await run_in_worker(self)
        with job_metadata(**self.metadata):
            args = self.args() if callable(self.args) else self.args
            return await cast("Awaitable[T]", self.func(*args, **self.kwargs))


@dataclass(frozen=True)
class WorkerContext:
    """Identity, assigned device and metadata for one worker call."""

    worker_id: str
    pid: int
    gpu_id: int | None
    job_id: int
    metadata: dict[str, Any]
    resource: Literal["cpu", "gpu"]
    batch_id: int = 0
    batch_index: int = 0
    batch_size: int = 1


@dataclass(frozen=True)
class JobEvent:
    """A job lifecycle notification delivered on the coordinating event-loop thread."""

    job_id: int
    status: Literal["queued", "started", "completed", "failed", "cancelled"]
    metadata: dict[str, Any]
    resource: Literal["cpu", "gpu"] = "cpu"
    worker: WorkerContext | None = None


_current_job_metadata: ContextVar[dict[str, Any] | None] = ContextVar("tpcp_job_metadata", default=None)
_current_worker_context: ContextVar[WorkerContext | None] = ContextVar("tpcp_worker_context", default=None)


@contextmanager
def job_metadata(**metadata: Any):
    """Attach metadata to jobs created in this scope and its nested asyncio tasks."""
    token = _current_job_metadata.set({**get_job_metadata(), **metadata})
    try:
        yield
    finally:
        _current_job_metadata.reset(token)


def get_job_metadata() -> dict[str, Any]:
    """Return a copy of the metadata for the current workflow or worker job."""
    return (_current_job_metadata.get() or {}).copy()


def get_worker_context() -> WorkerContext:
    """Read the current job's worker identity, assigned GPU and metadata."""
    context = _current_worker_context.get()
    if context is None:
        raise RuntimeError("Worker context is only available inside a worker job.")
    return context


@dataclass
class _JobOutcome:
    worker: WorkerContext
    value: Any = None
    error: Any = None


@dataclass
class _Request:
    job: Job
    metadata: dict[str, Any]
    job_id: int
    context: Context
    future: asyncio.Future


def _run_async_job(func, args, kwargs, metadata, job_id, gpu_id, resource, pass_context, batch_id, index, batch_size):
    from tpcp.parallel import _run_with_parallel_side_channels

    # Explicit per-job restoration avoids leaking context across reusable workers.
    context = WorkerContext(
        multiprocessing.current_process().name,
        os.getpid(),
        gpu_id,
        job_id,
        metadata,
        resource,
        batch_id,
        index,
        batch_size,
    )
    worker_token = _current_worker_context.set(context)
    token = _current_job_metadata.set(metadata)
    try:
        if pass_context:
            args = (context, *args)
        try:
            return _JobOutcome(context, value=_run_with_parallel_side_channels(func, args, kwargs))
        except Exception as error:  # noqa: BLE001 - transport individual worker failures
            # Preserve loky's remote traceback format while isolating failures
            # between individual jobs in the same batch.
            return _JobOutcome(context, error=_ExceptionWithTraceback(error))
    finally:
        _current_job_metadata.reset(token)
        _current_worker_context.reset(worker_token)


async def _gather_with_progress(coroutines, *, total: int, desc: str, progress_bar: bool):
    async def execute(effect):
        return await effect

    with tqdm(total=total, desc=desc, disable=not progress_bar) as bar:

        def completed(task):
            if not task.cancelled() and task.exception() is None:
                bar.update(1)

        async with asyncio.TaskGroup() as group:
            tasks = []
            for coroutine in coroutines:
                task = group.create_task(coroutine if asyncio.iscoroutine(coroutine) else execute(coroutine))
                task.add_done_callback(completed)
                tasks.append(task)
    return [task.result() for task in tasks]


class _AsyncExecution:
    def __init__(
        self,
        n_jobs: int,
        *,
        n_gpu_jobs: int = 0,
        gpu_ids: tuple[int, ...] = (),
        on_job_event: Callable[[JobEvent], None] | None = None,
        **parallel_kwargs: Any,
    ):
        self.configuration = (n_jobs, n_gpu_jobs, gpu_ids, on_job_event, parallel_kwargs)
        self.loop = asyncio.get_running_loop()
        self._on_job_event = on_job_event
        self._job_ids = count()
        self._batch_ids = count()
        self.users = 0
        # Keep joblib's pool configuration and serialization in one adapter.
        # Going through Parallel's configuration also preserves worker thread
        # limits and the exact arguments used to select its reusable executor.
        parallel: Any = joblib.Parallel(n_jobs=n_jobs, backend="loky", **parallel_kwargs)
        self._batch_size = parallel.batch_size
        gpu_batching = type(parallel._backend)(nesting_level=parallel._backend.nesting_level)
        self._batching = {"cpu": parallel._backend, "gpu": gpu_batching}
        for backend in self._batching.values():
            backend.parallel = parallel
        self._context_id = parallel._id
        self._nested_backend = parallel._backend.get_nested_backend()
        # joblib 1.3 calls this mapping _backend_args; newer releases renamed it
        # and also allow constructor settings on the backend itself.
        settings = parallel._backend_kwargs if hasattr(parallel, "_backend_kwargs") else parallel._backend_args
        executor_kwargs = {**getattr(parallel._backend, "backend_kwargs", {}), **settings}
        # LokyBackend.configure consumes these hints before selecting the pool.
        executor_kwargs.pop("prefer", None)
        executor_kwargs.pop("require", None)
        timeout = executor_kwargs.pop("idle_worker_timeout", 300)
        self._executor: Any = get_memmapping_executor(
            n_jobs,
            timeout=timeout,
            env=parallel._backend._prepare_worker_env(n_jobs),
            context_id=self._context_id,
            **executor_kwargs,
        )
        self._slots = asyncio.Semaphore(n_jobs)
        self._gpu_slots = asyncio.Semaphore(n_gpu_jobs)
        self._gpu_ids: asyncio.Queue[int] = asyncio.Queue()
        for slot in range(n_gpu_jobs):
            self._gpu_ids.put_nowait(gpu_ids[slot % len(gpu_ids)])
        self._n_gpu_jobs = n_gpu_jobs
        self._pending: set[asyncio.Future] = set()
        self._calls: set[asyncio.Future] = set()
        self._closing = False
        self._queues: dict[str, asyncio.Queue[_Request]] = {"cpu": asyncio.Queue(), "gpu": asyncio.Queue()}
        self._dispatchers = [asyncio.create_task(self._dispatch(resource)) for resource in self._queues]

    def _select_memmap_context(self) -> None:
        # BatchedCalls invokes this in loky's serialization thread, immediately
        # before reducing the job arguments, just as joblib.Parallel does.
        self._executor._temp_folder_manager.set_current_context(self._context_id)

    async def run(self, func: Callable[..., T] | Job[T], /, *args: Any, **kwargs: Any) -> T:
        from tpcp.parallel import _merge_parallel_side_channel_data

        if self._closing:
            raise RuntimeError("The execution context is closing.")
        if isinstance(func, Task) and func.kind == "workflow":
            return await func
        finished = asyncio.get_running_loop().create_future()
        self._calls.add(finished)
        job = func if isinstance(func, Job) else Job(func, args=args, kwargs=kwargs)
        metadata = {**get_job_metadata(), **job.metadata}
        job_id = next(self._job_ids)
        request = _Request(job, metadata, job_id, copy_context(), self.loop.create_future())
        worker = None
        try:
            self._emit(job_id, "queued", metadata, job.resource)
            if job.resource == "gpu" and self._n_gpu_jobs == 0:
                raise RuntimeError("GPU jobs require n_gpu_jobs > 0 in execution_context.")
            self._queues[job.resource].put_nowait(request)
            outcome = await asyncio.shield(request.future)
            worker = outcome.worker
            if outcome.error is not None:
                raise outcome.error
            result = _merge_parallel_side_channel_data(outcome.value)
        except asyncio.CancelledError:
            request.future.cancel()
            self._emit(job_id, "cancelled", metadata, job.resource)
            raise
        except Exception:
            self._emit(job_id, "failed", metadata, job.resource, worker)
            raise
        else:
            self._emit(job_id, "completed", metadata, job.resource, worker)
            return result
        finally:
            finished.set_result(None)
            self._calls.discard(finished)

    def _emit(self, job_id, status, metadata, resource, worker=None):
        if self._on_job_event is not None:
            token = _current_job_metadata.set(metadata)
            try:
                self._on_job_event(JobEvent(job_id, status, metadata.copy(), resource, worker))
            finally:
                _current_job_metadata.reset(token)

    def _prepare(self, request):
        from tpcp.parallel import delayed

        job = request.job
        args = cast("tuple[Any, ...]", job.args() if callable(job.args) else job.args)
        with job_metadata(**request.metadata):
            return delayed(job.func)(*args, **job.kwargs)

    def _release_slots(self, gpu_id):
        self._slots.release()
        if gpu_id is not None:
            self._gpu_ids.put_nowait(gpu_id)
            self._gpu_slots.release()

    def _take_batch(self, queue, first, resource):
        target = self._batching[resource].compute_batch_size() if self._batch_size == "auto" else self._batch_size
        requests = [first]
        while len(requests) < target and not queue.empty():
            candidate = queue.get_nowait()
            if not candidate.future.cancelled():
                requests.append(candidate)
        return requests

    @staticmethod
    def _fail_requests(requests, error=None):
        for request in requests:
            if not request.future.done():
                if error is None:
                    request.future.cancel()
                else:
                    request.future.set_exception(error)

    def _submit_batch(self, requests, resource, gpu_id):
        prepared = []
        for request in requests:
            if request.future.cancelled():
                continue
            try:
                self._emit(request.job_id, "started", request.metadata, resource)
                prepared.append((request, request.context.run(self._prepare, request)))
            except Exception as error:  # noqa: BLE001 - route preparation failures to their caller
                self._fail_requests([request], error)
        if not prepared:
            return False
        batch_id = next(self._batch_ids)
        size = len(prepared)
        packets = [
            (
                _run_async_job,
                (
                    func,
                    args,
                    kwargs,
                    request.metadata,
                    request.job_id,
                    gpu_id,
                    resource,
                    request.job.pass_context,
                    batch_id,
                    index,
                    size,
                ),
                {},
            )
            for index, (request, (func, args, kwargs)) in enumerate(prepared)
        ]
        started = time.perf_counter()
        batch = BatchedCalls(packets, self._nested_backend, reducer_callback=self._select_memmap_context)
        future = asyncio.wrap_future(self._executor.submit(batch))
        self._pending.add(future)
        future.add_done_callback(
            functools.partial(
                self._batch_finished,
                requests=tuple(request for request, _ in prepared),
                resource=resource,
                gpu_id=gpu_id,
                started=started,
            )
        )
        return True

    async def _dispatch(self, resource):
        queue = self._queues[resource]
        while True:
            request = await queue.get()
            if request.future.cancelled():
                continue
            requests = [request]
            gpu_id = None
            gpu_acquired = False
            cpu_acquired = False
            try:
                if resource == "gpu":
                    await self._gpu_slots.acquire()
                    gpu_acquired = True
                    gpu_id = self._gpu_ids.get_nowait()
                await self._slots.acquire()
                cpu_acquired = True
                requests = self._take_batch(queue, request, resource)
                if not self._submit_batch(requests, resource, gpu_id):
                    continue
                cpu_acquired = gpu_acquired = False  # The batch callback now owns both slots.
            except asyncio.CancelledError:
                self._fail_requests(requests)
                raise
            except Exception as error:  # noqa: BLE001 - route dispatch failures to the batch's callers
                self._fail_requests(requests, error)
            finally:
                if cpu_acquired:
                    self._slots.release()
                if gpu_acquired:
                    self._gpu_ids.put_nowait(cast("int", gpu_id))
                    self._gpu_slots.release()

    def _batch_finished(self, future, *, requests, resource, gpu_id, started):
        try:
            outcomes = future.result()
        except BaseException as error:  # noqa: BLE001 - forward executor failures, always releasing slots
            self._fail_requests(requests, error)
        else:
            self._batching[resource].batch_completed(len(requests), time.perf_counter() - started)
            for request, outcome in zip(requests, outcomes, strict=True):
                if not request.future.done():
                    request.future.set_result(outcome)
        finally:
            self._pending.discard(future)
            self._release_slots(gpu_id)

    async def close(self) -> None:
        self._closing = True
        cancellation = None
        # Calls accepted before shutdown may still be waiting for a slot.
        # Shield Futures directly, without a separate Task that runner-wide
        # cancellation could interrupt independently of the owner.
        for futures in (self._calls, self._pending, self._dispatchers):
            if futures is self._dispatchers:
                for task in self._dispatchers:
                    task.cancel()
            if not futures:
                continue
            waiting = asyncio.gather(*(asyncio.shield(future) for future in futures), return_exceptions=True)
            while not waiting.done():
                try:
                    await asyncio.shield(waiting)
                except asyncio.CancelledError as error:
                    cancellation = error
            waiting.result()
        # Release this context's memmaps, retaining the reusable loky workers.
        self._executor._temp_folder_manager._clean_temporary_resources(
            context_id=self._context_id, force=False, allow_non_empty=True
        )
        if cancellation is not None:
            raise cancellation


_current_execution: ContextVar[_AsyncExecution | None] = ContextVar("tpcp_async_execution", default=None)
_active_execution: _AsyncExecution | None = None
_execution_ownership_lock = threading.Lock()


@asynccontextmanager
async def execution_context(
    *,
    n_jobs: int = 1,
    n_gpu_jobs: int = 0,
    gpu_ids: Sequence[int] | None = None,
    on_job_event: Callable[[JobEvent], None] | None = None,
    **parallel_kwargs: Any,
):
    """Use joblib's reusable loky pool for experimental async actions.

    Use ``async with execution_context(n_jobs=2)`` around an awaited action.
    Nested asyncio tasks inherit the execution context. The positive ``n_jobs``
    limits concurrent batches; async orchestration stays in the calling process.
    ``n_gpu_jobs`` limits GPU batches, which also occupy a CPU slot. ``gpu_ids``
    assigns devices to those slots; the job callable handles device placement.
    ``batch_size`` uses joblib's fixed or automatic batching. ``on_job_event``
    receives lifecycle notifications on the coordinating event-loop thread.
    Exit waits for submitted work and releases temporary array files, including
    after failure, while leaving workers available for later joblib calls.
    Worker functions must not create their own pools.

    Additional keyword arguments are forwarded to ``joblib.Parallel`` for
    serialization configuration, such as ``max_nbytes``, ``mmap_mode`` and
    ``temp_folder``. TPCP worker context transfer is preserved. This prototype
    uses private joblib pool and batch interfaces, isolated in this module.
    Overlapping contexts share the same execution adapter and must use identical
    configuration, so they cannot replace joblib's active singleton pool.
    """
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive.")
    if n_gpu_jobs < 0:
        raise ValueError("n_gpu_jobs must be non-negative.")
    devices = tuple(range(n_gpu_jobs)) if gpu_ids is None else tuple(gpu_ids)
    if n_gpu_jobs and not devices:
        raise ValueError("GPU slots require at least one gpu_id.")
    global _active_execution  # noqa: PLW0603
    with _execution_ownership_lock:
        if _active_execution is not None:
            if _active_execution.loop is not asyncio.get_running_loop():
                raise RuntimeError("Overlapping execution contexts must use the same event loop.")
            if _active_execution._closing:
                raise RuntimeError("The active execution context is closing.")
            if _active_execution.configuration != (n_jobs, n_gpu_jobs, devices, on_job_event, parallel_kwargs):
                raise ValueError("Overlapping execution contexts require identical configuration.")
            execution = _active_execution
        else:
            execution = _AsyncExecution(
                n_jobs, n_gpu_jobs=n_gpu_jobs, gpu_ids=devices, on_job_event=on_job_event, **parallel_kwargs
            )
            _active_execution = execution
        execution.users += 1
    token = _current_execution.set(execution)
    try:
        yield execution
    finally:
        _current_execution.reset(token)
        with _execution_ownership_lock:
            execution.users -= 1
            last_user = execution.users == 0
        if last_user:
            try:
                await execution.close()
            finally:
                with _execution_ownership_lock:
                    _active_execution = None


async def run_in_worker(func: Callable[..., T] | Job[T], /, *args: Any, **kwargs: Any) -> T:
    """Await a leaf job in the active :func:`execution_context` pool.

    Result and side-channel merging happen in the calling event loop. Cancelling
    the await does not interrupt a job already running in a worker process.
    """
    if isinstance(func, Task) and func.kind == "workflow":
        return await func
    execution = _current_execution.get()
    if execution is None:
        raise RuntimeError("Async actions require an active execution_context.")
    return await execution.run(func, *args, **kwargs)
