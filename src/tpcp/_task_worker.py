"""Coordinate lazy calls across local experiment participants."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from joblib import Memory
from joblib import hash as joblib_hash

from tpcp._async_execution import BoundTask, execution_context
from tpcp._task_artifacts import CallArtifacts
from tpcp._task_storage import InMemoryTaskStorage, TaskRecord

T = TypeVar("T")
active_worker: ContextVar[Worker | None] = ContextVar("active_worker", default=None)


class ClaimLostError(RuntimeError):
    """The participant must reacquire this task or its enclosing subtask."""


@dataclass(frozen=True)
class PreparedTask(Generic[T]):
    task: BoundTask[T]
    artifacts: CallArtifacts
    parents: tuple[TaskRecord, ...]
    scope: tuple

    @property
    def task_id(self):
        return self.artifacts.task_id


@dataclass(frozen=True)
class Decision(Generic[T]):
    prepared: PreparedTask[T]
    record: TaskRecord
    acquired: bool


def _scope_value():
    """Identity for a manually completed scope."""


class Worker:
    """Run tasks with local capacity and share their results through storage and Memory."""

    def __init__(self, memory: Memory, storage=None, *, n_jobs=1, n_gpu_jobs=0, lease_seconds=60, **pool_options):
        if memory.store_backend is None or lease_seconds <= 0:
            raise ValueError("Worker requires filesystem Memory and a positive lease_seconds.")
        self.memory = memory
        self.storage = InMemoryTaskStorage() if storage is None else storage
        self.lease_seconds = lease_seconds
        self._pool = execution_context(n_jobs=n_jobs, n_gpu_jobs=n_gpu_jobs, **pool_options)
        self._slots = asyncio.Semaphore(n_jobs)
        self._gpu = asyncio.Semaphore(n_gpu_jobs)
        self._gpu_count = n_gpu_jobs
        self._scope_slots = asyncio.Semaphore(n_jobs)
        self._scope: ContextVar[tuple] = ContextVar("worker_scope", default=())
        self._parents: ContextVar[tuple[TaskRecord, ...]] = ContextVar("worker_parents", default=())
        self._pending = {}

    async def __aenter__(self):
        await self._pool.__aenter__()
        self._token = active_worker.set(self)
        return self

    async def __aexit__(self, *exc):
        try:
            while self._pending:
                await asyncio.gather(*self._pending.values(), return_exceptions=True)
        finally:
            active_worker.reset(self._token)
            await self._pool.__aexit__(*exc)

    def scope(self, key):
        return self._enter_scope(("scope", key))

    @contextmanager
    def _enter_scope(self, key):
        """Append a stable identity to ordinary tasks prepared inside this block."""
        token = self._scope.set((*self._scope.get(), joblib_hash(key)))
        try:
            yield
        finally:
            self._scope.reset(token)

    def prepare(self, task: BoundTask[T]) -> PreparedTask[T]:
        args = task.args() if callable(task.args) else task.args
        task = replace(task, args=args, kwargs=task.kwargs.copy())
        return PreparedTask(
            task,
            CallArtifacts(self.memory, task, self._scope.get()),
            () if task.cached else self._parents.get(),
            self._scope.get(),
        )

    @staticmethod
    def _check(records, claim):
        current = records.get(claim.task_id)
        if (
            not current
            or current.attempt != claim.attempt
            or current.status != "running"
            or current.expires <= time.time()
        ):
            raise ClaimLostError(f"Claim expired for {claim.task_id}")

    def _check_parents(self, records, prepared):
        for parent in prepared.parents:
            self._check(records, parent)

    def acquire(self, prepared: PreparedTask[T]) -> Decision[T]:
        with self.storage.transaction() as records:
            self._check_parents(records, prepared)
            now = time.time()
            previous = records.get(prepared.task_id)
            if previous and previous.status == "running" and previous.expires > now:
                return Decision(prepared, previous, False)
            attempt = uuid.uuid4().hex
            artifact = prepared.artifacts.reference(attempt) if prepared.task.cached else None
            if previous and previous.status == "completed":
                artifact = previous.artifact
            if prepared.artifacts.available(artifact):
                if not previous or previous.status != "completed":
                    previous = TaskRecord(prepared.task_id, "completed", attempt, now, now, now, artifact)
                    records.put(previous)
                return Decision(prepared, previous, False)
            record = TaskRecord(prepared.task_id, "running", attempt, now + self.lease_seconds, now, None, None)
            records.put(record)
            return Decision(prepared, record, True)

    def _update(self, decision, status, value=None):
        with self.storage.transaction() as records:
            self._check_parents(records, decision.prepared)
            record = decision.record
            self._check(records, record)
            now = time.time()
            reference = None
            if status == "completed":
                reference = decision.prepared.artifacts.reference(record.attempt)
                decision.prepared.artifacts.publish(reference, value, now - record.started)
            records.put(
                replace(
                    record,
                    status=status,
                    expires=now + self.lease_seconds,
                    finished=None if status == "running" else now,
                    artifact=reference,
                )
            )

    @asynccontextmanager
    async def _maintain(self, decision):
        async def heartbeat():
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                self._update(decision, "running")

        beat = asyncio.create_task(heartbeat())
        try:
            yield
        except BaseException:
            with suppress(ClaimLostError):
                self._update(decision, "failed")
            raise
        finally:
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)

    async def run(self, claim: Decision[T]) -> T:
        if not claim.acquired:
            raise ValueError("run requires an acquired claim")
        self._update(claim, "running")
        async with self._maintain(claim):
            worker_token = active_worker.set(self)
            scope_token = self._scope.set(claim.prepared.scope)
            parent_token = self._parents.set(claim.prepared.parents)
            try:
                value = await claim.prepared.task._execute()
            finally:
                self._parents.reset(parent_token)
                self._scope.reset(scope_token)
                active_worker.reset(worker_token)
            self._update(claim, "completed", value)
            return value

    async def result(self, decision: Decision[T]) -> T:
        """Load the completed artifact, waiting for its owner if necessary."""
        while True:
            with self.storage.transaction() as records:
                self._check_parents(records, decision.prepared)
                record = records.get(decision.prepared.task_id)
                if record and record.status == "completed" and decision.prepared.artifacts.available(record.artifact):
                    return decision.prepared.artifacts.load(record.artifact)
                if not record or record.status != "running" or record.expires <= time.time():
                    raise ClaimLostError("Result unavailable. Acquire the prepared task again.")
            await asyncio.sleep(min(0.05, self.lease_seconds / 3))

    async def _execute(self, prepared):
        while True:
            async with AsyncExitStack() as capacity:
                if prepared.task.kind == "worker":
                    if prepared.task.resource == "gpu":
                        if not self._gpu_count:
                            raise ValueError("GPU tasks require n_gpu_jobs > 0")
                        await capacity.enter_async_context(self._gpu)
                    await capacity.enter_async_context(self._slots)
                decision = self.acquire(prepared)
                if decision.acquired:
                    return await self.run(decision)
            try:
                return await self.result(decision)
            except ClaimLostError:
                continue

    async def execute(self, task: BoundTask[T]) -> T:
        prepared = self.prepare(task)
        key = (prepared.task_id, tuple(p.attempt for p in prepared.parents))
        if key not in self._pending:
            future = asyncio.create_task(self._execute(prepared))
            self._pending[key] = future

            def finished(done):
                self._pending.pop(key)
                if not done.cancelled():
                    done.exception()

            future.add_done_callback(finished)
        return await asyncio.shield(self._pending[key])

    @asynccontextmanager
    async def subtask(self, name, *, key):
        """Claim a manual scope. Only the acquired participant executes its body."""
        async with AsyncExitStack() as stack:
            stack.enter_context(self._enter_scope(("subtask", name, key)))
            if not self._parents.get():
                await stack.enter_async_context(self._scope_slots)
            prepared = self.prepare(BoundTask(_scope_value))
            decision = self.acquire(prepared)
            handle = _Subtask(self, decision)
            if decision.acquired:
                await stack.enter_async_context(self._maintain(decision))
                token = self._parents.set((*prepared.parents, decision.record))
                stack.callback(self._parents.reset, token)
            else:
                await stack.aclose()
            yield handle
            if decision.acquired and not handle.completed:
                raise RuntimeError("Acquired subtask exited without complete()")


class _Subtask:
    def __init__(self, worker, decision):
        self.worker = worker
        self.decision = decision
        self.acquired = decision.acquired
        self.completed = False

    async def complete(self, value):
        if not self.acquired:
            raise ValueError("Only the owner may complete a subtask")
        self.worker._update(self.decision, "completed", value)
        self.completed = True

    async def result(self):
        return await self.worker.result(self.decision)
