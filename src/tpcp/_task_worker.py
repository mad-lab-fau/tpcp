"""Coordinate lazy calls across local experiment participants."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Generic, TypeVar, cast

from joblib import Memory
from joblib import hash as joblib_hash

from tpcp._async_execution import BoundTask, _execution_context, _LocalWorkflow
from tpcp._task_artifacts import CallArtifacts
from tpcp._task_storage import InMemoryTaskStorage, TaskRecord, TaskStorage

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

    def __init__(
        self,
        memory: Memory,
        storage: TaskStorage | None = None,
        *,
        n_jobs=1,
        n_gpu_jobs=0,
        lease_seconds: float = 60,
        **pool_options,
    ):
        if memory.store_backend is None or lease_seconds <= 0:
            raise ValueError("Worker requires filesystem Memory and a positive lease_seconds.")
        self.memory = memory
        self.storage = InMemoryTaskStorage() if storage is None else storage
        self.lease_seconds = lease_seconds
        self._pool = _execution_context(coordinated=True, n_jobs=n_jobs, n_gpu_jobs=n_gpu_jobs, **pool_options)
        self._gpu_count = n_gpu_jobs
        self._scope: ContextVar[tuple] = ContextVar("worker_scope", default=())
        self._parents: ContextVar[tuple[TaskRecord, ...]] = ContextVar("worker_parents", default=())
        self._pending = {}

    async def __aenter__(self):
        execution = await self._pool.__aenter__()
        self._scope_slots = execution.scope_slots
        self._slots = execution.claim_slots
        self._token = active_worker.set(self)
        return self

    async def __aexit__(self, *exc):
        cancellation = None
        try:
            while self._pending:
                drain = asyncio.gather(*self._pending.values(), return_exceptions=True)
                while not drain.done():
                    try:
                        await asyncio.shield(drain)
                    except asyncio.CancelledError as error:
                        cancellation = error
            if cancellation is not None:
                raise cancellation
        finally:
            active_worker.reset(self._token)
            await self._pool.__aexit__(*exc)

    def scope(self, key):
        """Append a stable identity to ordinary tasks prepared inside this block."""
        return self._enter_scope(("scope", key))

    @contextmanager
    def _enter_scope(self, key):
        token = self._scope.set((*self._scope.get(), joblib_hash(key)))
        try:
            yield
        finally:
            self._scope.reset(token)

    def prepare(self, task: BoundTask[T]) -> PreparedTask[T]:
        """Prepare a replayable task. Await local workflows directly or use execute()."""
        if isinstance(task, _LocalWorkflow):
            raise TypeError("Local workflows require await or Worker.execute().")
        return self._prepare(self._resolve(task), self._scope.get(), self._parents.get())

    @staticmethod
    def _resolve(task):
        args = task.args() if callable(task.args) else task.args
        return replace(task, args=args, kwargs=task.kwargs.copy())

    def _prepare(self, task, scope, parents):
        return PreparedTask(
            task,
            CallArtifacts(self.memory, task, scope),
            () if task.cached else parents,
            scope,
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

    def _check_active_parents(self):
        parents = self._parents.get()
        if not parents:
            return
        with self.storage.transaction() as records:
            for parent in parents:
                self._check(records, parent)

    def _check_parents(self, records, prepared):
        for parent in prepared.parents:
            self._check(records, parent)

    def acquire(self, prepared: PreparedTask[T]) -> Decision[T]:
        """Claim available work or return the existing attempt."""
        return cast("Decision[T]", self._decide(prepared, claim=True))

    def _decide(self, prepared: PreparedTask[T], *, claim: bool) -> Decision[T] | None:
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
            # A plain Memory writer may have repaired the same call since its last failed load.
            damaged = previous.damaged_identity if previous else None
            if damaged and prepared.task.cached:
                identity = prepared.artifacts.output_identity(artifact)
                if damaged != json.dumps(identity):
                    damaged = None
            if not damaged and prepared.artifacts.available(artifact):
                if not previous or previous.status != "completed":
                    previous = TaskRecord(prepared.task_id, "completed", attempt, now, now, now, artifact)
                    records.put(previous)
                return Decision(prepared, previous, False)
            if not claim:
                return None
            record = TaskRecord(
                prepared.task_id,
                "running",
                attempt,
                now + self.lease_seconds,
                now,
                None,
                previous.artifact if previous and damaged else None,
                damaged,
            )
            records.put(record)
            return Decision(prepared, record, True)

    def _update(self, decision, status, stage=None, renew=None):
        with self.storage.transaction() as records:
            self._check_parents(records, decision.prepared)
            record = decision.record
            self._check(records, record)
            now = time.time()
            if renew is not None and not renew(now + self.lease_seconds):
                raise ClaimLostError("Admission reservation expired")
            reference = record.artifact
            if status == "completed":
                reference = decision.prepared.artifacts.reference(record.attempt)
                decision.prepared.artifacts.publish(reference, stage)
            records.put(
                replace(
                    record,
                    status=status,
                    expires=now + self.lease_seconds,
                    finished=None if status == "running" else now,
                    artifact=reference,
                    damaged_identity=None if status == "completed" else record.damaged_identity,
                )
            )

    async def _publish(self, decision, value):
        duration = time.time() - decision.record.started
        staging = decision.prepared.artifacts.stage(value, duration)
        writer = asyncio.create_task(asyncio.to_thread(staging.__enter__))
        try:
            stage = await asyncio.shield(writer)
        except asyncio.CancelledError:

            def discard(done):
                if not done.cancelled() and done.exception() is None:
                    staging.__exit__(None, None, None)

            writer.add_done_callback(discard)
            raise
        try:
            self._update(decision, "completed", stage)
        finally:
            staging.__exit__(None, None, None)

    @asynccontextmanager
    async def _maintain(self, decision, renew=None):
        async def heartbeat():
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                self._update(decision, "running", renew=renew)

        beat = asyncio.create_task(heartbeat())
        try:
            self._update(decision, "running", renew=renew)
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
        return await self._shared_call(("claim", claim.record.attempt), self._run, claim, shield=False)

    async def _run(self, claim, renew=None):
        async with self._maintain(claim, renew):
            worker_token = active_worker.set(self)
            scope_token = self._scope.set(claim.prepared.scope)
            parent_token = self._parents.set(claim.prepared.parents)
            try:
                value = await claim.prepared.task._execute()
            finally:
                self._parents.reset(parent_token)
                self._scope.reset(scope_token)
                active_worker.reset(worker_token)
            await self._publish(claim, value)
            if claim.prepared.task.cached and self.memory.mmap_mode:
                return await self.result(claim)
            return value

    async def result(self, decision: Decision[T]) -> T:
        """Load the completed artifact, waiting for its owner if necessary."""
        while True:
            with self.storage.transaction() as records:
                self._check_parents(records, decision.prepared)
                record = records.get(decision.prepared.task_id)
                if record and record.status == "completed" and decision.prepared.artifacts.available(record.artifact):
                    identity = decision.prepared.artifacts.output_identity(record.artifact)
                    break
                if not record or record.status != "running" or record.expires <= time.time():
                    raise ClaimLostError("Result unavailable. Acquire the prepared task again.")
            await asyncio.sleep(min(0.05, self.lease_seconds / 3))
        try:
            return await asyncio.to_thread(decision.prepared.artifacts.load, record.artifact)
        except Exception as error:
            with self.storage.transaction() as records:
                if (
                    records.get(record.task_id) == record
                    and decision.prepared.artifacts.output_identity(record.artifact) == identity
                ):
                    records.put(replace(record, status="failed", damaged_identity=json.dumps(identity)))
            raise ClaimLostError("Artifact load failed. Acquire the prepared task again.") from error

    async def _execute(self, prepared):
        while True:
            decision = self._decide(prepared, claim=False)
            if decision is None:
                async with AsyncExitStack() as capacity:
                    renew = None
                    if prepared.task.kind == "worker":
                        gpu = prepared.task.resource == "gpu"
                        if gpu and not self._gpu_count:
                            raise ValueError("GPU tasks require n_gpu_jobs > 0")
                        renew = await capacity.enter_async_context(self._slots.reserve(self.lease_seconds, gpu=gpu))
                    decision = self.acquire(prepared)
                    if decision.acquired:
                        return await self._run(decision, renew)
            try:
                return await self.result(decision)
            except ClaimLostError:
                continue

    async def execute(self, task: BoundTask[T]) -> T:
        if isinstance(task, _LocalWorkflow):
            token = active_worker.set(self)
            try:
                self._check_active_parents()
                result = await task._execute()
                self._check_active_parents()
                return result
            finally:
                active_worker.reset(token)
        prepared = await asyncio.to_thread(self._prepare, self._resolve(task), self._scope.get(), self._parents.get())
        key = (prepared.task_id, tuple(p.attempt for p in prepared.parents))
        return await self._shared_call(key, self._execute, prepared)

    async def _shared_call(self, key, operation, argument, *, shield=True):
        if key not in self._pending:
            future = asyncio.create_task(operation(argument))
            self._pending[key] = future

            def finished(done):
                self._pending.pop(key)
                if not done.cancelled():
                    done.exception()

            future.add_done_callback(finished)
        pending = self._pending[key]
        return await (asyncio.shield(pending) if shield else pending)

    @asynccontextmanager
    async def subtask(self, name, *, key):
        """Claim a manual scope. Only the acquired participant executes its body."""
        async with AsyncExitStack() as stack:
            stack.enter_context(self._enter_scope(("subtask", name, key)))
            renew = None
            if not self._parents.get():
                renew = await stack.enter_async_context(self._scope_slots.reserve(self.lease_seconds))
            prepared = self.prepare(BoundTask(_scope_value))
            decision = self.acquire(prepared)
            handle = _Subtask(self, decision)
            if decision.acquired:
                await stack.enter_async_context(self._maintain(decision, renew))
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
        await self.worker._publish(self.decision, value)
        self.completed = True

    async def result(self):
        return await self.worker.result(self.decision)
