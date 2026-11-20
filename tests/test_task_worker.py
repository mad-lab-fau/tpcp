"""Coordination through public Worker and joblib interfaces."""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from joblib import Memory

from tpcp.parallel import CachedTask, InMemoryTaskStorage, Task, Worker


def sample(*, resource):
    """Return a unique value to distinguish replay from recomputation."""
    return resource, uuid.uuid4().hex


def test_scoped_and_cached_calls(tmp_path):
    """Replay ordinary calls per scope and cached calls across scopes."""
    memory = Memory(tmp_path, verbose=0)

    async def scenario():
        async with Worker(memory, InMemoryTaskStorage()) as worker:
            call = Task(sample)(resource="argument")
            cached = CachedTask(sample)(resource="argument")
            with worker.scope("one"):
                a, b = await asyncio.gather(call, call)
                assert a == b == await call
                assert a[0] == "argument"
                shared = await cached
            with worker.scope("two"):
                assert await call != a
                plain: Any = memory.cache(sample)
                assert await cached == shared == plain(resource="argument")

    asyncio.run(scenario())


def test_cached_artifact_removal_and_existing_joblib_hit(tmp_path):
    """Honor existing joblib entries and rebuild cleared artifacts."""
    memory = Memory(tmp_path, verbose=0)
    cached: Any = memory.cache(sample)
    original = cached(resource="gpu")

    async def scenario():
        async with Worker(memory) as worker:
            task = CachedTask(sample)(resource="gpu")
            assert await task == original
            prepared = worker.prepare(task)
            cached.clear(warn=False)
            assert await worker.execute(task) != original
            assert await worker.result(worker.acquire(prepared)) == cached(resource="gpu")

    asyncio.run(scenario())


def test_scope_takeover_fences_child_and_reuses_completed_results(tmp_path):
    """Reject writes from both an expired scope and its child claim."""
    import pytest

    from tpcp.parallel import ClaimLostError

    async def scenario():
        store = InMemoryTaskStorage()
        memory = Memory(tmp_path, verbose=0)
        # Blocking the owner loop models a stalled coordinator that cannot heartbeat.
        async with (
            Worker(memory, store, n_jobs=2, lease_seconds=0.3) as first,
            Worker(memory, store, n_jobs=2) as second,
            first.subtask("fold", key=1) as old,
        ):
            completed = Task(sample)(resource="done")
            value = await first.execute(completed)
            pending = first.acquire(first.prepare(Task(sample)(resource="next")))
            import time

            time.sleep(0.35)
            async with second.subtask("fold", key=1) as new:
                assert new.acquired
                assert await second.execute(completed) == value
                with pytest.raises(ClaimLostError):
                    await first.run(pending)
                await new.complete(value)
            with pytest.raises(ClaimLostError):
                await old.complete("stale")
            raise ClaimLostError("leave expired owner")

    with pytest.raises(ClaimLostError, match="leave expired owner"):
        asyncio.run(scenario())


def test_sqlite_processes_share_result_and_resume(tmp_path):
    """Independent script invocations receive a single random draw."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = """
import asyncio
import sys
from pathlib import Path
from joblib import Memory
from test_task_worker import sample
from tpcp.parallel import Worker, SQLiteTaskStorage, Task

async def main():
    root = Path(sys.argv[1])
    async with Worker(Memory(root / "cache"), SQLiteTaskStorage(root / "records")) as worker:
        with worker.scope("trial"):
            print((await Task(sample)(resource="cpu"))[1])

asyncio.run(main())
"""
    command = [sys.executable, "-c", script, str(tmp_path)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), *sys.path])}
    processes = [subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True) for _ in range(2)]
    values = [process.communicate(timeout=45)[0].strip() for process in processes]
    assert all(process.returncode == 0 for process in processes)
    assert values[0] == values[1] == subprocess.check_output(command, env=env, text=True).strip()


def fail_first(path):
    """Fail once per marker path."""
    if not path.exists():
        path.touch()
        raise ValueError("first attempt")
    return 42


def test_failed_call_can_retry(tmp_path):
    """A failed attempt leaves the same logical call retryable."""
    import pytest

    async def scenario():
        async with Worker(Memory(tmp_path / "cache")):
            task = Task(fail_first)(tmp_path / "marker")
            with pytest.raises(ValueError, match="first attempt"):
                await task
            assert await task == 42

    asyncio.run(scenario())


def blocked_sample(directory):
    """Signal worker entry and wait for the test to release the computation."""
    import time
    from pathlib import Path

    with (directory / "calls").open("a") as calls:
        calls.write("started\n")
    deadline = time.monotonic() + 15
    while not Path(directory / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("worker was never released")
        time.sleep(0.01)
    return "done"


def test_remote_wait_leaves_capacity_and_cancelled_caller_keeps_shared_work(tmp_path):
    """A participant waiting on a peer can finish a different task."""

    async def scenario():
        memory = Memory(tmp_path / "cache", verbose=0)
        storage = InMemoryTaskStorage()
        async with Worker(memory, storage, n_jobs=2) as first, Worker(memory, storage, n_jobs=2) as second:
            bound = Task(blocked_sample)(tmp_path)
            waiter = asyncio.create_task(first.execute(bound))
            async with asyncio.timeout(10):
                while not (tmp_path / "calls").exists():
                    await asyncio.sleep(0.01)
            peer = asyncio.create_task(second.execute(bound))
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            try:
                assert (await asyncio.wait_for(second.execute(Task(sample)(resource="free")), 5))[0] == "free"
            finally:
                (tmp_path / "release").touch()
            assert await peer == "done"
            assert (tmp_path / "calls").read_text() == "started\n"

    asyncio.run(scenario())


def test_joblib_code_invalidation(tmp_path):
    """A completed coordination row must not outlive joblib's code validity."""
    import importlib.util

    path = tmp_path / "model.py"

    def implementation(value):
        path.write_text(f"def score(x):\n    return x + {value}\n")
        spec = importlib.util.spec_from_file_location("model", path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        return module.score

    async def scenario():
        memory = Memory(tmp_path / "cache", verbose=0)
        async with Worker(memory) as worker:
            assert await worker.execute(CachedTask(implementation(1))(2)) == 3
            updated = implementation(10)
            assert await worker.execute(CachedTask(updated)(2)) == 12
            assert memory.cache(updated)(2) == 12
            assert await worker.execute(CachedTask(updated)(3)) == 13

    asyncio.run(scenario())


def test_thread_participants_share_in_memory_claims(tmp_path):
    """Independent event loops can participate through the same metadata store."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)
    storage = InMemoryTaskStorage()

    async def participant():
        async with Worker(Memory(tmp_path), storage):
            await asyncio.to_thread(barrier.wait, 5)
            return await Task(sample)(resource="cpu")

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda _: asyncio.run(participant()), range(2)))
    assert results[0] == results[1]


if TYPE_CHECKING:
    from typing import assert_type

    from tpcp.parallel import BoundTask

    async def _check_types(worker: Worker):
        def score(value: int, *, resource: str) -> float:
            return float(value) if resource else 0.0

        async def fold(value: int) -> str:
            return str(value)

        leaf = Task(score, resource="gpu")(1, resource="local")
        assert_type(leaf, BoundTask[float])
        assert_type(await Task(fold)(1), str)
        assert_type(await CachedTask(score)(1, resource="local"), float)
        assert_type(await worker.run(worker.acquire(worker.prepare(leaf))), float)


def test_cached_call_requires_worker():
    """Do not silently bypass shared cache semantics."""
    import pytest

    from tpcp.parallel import execution_context

    async def scenario():
        async with execution_context():
            with pytest.raises(RuntimeError, match="active Worker"):
                await CachedTask(sample)(resource="cpu")

    asyncio.run(scenario())


def test_prototype_task_class_and_positional_binding():
    """Existing Task annotations, instance checks and positional calls still work."""
    effect: Task[str] = Task(str, (1,))
    assert isinstance(effect, Task)
    assert isinstance(Task(sample)(resource="cpu"), Task)


async def nested_sample():
    """Compose an ordinary leaf on the active coordinator."""
    return await Task(sample)(resource="nested")


def test_prepared_workflow_runs_children_in_captured_scope(tmp_path):
    """Preparing a workflow fixes its scope even if run much later."""

    async def scenario():
        async with Worker(Memory(tmp_path)) as worker:
            with worker.scope("A"), worker.scope("fold"):
                expected = await Task(sample)(resource="nested")
                prepared = worker.prepare(Task(nested_sample)())
            with worker.scope("B"):
                assert await worker.run(worker.acquire(prepared)) == expected
                assert await Task(sample)(resource="nested") != expected

    asyncio.run(scenario())


def test_staging_large_result_leaves_coordinator_and_storage_available(tmp_path):
    """A slow serializer cannot block heartbeats or prevent other claims."""
    from threading import Event

    writing, release = Event(), Event()

    class SlowResult:
        def __reduce__(self):
            writing.set()
            if not release.wait(5):
                raise TimeoutError("serialization blocked the coordinator")
            return int, (42,)

    async def scenario():
        async with Worker(Memory(tmp_path)) as worker:
            async with worker.subtask("fold", key=1) as fold:
                pending = asyncio.create_task(fold.complete(SlowResult()))
                try:
                    assert await asyncio.to_thread(writing.wait, 5)
                    assert worker.acquire(worker.prepare(Task(sample)(resource="independent"))).acquired
                finally:
                    release.set()
                await pending
            async with worker.subtask("fold", key=1) as replay:
                assert not replay.acquired
                assert await replay.result() == 42

    asyncio.run(scenario())


def context_sample(context, suffix):
    """Identify the injected context and distinguish separate executions."""
    return context.resource, suffix, uuid.uuid4().hex


def test_public_dispatch_coordinates_context_calls(tmp_path):
    """Both submission paths honor cache admission and inject worker context."""
    import pytest

    from tpcp.parallel import execution_context, run_in_worker

    async def scenario():
        async with execution_context() as adapter:
            cached = CachedTask(sample)(resource="cpu")
            for submit in (adapter.run, run_in_worker):
                with pytest.raises(RuntimeError, match="active Worker"):
                    await submit(cached)
            async with Worker(Memory(tmp_path)):
                call = Task(context_sample, ("arg",), pass_context=True)
                value = await run_in_worker(call)
                assert value[:2] == ("cpu", "arg")
                assert await adapter.run(call) == await call == value

    asyncio.run(scenario())


def test_corrupt_artifact_recovery(tmp_path):
    """Recompute damaged ordinary and persistent entries once."""
    import json
    from pathlib import Path

    async def scenario():
        async with Worker(Memory(tmp_path)) as worker:
            for task in (Task(sample)(resource="cpu"), CachedTask(sample)(resource="cpu")):
                old = await task
                decision = worker.acquire(worker.prepare(task))
                assert decision.record.artifact is not None
                path = Path(decision.prepared.artifacts.backend.location).joinpath(
                    *json.loads(decision.record.artifact)
                )
                (path / "output.pkl").write_bytes(b"damaged")
                new = await task
                assert new != old
                assert await task == new

    asyncio.run(scenario())


def test_subtask_capacity_shared_across_workers(tmp_path):
    """Top-level scopes share a slot while the admitted scope can still run leaves."""

    async def scenario():
        memory = Memory(tmp_path)
        started = asyncio.Event()
        entered = asyncio.Event()
        async with Worker(memory) as first, Worker(memory) as second:

            async def contender():
                started.set()
                async with second.subtask("fold", key=2) as sub:
                    entered.set()
                    await sub.complete(2)

            async with first.subtask("fold", key=1) as sub:
                pending = asyncio.create_task(contender())
                await started.wait()
                value = await first.execute(Task(sample)(resource="cpu"))
                assert not entered.is_set()
                await sub.complete(value)
            await asyncio.wait_for(pending, 5)

    asyncio.run(scenario())


def test_replacement_during_failed_load_is_preserved(tmp_path, monkeypatch):
    """An external joblib writer can replace the output while a reader fails."""
    import json
    from pathlib import Path

    from tpcp._task_artifacts import CallArtifacts

    original_load = CallArtifacts.load
    memory = Memory(tmp_path, verbose=0)
    cached: Any = memory.cache(sample)
    replacement = []

    def racing_load(self, reference):
        if not replacement:
            path = Path(self.backend.location).joinpath(*json.loads(reference), "output.pkl")
            path.unlink()
            replacement.append(cached(resource="cpu"))
            raise FileNotFoundError
        return original_load(self, reference)

    async def scenario():
        async with Worker(memory):
            task = CachedTask(sample)(resource="cpu")
            old = await task
            monkeypatch.setattr(CallArtifacts, "load", racing_load)
            value = await task
            assert value == replacement[0] != old

    asyncio.run(scenario())


def test_worker_loop_binding_and_owner_identity(tmp_path):
    """Deferred arguments run on the loop and an ordinary owner keeps its value."""

    async def scenario():
        value = []

        async def workflow(*, argument):
            assert argument == 2
            return value

        kwargs = {}
        metadata = {}

        def args():
            asyncio.get_running_loop()
            return ()

        task = Task(workflow, args=args, kwargs=kwargs, metadata=metadata)
        kwargs["argument"] = 2
        metadata["fold"] = 1
        assert task.metadata == metadata
        async with Worker(Memory(tmp_path, mmap_mode="r")):
            assert await task is value
            assert await task is not value

    asyncio.run(scenario())


def test_explicit_run_cancellation_releases_lease(tmp_path):
    """A timed-out owner immediately yields its claim."""

    async def scenario():
        release = asyncio.Event()

        async def blocked():
            await release.wait()

        async with Worker(Memory(tmp_path)) as worker:
            prepared = worker.prepare(Task(blocked)())
            try:
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(worker.run(worker.acquire(prepared)), 0.05)
                assert worker.acquire(prepared).acquired
            finally:
                release.set()

    asyncio.run(scenario())


def test_cancelled_staging_cannot_publish(tmp_path, monkeypatch):
    """A cancelled serializer cannot overwrite its replacement."""
    from contextlib import contextmanager
    from threading import Event

    from tpcp._task_artifacts import CallArtifacts

    entered, release, discarded = Event(), Event(), Event()
    original_stage = CallArtifacts.stage

    @contextmanager
    def stage(self, value, duration):
        try:
            with original_stage(self, value, duration) as path:
                yield path
        finally:
            if isinstance(value, SlowResult):
                discarded.set()

    class SlowResult:
        def __reduce__(self):
            entered.set()
            assert release.wait(5)
            return int, (99,)

    monkeypatch.setattr(CallArtifacts, "stage", stage)

    async def scenario():
        async with Worker(Memory(tmp_path)) as worker:
            async with worker.subtask("fold", key=1) as sub:
                pending = asyncio.create_task(sub.complete(SlowResult()))
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                finally:
                    release.set()
                assert await asyncio.to_thread(discarded.wait, 5)
                await sub.complete(42)
            async with worker.subtask("fold", key=1) as peer:
                assert await peer.result() == 42

    asyncio.run(scenario())


def test_plain_memory_repairs_recorded_damage(tmp_path):
    """Replay an external repair committed after a failed load."""
    import json
    from pathlib import Path

    from tpcp.parallel import ClaimLostError

    memory = Memory(tmp_path)

    async def scenario():
        async with Worker(memory) as worker:
            task = CachedTask(sample)(resource="cpu")
            old = await task
            prepared = worker.prepare(task)
            decision = worker.acquire(prepared)
            assert decision.record.artifact is not None
            output = Path(prepared.artifacts.backend.location, *json.loads(decision.record.artifact), "output.pkl")
            output.write_bytes(b"broken")
            with pytest.raises(ClaimLostError):
                await worker.result(decision)
            output.unlink()
            cached: Any = memory.cache(sample)
            repaired = cached(resource="cpu")
            assert await task == repaired != old

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", [False, True])
def test_stalled_coordinator_releases_expired_capacity(tmp_path, scope):
    """A second loop can take over a stalled leaf or top-level scope with one slot."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from tpcp.parallel import ClaimLostError

    stalled, resume = Event(), Event()
    storage = InMemoryTaskStorage()
    events = []

    def on_event(event):
        events.append(event)
        if not scope and event.status == "completed" and current_thread().name.startswith("owner"):
            stalled.set()
            assert resume.wait(10)

    async def participant(owner):
        async with Worker(Memory(tmp_path), storage, n_gpu_jobs=1, lease_seconds=0.25, on_job_event=on_event) as worker:
            if scope:
                async with worker.subtask("fold", key=1) as sub:
                    assert sub.acquired
                    if owner:
                        stalled.set()
                        assert resume.wait(10)
                    await sub.complete(42)
            else:
                await Task(sample, resource="gpu")(resource="gpu")

    with ThreadPoolExecutor(1, thread_name_prefix="owner") as executor:
        pending = executor.submit(asyncio.run, participant(True))
        try:
            assert stalled.wait(10)

            async def takeover():
                await asyncio.sleep(0.35)
                await asyncio.wait_for(participant(False), 5)

            asyncio.run(takeover())
        finally:
            resume.set()
        with pytest.raises(ClaimLostError):
            pending.result()
    if not scope:
        completed = [event for event in events if event.status == "completed"]
        assert len({event.job_id for event in completed}) == 2
        assert len({event.worker.batch_id for event in completed}) == 2


def test_publication_uses_coordinator_storage_thread(tmp_path):
    """A storage transaction can use a thread-affine connection."""
    from threading import get_ident

    owner = get_ident()

    class Storage(InMemoryTaskStorage):
        def transaction(self):
            assert get_ident() == owner
            return super().transaction()

    async def scenario():
        async with Worker(Memory(tmp_path), Storage()):
            await Task(sample)(resource="cpu")

    asyncio.run(scenario())


def test_legacy_context_checks_existing_loop_adapter(tmp_path):
    """An existing Worker adapter cannot bypass single-loop context ownership."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from tpcp.parallel import execution_context

    ready, try_context, checked = Event(), Event(), Event()

    async def other_loop():
        async with Worker(Memory(tmp_path)):
            ready.set()
            assert await asyncio.to_thread(try_context.wait, 5)
            with pytest.raises(RuntimeError, match="same event loop"):
                async with execution_context():
                    pass
            checked.set()

    async def scenario():
        async with Worker(Memory(tmp_path)):
            with ThreadPoolExecutor(1) as executor:
                peer = executor.submit(asyncio.run, other_loop())
                try:
                    assert await asyncio.to_thread(ready.wait, 5)
                    async with execution_context():
                        try_context.set()
                        assert await asyncio.to_thread(checked.wait, 5)
                finally:
                    try_context.set()
                peer.result()

    asyncio.run(scenario())


def test_upgrade_existing_task_records(tmp_path):
    """Open a seven-column task database and preserve its existing rows."""
    import sqlite3
    from contextlib import closing

    from tpcp.parallel import SQLiteTaskStorage

    path = tmp_path / "tasks.sqlite"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE tasks (task_id TEXT PRIMARY KEY, status, attempt, expires, started, finished, artifact)"
        )
        connection.execute("INSERT INTO tasks VALUES ('key','completed','owner',0,0,0,'artifact')")
    with SQLiteTaskStorage(path).transaction() as records:
        record = records.get("key")
        assert record is not None
        assert record.artifact == "artifact"
        assert record.damaged_identity is None
        records.put(record)
