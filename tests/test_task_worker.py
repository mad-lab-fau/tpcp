"""Coordination through public Worker and joblib interfaces."""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

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
            Worker(memory, store, lease_seconds=0.3) as first,
            Worker(memory, store) as second,
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
