import asyncio
import uuid

from joblib import Memory

from tpcp.parallel import CachedTask, InMemoryTaskStorage, Task, Worker


def sample(*, resource):
    return resource, uuid.uuid4().hex


def test_scoped_and_cached_calls(tmp_path):
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
                assert await cached == shared == memory.cache(sample)(resource="argument")

    asyncio.run(scenario())


def test_cached_artifact_removal_and_existing_joblib_hit(tmp_path):
    memory = Memory(tmp_path, verbose=0)
    cached = memory.cache(sample)
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
    from tpcp.parallel import ClaimLostError
    import pytest

    async def scenario():
        store = InMemoryTaskStorage()
        memory = Memory(tmp_path, verbose=0)
        async with Worker(memory, store, lease_seconds=0.3) as first, Worker(memory, store) as second:
            # Low-level claims do not heartbeat until run. Expire the old scope
            # while its body is suspended to model a stalled participant.
            async with first.subtask("fold", key=1) as old:
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
    if not path.exists():
        path.touch()
        raise ValueError("first attempt")
    return 42


def test_failed_call_can_retry(tmp_path):
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
        time.sleep(.01)
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
                    await asyncio.sleep(.01)
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
