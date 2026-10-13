"""Integration checks for context-local async search and cross-validation."""

import asyncio
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from tests.test_pipelines.conftest import DummyDataset, DummyPipeline
from tpcp.exceptions import TestError as ScoringError
from tpcp.optimize import GridSearch
from tpcp.parallel import Job, execution_context, get_job_metadata, get_worker_context, job_metadata
from tpcp.validate import Scorer, async_cross_validate, cross_validate, cross_validate_async, no_agg


def score_candidate(pipeline, datapoint):
    """Score candidates against a known quadratic optimum."""
    return -((pipeline.para_1 - datapoint.group_label.value) ** 2)


def score_with_process_ids(pipeline, datapoint):
    """Return worker identities as ordinary scorer results."""
    time.sleep(0.02)
    return {
        "score": score_candidate(pipeline, datapoint),
        "pid": no_agg(os.getpid()),
        "parent_pid": no_agg(os.getppid()),
    }


def test_async_grid_search_populates_original_object():
    """Awaiting a search populates the original instance with the best candidate."""
    search = GridSearch(
        DummyPipeline(), ParameterGrid({"para_1": [0, 2, 4]}), scoring=score_candidate, progress_bar=False
    )

    async def run():
        async with execution_context(n_jobs=2):
            return await search.optimize_async(DummyDataset())

    assert asyncio.run(run()) is search
    assert search.best_params_ == {"para_1": 2}
    assert search.best_score_ == -2
    assert search.optimized_pipeline_.para_1 == 2
    np.testing.assert_array_equal(search.gs_results_["agg__score"], [-6, -2, -6])


@pytest.mark.parametrize("return_train_score", [False, True])
def test_async_cross_validation_matches_sync(return_train_score):
    """Nested async search preserves fold ordering, scores, labels and selected parameters."""
    search = GridSearch(
        DummyPipeline(), ParameterGrid({"para_1": range(5)}), scoring=score_candidate, progress_bar=False
    )
    kwargs = {
        "scoring": score_candidate,
        "cv": 2,
        "return_optimizer": True,
        "return_train_score": return_train_score,
    }
    expected = cross_validate(search, DummyDataset(), progress_bar=False, **kwargs)

    async def run():
        async with execution_context(n_jobs=2):
            return await cross_validate_async(search, DummyDataset(), **kwargs)

    actual = asyncio.run(run())
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "optimizer":
            assert [o.best_params_ for o in actual[key]] == [o.best_params_ for o in expected[key]]
        elif not key.startswith("debug__"):
            for actual_fold, expected_fold in zip(actual[key], expected[key], strict=True):
                np.testing.assert_array_equal(actual_fold, expected_fold)
    assert not hasattr(search, "gs_results_")  # Each fold owns an independent clone.


def test_nested_search_and_cv_use_only_the_shared_workers():
    """Both layers use the same two direct child processes despite synchronous n_jobs settings."""
    scorer = Scorer(score_with_process_ids, n_jobs=3, progress_bar=False)
    search = GridSearch(
        DummyPipeline(),
        ParameterGrid({"para_1": [0, 2, 4]}),
        scoring=scorer,
        return_optimized="score",
        n_jobs=3,
        progress_bar=False,
    )

    async def run():
        async with execution_context(n_jobs=2):
            return await cross_validate_async(search, DummyDataset(), scoring=scorer, cv=2, return_optimizer=True)

    results = asyncio.run(run())
    pids = {pid for fold in results["test__single__pid"] for pid in fold}
    parent_pids = {pid for fold in results["test__single__parent_pid"] for pid in fold}
    for optimizer in results["optimizer"]:
        pids.update(pid for candidate in optimizer.gs_results_["single__pid"] for pid in candidate)
        parent_pids.update(pid for candidate in optimizer.gs_results_["single__parent_pid"] for pid in candidate)
    assert len(pids) == 2
    assert os.getpid() not in pids
    assert parent_pids == {os.getpid()}
    assert scorer.n_jobs == search.n_jobs == 3


def test_execution_reuses_joblib_workers_and_array_serialization():
    """Async jobs reuse a warm joblib pool, cloudpickle closures and memmap large arrays."""
    data = np.ones(200_000)

    def inspect_array(array):
        time.sleep(0.05)
        return os.getpid(), isinstance(array, np.memmap), array.sum()

    before = Parallel(n_jobs=2)(delayed(inspect_array)(data) for _ in range(2))

    async def run():
        async with execution_context(n_jobs=2) as execution:
            return await asyncio.gather(*(execution.run(inspect_array, data) for _ in range(2)))

    during = asyncio.run(run())
    after = Parallel(n_jobs=2)(delayed(inspect_array)(data) for _ in range(2))
    assert {pid for pid, _, _ in before} == {pid for pid, _, _ in during} == {pid for pid, _, _ in after}
    assert all(is_memmap and total == 200_000 for _, is_memmap, total in during)


@pytest.mark.parametrize("resource", ["cpu", "gpu"])
def test_failed_search_leaves_execution_available_for_another_search(resource):
    """Worker errors propagate, while cancelled sibling jobs do not strand pool capacity."""
    bad = GridSearch(DummyPipeline(), ParameterGrid({"para_1": [None, 2]}), scoring=score_candidate, progress_bar=False)
    good = GridSearch(
        DummyPipeline(), ParameterGrid({"para_1": [0, 2, 4]}), scoring=score_candidate, progress_bar=False
    )

    async def run():
        async with execution_context(n_jobs=2, n_gpu_jobs=1):
            with pytest.raises(ExceptionGroup) as error:
                await bad.optimize_async(DummyDataset(), resource=resource)
            assert error.value.subgroup(ScoringError) is not None
            await good.optimize_async(DummyDataset(), resource=resource)

    asyncio.run(run())
    assert good.best_params_ == {"para_1": 2}


class BoundedCopyParameter:
    """Model a large parameter whose parent-process copies must stay bounded."""

    active_copies = 0

    def __init__(self):
        self.owner_pid = os.getpid()
        self.counted = False

    def __deepcopy__(self, memo):
        """Reject eager allocation across the full grid."""
        if os.getpid() == self.owner_pid:
            if type(self).active_copies >= 3:
                raise RuntimeError("Too many queued pipeline copies.")
            copy = type(self)()
            copy.counted = True
            type(self).active_copies += 1
            return copy
        return self

    def __del__(self):
        """Release the parent copy budget when a prepared job is discarded."""
        if self.counted and os.getpid() == self.owner_pid:
            type(self).active_copies -= 1


def test_candidate_pipeline_copies_are_bounded_by_admitted_work():
    """A queued grid does not allocate one full pipeline copy per candidate in the parent."""
    search = GridSearch(
        DummyPipeline(para_2=BoundedCopyParameter()),
        ParameterGrid({"para_1": range(5)}),
        scoring=score_candidate,
        return_optimized=False,
        progress_bar=False,
    )

    async def run():
        async with execution_context(n_jobs=2, batch_size=1):
            await search.optimize_async(DummyDataset())

    asyncio.run(run())
    np.testing.assert_array_equal(search.gs_results_["agg__score"], [-6, -3, -2, -3, -6])


def test_overlapping_contexts_cannot_replace_an_active_joblib_pool():
    """Reject conflicting settings before replacing a pool still borrowed by another scope."""

    async def run():
        async with execution_context(n_jobs=2) as execution:
            with pytest.raises(ValueError, match="configuration"):
                async with execution_context(n_jobs=2, max_nbytes=None):
                    pass
            assert await execution.run(lambda: 42) == 42

    asyncio.run(run())


def write_array_result(array, path, *, fail=False):
    """Record when a real process job starts and finishes using a memmapped array."""
    if fail:
        raise ValueError("Worker failed")
    Path(f"{path}.started").write_text("started")
    time.sleep(0.15)
    Path(path).write_text(str(array.sum()))


def test_context_exit_drains_calls_waiting_for_capacity(tmp_path):
    """Failure in gather still drains already accepted calls before deleting their memmaps."""
    array = np.ones(200_000)
    temp_folder = tmp_path / "memmaps"
    temp_folder.mkdir()

    async def run():
        async with execution_context(n_jobs=2, max_nbytes=1, temp_folder=temp_folder) as execution:
            await asyncio.gather(
                execution.run(write_array_result, array, tmp_path / "failed", fail=True),
                *(execution.run(write_array_result, array, tmp_path / str(i)) for i in range(5)),
            )

    with pytest.raises(ValueError, match="Worker failed"):
        asyncio.run(run())
    assert all((tmp_path / str(i)).read_text() == "200000.0" for i in range(5))
    assert not list(temp_folder.iterdir())


def test_cancellation_during_exit_waits_for_jobs_and_cleans_memmaps(tmp_path):
    """A second cancellation cannot interrupt draining and leave temporary array files behind."""
    result_path = tmp_path / "result"
    temp_folder = tmp_path / "memmaps"
    temp_folder.mkdir()

    async def run():
        exiting = asyncio.Event()

        async def owner():
            async with execution_context(n_jobs=2, max_nbytes=1, temp_folder=temp_folder) as execution:
                try:
                    await execution.run(write_array_result, np.ones(200_000), result_path)
                except asyncio.CancelledError:
                    exiting.set()
                    raise

        task = asyncio.create_task(owner())

        async def wait_for_worker():
            while not Path(f"{result_path}.started").exists():
                if task.done():
                    await task
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_worker(), timeout=10)
        task.cancel()
        await exiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert result_path.read_text() == "200000.0"
        assert not list(temp_folder.iterdir())

    asyncio.run(run())


def test_job_metadata_reaches_worker_and_events_run_on_main_thread():
    """Scoped and per-job metadata survives process serialization without leaking to the next job."""
    events = []
    main_thread = threading.get_ident()

    def on_event(event):
        assert threading.get_ident() == main_thread
        events.append(event)

    async def run():
        async with execution_context(n_jobs=1, on_job_event=on_event) as execution:
            with job_metadata(experiment="nested-search"):
                metadata = await execution.run(Job(get_job_metadata, metadata={"cv_fold": 2}))
            next_metadata = await execution.run(get_job_metadata)
            return metadata, next_metadata

    metadata, next_metadata = asyncio.run(run())
    assert metadata == {"experiment": "nested-search", "cv_fold": 2}
    assert next_metadata == {}
    first_job = [event for event in events if event.job_id == 0]
    assert [event.status for event in first_job] == ["queued", "started", "completed"]
    assert all(event.metadata == metadata for event in first_job)


def test_nested_progress_bars_and_fold_candidate_metadata(capsys):
    """CV and candidate progress is visible, and job events identify the fold and candidate."""
    events = []
    search = GridSearch(DummyPipeline(), ParameterGrid({"para_1": [0, 2, 4]}), scoring=score_candidate)

    async def run():
        async with execution_context(n_jobs=2, on_job_event=events.append):
            return await cross_validate_async(search, DummyDataset(), scoring=score_candidate, cv=2, progress_bar=True)

    asyncio.run(run())
    output = capsys.readouterr().err
    assert "CV Folds" in output
    assert "Parameter Combinations" in output
    assert "100%" in output
    candidate_events = [e for e in events if e.status == "completed" and e.metadata.get("stage") == "candidate_score"]
    assert {(e.metadata["cv_fold"], e.metadata["parameter_candidate"]) for e in candidate_events} == {
        (fold, candidate) for fold in range(2) for candidate in range(3)
    }


def worker_interval(label):
    """Record actual process execution intervals for resource scheduling checks."""
    start = time.monotonic()
    time.sleep(0.15)
    return label, start, time.monotonic()


@pytest.mark.parametrize("gpu_slots", [1, 2])
def test_gpu_jobs_take_gpu_and_cpu_slots_without_blocking_waiting_workers(gpu_slots):
    """GPU waits do not consume CPU capacity, while running GPU jobs share the CPU budget."""
    # Warm both reusable workers so import timing cannot disguise scheduling.
    Parallel(n_jobs=2)(delayed(worker_interval)("warm") for _ in range(2))

    async def run():
        async with execution_context(n_jobs=2, n_gpu_jobs=gpu_slots) as execution:
            return await asyncio.gather(
                execution.run(Job(worker_interval, args=("gpu-a",), resource="gpu")),
                execution.run(Job(worker_interval, args=("gpu-b",), resource="gpu")),
                execution.run(Job(worker_interval, args=("cpu",))),
            )

    intervals = {label: (start, end) for label, start, end in asyncio.run(run())}

    def overlaps(first, second):
        return max(first[0], second[0]) < min(first[1], second[1])

    if gpu_slots == 1:
        assert not overlaps(intervals["gpu-a"], intervals["gpu-b"])
        assert overlaps(intervals["gpu-a"], intervals["cpu"])
    else:
        assert overlaps(intervals["gpu-a"], intervals["gpu-b"])
        assert not (overlaps(intervals["gpu-a"], intervals["cpu"]) and overlaps(intervals["gpu-b"], intervals["cpu"]))


def test_nested_grid_search_can_schedule_all_evaluations_as_gpu_jobs():
    """The async action option marks candidate and held-out scoring jobs for the GPU queue."""
    events = []
    search = GridSearch(
        DummyPipeline(), ParameterGrid({"para_1": [0, 2, 4]}), scoring=score_candidate, progress_bar=False
    )

    async def run():
        async with execution_context(n_jobs=2, n_gpu_jobs=1, on_job_event=events.append):
            return await cross_validate_async(
                search, DummyDataset(), scoring=score_candidate, cv=2, resource="gpu", progress_bar=False
            )

    results = asyncio.run(run())
    assert len(results["test__agg__score"]) == 2
    completed = [event for event in events if event.status == "completed"]
    assert len(completed) == 8  # Six candidates and two held-out scoring jobs.
    assert all(event.resource == "gpu" for event in completed)


def test_previously_run_pipeline_results_are_not_serialized_for_candidates():
    """Cloning candidates strips transient unpicklable results before process submission."""
    pipeline = DummyPipeline()
    pipeline.transient_result_ = (i for i in range(3))
    search = GridSearch(pipeline, ParameterGrid({"para_1": [0, 2, 4]}), scoring=score_candidate, progress_bar=False)

    async def run():
        async with execution_context(n_jobs=2):
            await search.optimize_async(DummyDataset())

    asyncio.run(run())
    assert search.best_params_ == {"para_1": 2}


def test_overlapping_contexts_on_different_event_loops_are_rejected():
    """Loop-bound scheduling state is never shared across concurrent asyncio.run callers."""

    async def other_loop():
        async with execution_context(n_jobs=2):
            pass

    async def run():
        async with execution_context(n_jobs=2) as execution:
            with pytest.raises(RuntimeError, match="event loop"):
                await asyncio.to_thread(asyncio.run, other_loop())
            assert await execution.run(lambda: 42) == 42

    asyncio.run(run())


def test_asyncio_runner_shutdown_cannot_cancel_resource_cleanup(tmp_path):
    """Runner-wide task cancellation still finishes process jobs and releases their memmaps."""
    path = tmp_path / "result"
    temp_folder = tmp_path / "memmaps"
    temp_folder.mkdir()

    async def run():
        exiting = asyncio.Event()

        async def owner():
            async with execution_context(n_jobs=2, max_nbytes=1, temp_folder=temp_folder) as execution:
                try:
                    await execution.run(write_array_result, np.ones(200_000), path)
                except asyncio.CancelledError:
                    exiting.set()
                    raise

        task = asyncio.create_task(owner())
        while not Path(f"{path}.started").exists():
            if task.done():
                await task
            await asyncio.sleep(0.01)
        task.cancel()
        await exiting.wait()
        # Returning lets asyncio.run cancel every outstanding Task, including
        # any cleanup tasks created by the implementation.

    asyncio.run(run())
    finished_on_exit = path.exists()
    # Keep the fixture alive until a failing implementation's worker finishes.
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert finished_on_exit
    assert not list(temp_folder.iterdir())


def test_worker_context_is_available_and_can_be_injected():
    """Each call receives its worker identity and assigned GPU, without leaking GPU state to CPU jobs."""
    events = []

    async def run():
        async with execution_context(n_jobs=1, n_gpu_jobs=1, gpu_ids=(7,), on_job_event=events.append) as execution:
            gpu_context = await execution.run(
                Job(lambda ctx: ctx, pass_context=True, resource="gpu", metadata={"fold": 2})
            )
            cpu_context = await execution.run(get_worker_context)
            return gpu_context, cpu_context

    gpu, cpu = asyncio.run(run())
    assert gpu.worker_id == cpu.worker_id
    assert gpu.pid == cpu.pid != os.getpid()
    assert gpu.gpu_id == 7
    assert cpu.gpu_id is None
    assert gpu.metadata == {"fold": 2}
    assert cpu.metadata == {}
    assert [event.worker for event in events if event.status == "completed"] == [gpu, cpu]


def test_fixed_batches_keep_individual_contexts_and_results():
    """A fixed batch groups calls while preserving individual job IDs, metadata and result order."""

    async def run():
        async with execution_context(n_jobs=1, batch_size=3) as execution:
            return await asyncio.gather(
                *(execution.run(Job(get_worker_context, metadata={"item": i})) for i in range(7))
            )

    contexts = asyncio.run(run())
    assert [context.metadata["item"] for context in contexts] == list(range(7))
    batches = {}
    for context in contexts:
        batches.setdefault(context.batch_id, []).append(context)
    assert [len(batch) for batch in batches.values()] == [3, 3, 1]
    assert [[context.batch_index for context in batch] for batch in batches.values()] == [[0, 1, 2], [0, 1, 2], [0]]


def test_auto_batching_adapts_and_keeps_failures_independent():
    """Joblib's adaptive batching groups fast jobs and a failed call does not fail its batch siblings."""
    Parallel(n_jobs=2)(delayed(worker_interval)("warm") for _ in range(2))

    async def run():
        async with execution_context(n_jobs=2, batch_size="auto") as execution:
            contexts = await asyncio.gather(*(execution.run(get_worker_context) for _ in range(64)))
        async with execution_context(n_jobs=1, batch_size=3) as execution:
            results = await asyncio.gather(
                *(execution.run(lambda divisor: 10 // divisor, divisor) for divisor in [2, 0, 5]),
                return_exceptions=True,
            )
        return contexts, results

    contexts, results = asyncio.run(run())
    assert max(context.batch_size for context in contexts) > 1
    assert results[0] == 5
    assert isinstance(results[1], ZeroDivisionError)
    assert results[2] == 2


def test_action_effects_bind_calls_and_declare_resources():
    """An action chooses its queue before callers execute the returned effect."""
    from tpcp.parallel import Task

    class Algorithm:
        def async_optimize(self, value):
            return Task(lambda bound: (bound, get_worker_context()), args=(value,), resource="gpu")

    effect = Algorithm().async_optimize(42)
    assert effect.config.kind == "worker"
    assert effect.config.resource == "gpu"

    async def run():
        async with execution_context(n_jobs=2, n_gpu_jobs=1, gpu_ids=(3,)):
            return await effect

    value, worker = asyncio.run(run())
    assert value == 42
    assert worker.gpu_id == 3
    assert worker.pid != os.getpid()


def test_workflow_effects_are_lazy_and_compose_on_main_thread():
    """Constructing nested effects leaves results untouched until execution."""
    from tpcp.parallel import Task

    search = GridSearch(
        DummyPipeline(), ParameterGrid({"para_1": [0, 2, 4]}), scoring=score_candidate, progress_bar=False
    )
    dataset = DummyDataset()
    effect = search.async_optimize(dataset)
    assert effect.config.kind == "workflow"
    assert not hasattr(search, "gs_results_")

    async def compose():
        result = await effect
        return result, os.getpid()

    workflow = Task(compose, kind="workflow")

    async def run():
        async with execution_context(n_jobs=2):
            return await workflow

    result, pid = asyncio.run(run())
    assert result is search
    assert pid == os.getpid()
    assert search.best_params_ == {"para_1": 2}
    assert (
        async_cross_validate(search, dataset, scoring=score_candidate, cv=2, progress_bar=False).config.kind
        == "workflow"
    )


def test_nested_action_owns_its_gpu_requirement():
    """CV executes the callee's effect without choosing its candidate queue."""

    class GpuSearch(GridSearch):
        def async_optimize(self, dataset, **kwargs):
            return super().async_optimize(dataset, **{**kwargs, "resource": "gpu"})

    search = GpuSearch(DummyPipeline(), ParameterGrid({"para_1": [0, 2]}), scoring=score_candidate, progress_bar=False)
    events = []

    async def run():
        async with execution_context(n_jobs=2, n_gpu_jobs=1, on_job_event=events.append):
            return await cross_validate_async(search, DummyDataset(), scoring=score_candidate, cv=2, progress_bar=False)

    results = asyncio.run(run())
    assert len(results["test__agg__score"]) == 2
    candidates = [
        event for event in events if event.status == "completed" and event.metadata["stage"] == "candidate_score"
    ]
    assert len(candidates) == 4
    assert all(event.resource == "gpu" and event.worker.gpu_id == 0 for event in candidates)


def test_effects_can_be_gathered_directly():
    """Independent effects compose in gather, with identity-based deduplication."""
    from tpcp.parallel import Task

    events = []

    def leaf():
        return get_worker_context().job_id

    async def workflow():
        return await Task(leaf)

    first = Task(leaf)
    second = Task(leaf)

    async def run():
        async with execution_context(n_jobs=2, on_job_event=events.append):
            return await asyncio.gather(first, first, second, Task(workflow, kind="workflow"))

    results = asyncio.run(run())
    assert results[0] == results[1]
    assert len(set(results)) == 3
    assert len([event for event in events if event.status == "completed"]) == 3


def test_coroutine_interfaces_support_create_task():
    """The previously agreed suffix interfaces remain coroutine entry points."""
    search = GridSearch(DummyPipeline(), ParameterGrid({"para_1": [0, 2]}), scoring=score_candidate, progress_bar=False)

    async def run():
        async with execution_context(n_jobs=2):
            await asyncio.create_task(search.optimize_async(DummyDataset()))
            async with asyncio.TaskGroup() as group:
                cv_task = group.create_task(
                    cross_validate_async(search, DummyDataset(), scoring=score_candidate, cv=2, progress_bar=False)
                )
            return cv_task.result()

    assert len(asyncio.run(run())["test__agg__score"]) == 2


def test_cross_validation_preserves_coroutine_override_dispatch():
    """An existing override need not accept new scheduling keywords."""
    calls = []

    class CustomSearch(GridSearch):
        async def optimize_async(self, dataset):
            calls.append(dataset.group_labels)
            return await super().optimize_async(dataset)

    search = CustomSearch(
        DummyPipeline(), ParameterGrid({"para_1": [0, 2]}), scoring=score_candidate, progress_bar=False
    )

    async def run():
        async with execution_context(n_jobs=2):
            return await cross_validate_async(search, DummyDataset(), scoring=score_candidate, cv=2, progress_bar=False)

    assert len(asyncio.run(run())["test__agg__score"]) == 2
    assert len(calls) == 2
