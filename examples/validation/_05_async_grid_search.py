"""
Async GridSearch inside cross-validation
========================================

This experimental example coordinates CV folds and parameter candidates on the
main thread. Candidate evaluation and held-out scoring share one process pool.
Run it with ``uv run python examples/validation/_05_async_grid_search.py``.

The execution context sets the worker count. Synchronous ``n_jobs`` settings are
ignored by async actions, and datapoints within each worker job run serially.
"""  # noqa: D205, D400

import asyncio
import os

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tpcp import Dataset, Pipeline
from tpcp.optimize import GridSearch
from tpcp.parallel import Task, execution_context, get_worker_context
from tpcp.validate import async_cross_validate, no_agg


class NumberDataset(Dataset):
    """Five targets for a constant prediction pipeline."""

    def create_index(self):
        """Create one datapoint per target."""
        return pd.DataFrame({"target": range(5)})


class ConstantPipeline(Pipeline):
    """Predict the same value for every datapoint."""

    def __init__(self, prediction=0):
        self.prediction = prediction

    def run(self, datapoint):  # noqa: ARG002
        """Store the prediction as an action result."""
        self.prediction_ = self.prediction
        return self


def score(pipeline, datapoint):
    """Return negative squared error and the worker process ID."""
    prediction = pipeline.safe_run(datapoint).prediction_
    return {
        "score": -((prediction - datapoint.group_label.target) ** 2),
        "worker_pid": no_agg(os.getpid()),
    }


async def main():
    """Evaluate all folds using a shared pool of two workers."""
    search = GridSearch(
        ConstantPipeline(),
        ParameterGrid({"prediction": range(5)}),
        scoring=score,
        return_optimized="score",
        progress_bar=True,
    )
    # Binding an effect is lazy; its config can be inspected before execution.
    effect = async_cross_validate(
        search, NumberDataset(), scoring=score, cv=2, return_optimizer=True
    )
    print("Workflow config:", effect.config)
    events = []
    async with execution_context(
        n_jobs=2, n_gpu_jobs=1, batch_size="auto", on_job_event=events.append
    ):
        results = await effect
        # This simulates a GPU action without requiring GPU hardware. A real
        # action would use the assigned ID for device placement.
        gpu_effect = Task(get_worker_context, resource="gpu")
        gpu_worker = await gpu_effect
    print("GPU slot context:", gpu_worker)
    print("Completed jobs:", sum(e.status == "completed" for e in events))

    print("Main process:", os.getpid())
    print(
        "Best parameters per fold:",
        [optimizer.best_params_ for optimizer in results["optimizer"]],
    )
    print("Test scores:", results["test__agg__score"])
    worker_pids = {
        pid
        for optimizer in results["optimizer"]
        for candidate in optimizer.gs_results_["single__worker_pid"]
        for pid in candidate
    }
    print("Shared worker processes:", sorted(worker_pids))


if __name__ == "__main__":
    asyncio.run(main())
