"""Asynchronous optimization actions."""

from typing import Any, Literal, Self

from tpcp._async_execution import _gather_with_progress, _LocalWorkflow, _optimization_resource
from tpcp._dataset import DatasetT
from tpcp._pipeline import PipelineT
from tpcp._utils._score import _score
from tpcp.aio.parallel import BoundTask, Task, get_job_metadata, job_metadata
from tpcp.misc import warning_error_context
from tpcp.optimize._optimize import _GridSearch
from tpcp.validate._scorer import _validate_scorer

__all__ = ["GridSearch"]


def _score_candidate(pipeline, dataset, scoring, parameters):
    return _score(
        pipeline.clone(),
        dataset,
        scoring,
        parameters,
        return_parameters=True,
        return_times=True,
    )


class GridSearch(_GridSearch[PipelineT, DatasetT]):
    """Search a parameter grid using the current async worker pool.

    Configuration and results follow :class:`tpcp.optimize.GridSearch`.
    Set concurrency on the execution context or worker.
    """

    def optimize(
        self, dataset: DatasetT, *, resource: Literal["cpu", "gpu"] | None = None, **kwargs: Any
    ) -> BoundTask[Self]:
        """Bind an async optimization workflow without executing it.

        Await the returned effect inside :func:`tpcp.parallel.execution_context`.
        Candidates share its pool and run serial scoring within each worker.
        ``resource="gpu"`` requests both GPU and CPU slots. ``progress_bar``
        shows candidate completion; synchronous ``n_jobs`` is ignored. Selection
        and result attributes are applied to this instance on the main thread.
        """
        if resource is None:
            resource = _optimization_resource.get()
        return _LocalWorkflow(self._run_search)(dataset, resource=resource, **kwargs)

    async def _run_search(self, dataset: DatasetT, *, resource: Literal["cpu", "gpu"] = "cpu", **_: Any) -> Self:
        """Run experimental async GridSearch in the active execution context.

        Candidate evaluations share the context's pool with enclosing async CV
        folds. The execution context controls pool concurrency; synchronous
        ``n_jobs`` and ``pre_dispatch`` settings are ignored. ``progress_bar``
        displays candidate completions using tqdm.
        ``resource="gpu"`` requires both a GPU slot and a CPU/process slot.
        Scorers run serially inside each candidate job to avoid nested pools.
        Results are stored on this instance in the calling event loop.
        Failures propagate as an ``ExceptionGroup`` from the task group.
        """
        self.dataset = dataset
        scoring = _validate_scorer(self.scoring).clone().set_params(n_jobs=1, progress_bar=False)

        pipeline = self.pipeline.clone()

        # Pandas creates the label namedtuple class at runtime. Keep it out of persisted scores.
        labels = dataset.group_labels

        async def evaluate(index, parameters):
            with (
                job_metadata(parameter_candidate=index, parameters=parameters, stage="candidate_score"),
                warning_error_context("parameter_candidate", {"i": index, "parameters": parameters}),
            ):
                result = await Task(_score_candidate, resource=resource)(pipeline, dataset, scoring, parameters)
                return {**result, "data_labels": list(labels)}

        fold = get_job_metadata().get("cv_fold")
        desc = "Parameter Combinations" if fold is None else f"Parameter Combinations [fold {fold}]"
        results = await _gather_with_progress(
            (evaluate(i, paras) for i, paras in enumerate(self.parameter_grid)),
            total=len(self.parameter_grid),
            desc=desc,
            progress_bar=self.progress_bar,
        )
        return self._set_search_results(results)
