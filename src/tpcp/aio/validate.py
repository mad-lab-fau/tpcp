"""Asynchronous pipeline validation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal, cast

from tpcp._async_execution import _gather_with_progress, _LocalWorkflow, _optimization_resource
from tpcp._utils._score import _score
from tpcp.aio.optimize import optimize
from tpcp.misc import warning_error_context
from tpcp.parallel import BoundTask, Task, job_metadata
from tpcp.validate._cross_val_helper import _normalize_splitter
from tpcp.validate._scorer import _validate_scorer
from tpcp.validate._validate import _format_cv_results

if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator

    from tpcp import Dataset
    from tpcp._dataset import DatasetT
    from tpcp._pipeline import PipelineT
    from tpcp.optimize import GridSearch
    from tpcp.validate._cross_val_helper import BaseDatasetSplitter
    from tpcp.validate._scorer import ScorerTypes

__all__ = ["cross_validate"]


def cross_validate(
    optimizable: GridSearch[PipelineT, DatasetT],
    dataset: DatasetT,
    *,
    scoring: ScorerTypes[PipelineT, DatasetT],
    cv: BaseDatasetSplitter | int | BaseCrossValidator | list | None = None,
    optimize_params: dict[str, Any] | None = None,
    return_train_score: bool = False,
    return_optimizer: bool = False,
    progress_bar: bool = True,
    resource: Literal["cpu", "gpu"] = "cpu",
) -> BoundTask:
    """Bind a lazy cross-validation workflow; await it inside an execution context.

    Fold workflows and GridSearch candidates share one process pool. Search
    selection and results stay on the main thread; scoring runs in workers.
    ``resource="gpu"`` requests both CPU and GPU slots for leaf jobs.
    ``progress_bar`` shows fold completion; candidate bars follow the search's
    settings. Scorers run serially within each job. Timings include queue waits.
    See :func:`tpcp.validate.cross_validate` for result fields. Failures from composed jobs
    propagate as an ``ExceptionGroup``.
    """
    return _LocalWorkflow(_cross_validate_async)(
        optimizable,
        dataset,
        scoring=scoring,
        cv=cv,
        optimize_params=optimize_params,
        return_train_score=return_train_score,
        return_optimizer=return_optimizer,
        progress_bar=progress_bar,
        resource=resource,
    )


async def _cross_validate_async(
    optimizable: GridSearch[PipelineT, DatasetT],
    dataset: DatasetT,
    *,
    scoring: ScorerTypes[PipelineT, DatasetT],
    cv: BaseDatasetSplitter | int | BaseCrossValidator | list | None = None,
    optimize_params: dict[str, Any] | None = None,
    return_train_score: bool = False,
    return_optimizer: bool = False,
    progress_bar: bool = True,
    resource: Literal["cpu", "gpu"] = "cpu",
):
    """Evaluate GridSearch folds through one experimental async execution context.

    Use inside ``async with tpcp.parallel.execution_context(n_jobs=...)``.
    Each fold awaits its own cloned GridSearch, which submits candidate jobs to
    the shared pool. Test and optional train scoring use the same pool, with
    datapoints scored serially inside each job. Results have the same structure
    as :func:`tpcp.validate.cross_validate`; timings measure elapsed time including pool waits.
    ``progress_bar`` displays completed folds using tqdm. Inner search progress
    follows ``GridSearch.progress_bar``. Scorer ``n_jobs`` and datapoint progress
    bars do not apply to this prototype.
    ``resource="gpu"`` schedules candidate and held-out scoring through the GPU
    gate. ``optimize_params`` can override the resource for optimization only.
    Failures propagate as an ``ExceptionGroup`` from the task group.
    """
    scorer = _validate_scorer(scoring).clone().set_params(n_jobs=1, progress_bar=False)
    splits = list(_normalize_splitter(cv).split(cast("Dataset", dataset)))

    async def evaluate_fold(index, train_labels, test_labels):
        optimizer = optimizable.clone()
        train = dataset.get_subset(group_labels=train_labels)
        test = dataset.get_subset(group_labels=test_labels)
        with job_metadata(cv_fold=index), warning_error_context("cv_fold", {"i": index}):
            start = time.perf_counter()
            with warning_error_context("optimize", {"data_labels": train.group_labels}):
                parameters = dict(optimize_params or {})
                token = _optimization_resource.set(resource)
                try:
                    optimizer = await optimize(optimizer, train, **parameters)
                finally:
                    _optimization_resource.reset(token)
            optimize_time = time.perf_counter() - start
            score_start = time.perf_counter()
            result: dict[str, Any] = {"debug__optimize_time": optimize_time}
            scoring_sets = [("test", test)]
            if return_train_score:
                scoring_sets.append(("train", train))
            for prefix, subset in scoring_sets:
                with (
                    job_metadata(stage=f"{prefix}_score"),
                    warning_error_context(f"{prefix}_score", {"data_labels": subset.group_labels}),
                ):
                    scores = cast(
                        "dict[str, Any]",
                        await Task(_score, resource=resource)(
                            optimizer.optimized_pipeline_, cast("Dataset", subset), scorer, None
                        ),
                    )
                result[f"{prefix}__scores"] = scores["scores"]
                result[f"{prefix}__single__scores"] = scores["single__scores"]
            result["debug__score_time"] = time.perf_counter() - score_start
            result["train__data_labels"] = train.group_labels
            result["test__data_labels"] = test.group_labels
            if return_optimizer:
                result["optimizer"] = optimizer
            return result

    results = await _gather_with_progress(
        (evaluate_fold(i, train, test) for i, (train, test) in enumerate(splits)),
        total=len(splits),
        desc="CV Folds",
        progress_bar=progress_bar,
    )
    return _format_cv_results(results)
