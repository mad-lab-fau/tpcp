"""Asynchronous pipeline validation."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal, cast

from tqdm.auto import tqdm

from tpcp._async_execution import _gather_with_progress, _LocalWorkflow, _optimization_resource
from tpcp._dataset import DatasetT
from tpcp._pipeline import PipelineT
from tpcp._utils._score import _score
from tpcp.aio._pipeline import Pipeline as AsyncPipeline
from tpcp.misc import warning_error_context
from tpcp.parallel import BoundTask, Task, job_metadata
from tpcp.validate._cross_val_helper import _normalize_splitter
from tpcp.validate._scorer import Scorer as SyncScorer
from tpcp.validate._scorer import _aggregate_scores, _record_score, _score_datapoint, _Scorer, _validate_scorer
from tpcp.validate._validate import _format_cv_results, _format_validation_results

if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator

    from tpcp import Dataset
    from tpcp.aio.optimize import BaseOptimize
    from tpcp.validate._cross_val_helper import BaseDatasetSplitter
    from tpcp.validate._scorer import ScorerTypes

__all__ = ["Scorer", "cross_validate", "validate"]


def cross_validate(
    optimizable: BaseOptimize[PipelineT, DatasetT],
    dataset: DatasetT,
    *,
    scoring: ScorerTypes[PipelineT, DatasetT] | Scorer[PipelineT, DatasetT],
    cv: BaseDatasetSplitter | int | BaseCrossValidator | list | None = None,
    optimize_params: dict[str, Any] | None = None,
    return_train_score: bool = False,
    return_optimizer: bool = False,
    progress_bar: bool = True,
    resource: Literal["cpu", "gpu"] = "cpu",
) -> BoundTask:
    """Bind a lazy cross-validation workflow; await it inside an execution context.

    Fold workflows and optimizer jobs share one process pool. Search
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
    optimizable: BaseOptimize[PipelineT, DatasetT],
    dataset: DatasetT,
    *,
    scoring: ScorerTypes[PipelineT, DatasetT] | Scorer[PipelineT, DatasetT],
    cv: BaseDatasetSplitter | int | BaseCrossValidator | list | None = None,
    optimize_params: dict[str, Any] | None = None,
    return_train_score: bool = False,
    return_optimizer: bool = False,
    progress_bar: bool = True,
    resource: Literal["cpu", "gpu"] = "cpu",
):
    """Clone each optimizer, await its action and score the pipeline it returns."""
    scorer = _serial_scorer(optimizable.pipeline, scoring)
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
                    optimizer = await optimizer.optimize(train, **parameters)
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


class Scorer(_Scorer[PipelineT, DatasetT]):
    """Score independent pipeline clones, then process results in dataset order."""

    def __call__(self, pipeline: PipelineT, dataset: DatasetT, *, resource="cpu") -> BoundTask:
        """Bind per-datapoint scoring and ordered aggregation."""
        return _LocalWorkflow(_score_dataset)(self, pipeline, dataset, resource)


async def _score_dataset(scorer, pipeline, dataset, resource):
    _require_sync_pipeline(pipeline)
    scores = []
    with tqdm(total=len(dataset), desc="Datapoints", disable=not scorer.progress_bar) as bar:

        async def score(index, datapoint):
            result = await Task(_score_datapoint, resource=resource)(scorer.score_func, pipeline, index, datapoint)
            bar.update()
            return result

        tasks = []
        try:
            for index, datapoint in enumerate(dataset):
                tasks.append(asyncio.create_task(score(index, datapoint)))
            for task in tasks:
                try:
                    result = await task
                except Exception as error:  # noqa: BLE001 - preserve the workflow exception group
                    raise ExceptionGroup("Datapoint scoring failed", [error]) from None
                _record_score(scorer, pipeline, dataset, scores, result)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return _aggregate_scores(scorer, pipeline, dataset, scores)


def _require_sync_pipeline(pipeline):
    from tpcp.aio.optimize import BaseOptimize

    if isinstance(pipeline, (AsyncPipeline, BaseOptimize)):
        raise TypeError(
            "This operation requires a synchronous pipeline. Compose native async pipeline actions directly."
        )


def _serial_scorer(pipeline, scoring):
    _require_sync_pipeline(pipeline)
    if type(scoring) is Scorer:
        scoring = SyncScorer(**scoring.get_params(deep=False))
    elif isinstance(scoring, Scorer):
        raise TypeError("Use a synchronous Scorer subclass for scoring inside worker jobs.")
    return _validate_scorer(scoring).clone().set_params(n_jobs=1, progress_bar=False)


def validate(
    pipeline: PipelineT,
    dataset: DatasetT,
    *,
    scoring: ScorerTypes[PipelineT, DatasetT] | Scorer[PipelineT, DatasetT],
    progress_bar: bool | None = None,
    resource: Literal["cpu", "gpu"] = "cpu",
) -> BoundTask:
    """Evaluate a synchronous pipeline and return the regular validation result fields."""
    return _LocalWorkflow(_validate)(pipeline, dataset, scoring, resource, progress_bar)


async def _validate(pipeline, dataset, scoring, resource, progress_bar):
    _require_sync_pipeline(pipeline)
    if progress_bar is not None:
        if isinstance(scoring, _Scorer):
            raise ValueError("Set progress_bar on the supplied Scorer.")
        scoring = Scorer(scoring, progress_bar=progress_bar)
    start = time.perf_counter()
    if isinstance(scoring, SyncScorer) and type(scoring) is not SyncScorer:
        result = await Task(_score, resource=resource)(
            pipeline.clone(), dataset, scoring.clone().set_params(n_jobs=1), None, return_times=True
        )
    else:
        pipeline = pipeline.clone()
        if isinstance(scoring, Scorer):
            agg, single = await scoring(pipeline, dataset, resource=resource)
        else:
            agg, single = await _score_dataset(_validate_scorer(scoring), pipeline, dataset, resource)
        result = {"scores": agg, "single__scores": single, "debug__score_time": time.perf_counter() - start}
    return _format_validation_results({**result, "data_labels": dataset.group_labels})
