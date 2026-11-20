"""Asynchronous optimization actions."""

import time
from typing import Any, Literal, Self

from joblib import Memory

from tpcp._async_execution import _gather_with_progress, _LocalWorkflow, _optimization_resource
from tpcp._dataset import DatasetT
from tpcp._hash import custom_hash
from tpcp._optimize import _Optimizer
from tpcp._pipeline import OptimizablePipelineT, PipelineT
from tpcp._utils._general import _prefix_para_dict
from tpcp._utils._score import _cached_optimize, _clone_parameter_dict, _score
from tpcp.aio._pipeline import Algorithm
from tpcp.aio.parallel import BoundTask, Task, get_job_metadata, job_metadata
from tpcp.aio.validate import _require_sync_pipeline, _serial_scorer
from tpcp.misc import warning_error_context
from tpcp.optimize import DummyOptimize as SyncDummyOptimize
from tpcp.optimize import Optimize as SyncOptimize
from tpcp.optimize._optimize import _DummyOptimize, _GridSearch, _GridSearchCV, _Optimize

__all__ = ["BaseOptimize", "DummyOptimize", "GridSearch", "GridSearchCV", "Optimize"]


class BaseOptimize(_Optimizer[PipelineT, DatasetT], Algorithm):
    """Base for optimizers whose optimize action binds a task."""

    def optimize(self, dataset: DatasetT, **kwargs: Any) -> BoundTask[Self]:
        """Bind the optimization action for this dataset."""
        raise NotImplementedError

    def run(self, datapoint: DatasetT, *, resource: Literal["cpu", "gpu"] = "cpu") -> BoundTask[PipelineT]:
        """Bind prediction with the fitted pipeline."""
        return self.action(self._predict, datapoint, resource, False)

    def safe_run(self, datapoint: DatasetT, *, resource: Literal["cpu", "gpu"] = "cpu") -> BoundTask[PipelineT]:
        """Bind prediction with the fitted pipeline's action checks."""
        return self.action(self._predict, datapoint, resource, True)

    async def _predict(self, datapoint, resource, safe):
        method = self.optimized_pipeline_.safe_run if safe else self.optimized_pipeline_.run
        self.optimized_pipeline_ = await Task(method, resource=resource)(datapoint)
        return self.optimized_pipeline_


def _fit(optimizer, dataset, params):
    fitted = optimizer.optimize(dataset, **params)
    state = {"optimized_pipeline_": fitted.optimized_pipeline_, "transformed_dataset_": fitted.transformed_dataset_}
    if hasattr(fitted, "optimization_info_"):
        state["optimization_info_"] = fitted.optimization_info_
    return state


class DummyOptimize(_DummyOptimize[PipelineT, DatasetT], BaseOptimize[PipelineT, DatasetT]):
    """Clone the pipeline without fitting it."""

    def optimize(self, dataset: DatasetT, **kwargs: Any) -> BoundTask[Self]:
        """Bind a workflow that clones the configured pipeline."""
        return _LocalWorkflow(self._run)(dataset, kwargs)

    async def _run(self, dataset, kwargs):
        _require_sync_pipeline(self.pipeline)
        self.dataset = dataset
        fitted = SyncDummyOptimize(**self.get_params(deep=False)).optimize(dataset, **kwargs)
        self.optimized_pipeline_ = fitted.optimized_pipeline_
        return self


class Optimize(_Optimize[OptimizablePipelineT, DatasetT], BaseOptimize[OptimizablePipelineT, DatasetT]):
    """Fit a synchronous pipeline in one worker slot."""

    def optimize(
        self, dataset: DatasetT, *, resource: Literal["cpu", "gpu"] | None = None, **kwargs: Any
    ) -> BoundTask[Self]:
        """Bind a worker fit and apply its results to this instance when awaited."""
        return _LocalWorkflow(self._run)(dataset, resource or _optimization_resource.get(), kwargs)

    async def _run(self, dataset, resource, kwargs):
        _require_sync_pipeline(self.pipeline)
        state = await Task(_fit, resource=resource)(SyncOptimize(**self.get_params(deep=False)), dataset, kwargs)
        self.dataset = dataset
        vars(self).pop("optimization_info_", None)
        vars(self).update(state)
        return self


def _score_candidate(pipeline, dataset, scoring, parameters):
    return _score(
        pipeline.clone(),
        dataset,
        scoring,
        parameters,
        return_parameters=True,
        return_times=True,
    )


class GridSearch(_GridSearch[PipelineT, DatasetT], BaseOptimize[PipelineT, DatasetT]):
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
        scoring = _serial_scorer(self.pipeline, self.scoring)

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


def _fit_fold(optimizer, train, hyper, pure_names, kwargs):
    start = time.perf_counter()
    fitted = _cached_optimize(
        optimizer,
        train,
        _clone_parameter_dict(_prefix_para_dict(hyper)),
        dict.fromkeys(pure_names),
        Memory(None),
        kwargs,
    )
    return fitted.optimized_pipeline_, time.perf_counter() - start


def _score_fold(pipeline, pure, scorer, train, test, return_train):
    pipeline = pipeline.clone().set_params(**_clone_parameter_dict(pure))
    start = time.perf_counter()

    def score(prefix, subset):
        values = _score(pipeline, subset, scorer, None)
        return {f"{prefix}__{key}": value for key, value in values.items()}

    result = score("test", test)
    result["debug__score_time"] = time.perf_counter() - start
    if return_train:
        result.update(score("train", train))
    return result


class GridSearchCV(_GridSearchCV[OptimizablePipelineT, DatasetT], BaseOptimize[OptimizablePipelineT, DatasetT]):
    """Score candidate/fold pairs on the shared worker pool and refit the winner."""

    def optimize(
        self, dataset: DatasetT, *, resource: Literal["cpu", "gpu"] | None = None, **kwargs: Any
    ) -> BoundTask[Self]:
        """Bind fold fits, candidate scoring and optional final refitting."""
        return _LocalWorkflow(self._run)(dataset, resource or _optimization_resource.get(), kwargs)

    async def _run(self, dataset, resource, kwargs):
        self.dataset = dataset
        vars(self).pop("optimization_info_", None)
        scorer = _serial_scorer(self.pipeline, self.scoring)
        n_splits, optimizer, pure, parameters, pairs, splits = self._plan(dataset)

        folds = [
            (dataset.get_subset(group_labels=train), dataset.get_subset(group_labels=test)) for train, test in splits
        ]
        # A pure parameter's value affects scoring only. Its name matters for the safety check.
        fit_jobs = {}
        candidate_fits = []
        for i, (hyper, pure_params) in enumerate(pairs):
            names = tuple(sorted(_prefix_para_dict(pure_params) or {}))
            group = custom_hash((hyper, names)) if pure else i
            for j, (train, _) in enumerate(folds):
                key = group, j
                if key not in fit_jobs:
                    fit_jobs[key] = Task(
                        _fit_fold, resource=resource, metadata={"cv_fold": j, "parameter_candidate": i}
                    )(optimizer.clone(), train, hyper, names, kwargs)
                candidate_fits.append(key)
        fitted = dict(
            zip(
                fit_jobs,
                await _gather_with_progress(
                    fit_jobs.values(),
                    total=len(fit_jobs),
                    desc="Fit folds",
                    progress_bar=self.progress_bar,
                ),
                strict=True,
            )
        )

        async def evaluate(index, key):
            i, j = divmod(index, len(folds))
            train, test = folds[j]
            pipeline, fit_time = fitted[key]
            with job_metadata(parameter_candidate=i, cv_fold=j):
                scores = await Task(_score_fold, resource=resource)(
                    pipeline, pairs[i][1], scorer, train, test, self.return_train_score
                )
            return {
                **scores,
                "debug__optimize_time": fit_time,
                "train__data_labels": train.group_labels,
                "test__data_labels": test.group_labels,
            }

        results = await _gather_with_progress(
            (evaluate(index, key) for index, key in enumerate(candidate_fits)),
            total=len(candidate_fits),
            desc="Split-Para Combos",
            progress_bar=self.progress_bar,
        )
        if self._select(parameters, n_splits, results):
            start = time.perf_counter()
            optimizer.pipeline = self.pipeline.clone().set_params(**self.best_params_).clone()
            fitted = await Task(_fit, resource=resource)(optimizer, dataset, kwargs)
            self.optimized_pipeline_ = fitted["optimized_pipeline_"]
            if "optimization_info_" in fitted:
                self.optimization_info_ = fitted["optimization_info_"]
            self.final_optimize_time_ = time.perf_counter() - start
        return self
