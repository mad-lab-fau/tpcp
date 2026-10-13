"""Helper to validate/evaluate pipelines and Optimize."""

import time
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

from sklearn.model_selection import BaseCrossValidator
from tqdm.auto import tqdm

from tpcp._async_execution import _gather_with_progress
from tpcp._base import _Default
from tpcp._dataset import DatasetT
from tpcp._optimize import BaseOptimize
from tpcp._pipeline import PipelineT
from tpcp._utils._general import _aggregate_final_results, _normalize_score_results, _passthrough, _prefix_para_dict
from tpcp._utils._score import _optimize_and_score, _score
from tpcp.misc import iter_with_warning_error_context, warning_error_context
from tpcp.parallel import Parallel, Task, delayed, job_metadata
from tpcp.validate._cross_val_helper import BaseDatasetSplitter, _normalize_splitter
from tpcp.validate._scorer import ScoreFunc, Scorer, ScorerTypes, _validate_scorer

if TYPE_CHECKING:
    from tpcp import Dataset
    from tpcp.optimize import GridSearch


def cross_validate(
    optimizable: BaseOptimize[PipelineT, DatasetT],
    dataset: DatasetT,
    *,
    scoring: ScoreFunc[PipelineT, DatasetT],
    cv: BaseDatasetSplitter | int | BaseCrossValidator | list | None = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    optimize_params: dict[str, Any] | None = None,
    pre_dispatch: str | int = "2*n_jobs",
    return_train_score: bool = False,
    return_optimizer: bool = False,
    progress_bar: bool = True,
):
    """Evaluate a pipeline on a dataset using cross validation.

    This function follows as much as possible the interface of :func:`~sklearn.model_selection.cross_validate`.
    If the tpcp documentation is missing some information, the respective documentation of sklearn might be helpful.

    Parameters
    ----------
    optimizable
        A optimizable class instance like :class:`~tpcp.optimize.GridSearch`/:class:`~tpcp.optimize.GridSearchCV` or a
        :class:`~tpcp.Pipeline` wrapped in an `Optimize` object (:class:`~tpcp.OptimizablePipeline`).
    dataset
        A :class:`~tpcp.Dataset` containing all information.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
    cv
        The cross-validation strategy to use.
        Accepts an integer, a sklearn-style splitter, a native tpcp splitter, or an explicit list of
        positional fold assignments. Convert other positional iterables with ``list(folds)`` before passing them.

        For grouping or stratification, use :class:`~tpcp.validate.DatasetSplitter`. Native tpcp splitters such as
        :class:`~tpcp.validate.CombinedSplitter` can also be passed directly.
    n_jobs
        Number of jobs to run in parallel.
        One job is created per CV fold.
        The default (`None`) means 1 job at the time, hence, no parallel computing.
    verbose
        The verbosity level (larger number -> higher verbosity).
        At the moment this only effects `Parallel`.
    optimize_params
        Additional parameter that are forwarded to the `optimize` method.
    pre_dispatch
        The number of jobs that should be pre dispatched.
        For an explanation see the documentation of :class:`~joblib.Parallel`.
    return_train_score
        If True the performance on the train score is returned in addition to the test score performance.
        Note, that this increases the runtime.
        If `True`, the fields `train_data_labels`, `train_score`, and `train_score_single` are available in the results.
    return_optimizer
        If the optimized instance of the input optimizable should be returned.
        If `True`, the field `optimizer` is available in the results.
    progress_bar
        True/False to enable/disable a `tqdm` progress bar.

    Returns
    -------
    result_dict
        Dictionary with results.
        Each element is either a list or array of length `n_folds`.
        The dictionary can be directly passed into the pandas DataFrame constructor for a better representation.

        The following fields are in the results:

        test__score / test__{scorer-name}
            The aggregated value of a score over all data-points.
            If a single score is used for scoring, then the generic name "score" is used.
            Otherwise, multiple columns with the name of the respective scorer exist.
        test__single__score / test__single__{scorer-name}
            The individual scores per datapoint per fold.
            This is a list of values with the `len(train_set)`.
        test__data_labels
            A list of data labels of the train set in the order the single score values are provided.
            These can be used to associate the `single_score` values with a certain data-point.
        train__score / train__{scorer-name}
            Results for train set of each fold.
        train__single__score / train__single__{scorer-name}
            Results for individual data points in the train set of each fold
        train__data_labels
           The data labels for the train set.
        optimize_time
            Time required to optimize the pipeline in each fold.
        score_time
            Cumulative score time to score all data points in the test set.
        optimizer
            The optimized instances per fold.
            One instance per fold is returned.
            The optimized version of the pipeline can be obtained via the `optimized_pipeline_` attribute on the
            instance.

    """
    scoring = _validate_scorer(scoring)

    cv = _normalize_splitter(cv)

    splits = list(cv.split(dataset))

    pbar = partial(tqdm, total=len(splits), desc="CV Folds") if progress_bar else _passthrough

    def tasks():
        for make_context, (train, test) in iter_with_warning_error_context(splits):
            with make_context("cv_fold"):
                # delayed snapshots the active context; yield only after it has closed.
                task = delayed(_optimize_and_score)(
                    # We clone the estimator to make sure that all the folds are
                    # independent, and that it is pickle-able.
                    optimizable.clone(),
                    scoring,
                    dataset.get_subset(group_labels=train),
                    dataset.get_subset(group_labels=test),
                    optimize_params=optimize_params,
                    hyperparameters=None,
                    pure_parameters=None,
                    return_train_score=return_train_score,
                    return_times=True,
                    return_data_labels=True,
                    return_optimizer=return_optimizer,
                )
            yield task

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch, return_as="generator")
    with parallel:
        results = list(pbar(parallel(tasks())))
    assert results is not None  # For the typechecker
    return _format_cv_results(results)


def async_cross_validate(
    optimizable: "GridSearch[PipelineT, DatasetT]",
    dataset: DatasetT,
    *,
    scoring: ScorerTypes[PipelineT, DatasetT],
    cv: BaseDatasetSplitter | int | BaseCrossValidator | list | None = None,
    optimize_params: dict[str, Any] | None = None,
    return_train_score: bool = False,
    return_optimizer: bool = False,
    progress_bar: bool = True,
    resource: Literal["cpu", "gpu"] = "cpu",
) -> Task:
    """Bind a lazy cross-validation workflow; await it inside an execution context.

    Fold workflows and GridSearch candidates share one process pool. Search
    selection and results stay on the main thread; scoring runs in workers.
    ``resource="gpu"`` requests both CPU and GPU slots for leaf jobs.
    ``progress_bar`` shows fold completion; candidate bars follow the search's
    settings. Scorers run serially within each job. Timings include queue waits.
    See :func:`cross_validate` for result fields. Failures from composed jobs
    propagate as an ``ExceptionGroup``.
    """
    return Task(
        _cross_validate_async,
        args=(optimizable, dataset),
        kwargs={
            "scoring": scoring,
            "cv": cv,
            "optimize_params": optimize_params,
            "return_train_score": return_train_score,
            "return_optimizer": return_optimizer,
            "progress_bar": progress_bar,
            "resource": resource,
        },
        kind="workflow",
    )


async def cross_validate_async(
    optimizable: "GridSearch[PipelineT, DatasetT]",
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
    """Execute :func:`async_cross_validate` through a coroutine entry point."""
    return await async_cross_validate(
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
    optimizable: "GridSearch[PipelineT, DatasetT]",
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
    as :func:`cross_validate`; timings measure elapsed time including pool waits.
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
                if resource != "cpu":
                    parameters.setdefault("resource", resource)
                await optimizer.optimize_async(train, **parameters)
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
                        await Task(
                            _score, args=(optimizer.optimized_pipeline_, subset, scorer, None), resource=resource
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


def _format_cv_results(results):
    results = _aggregate_final_results(results)

    # Fix the formatting of all the score results
    for group in ["test", "train"]:
        scores = _prefix_para_dict(_normalize_score_results(results.pop(f"{group}__scores", [])), f"{group}__agg__")
        single_scores = _prefix_para_dict(
            _normalize_score_results(results.pop(f"{group}__single__scores", [])), f"{group}__single__"
        )
        results = {**results, **single_scores, **scores}

    return results


def validate(
    pipeline: PipelineT,
    dataset: DatasetT,
    *,
    scoring: ScorerTypes[PipelineT, DatasetT],
    n_jobs: int | None = _Default(None),
    verbose: int = _Default(0),
    pre_dispatch: str | int = _Default("2*n_jobs"),
    progress_bar: bool = _Default(True),
) -> dict[str, Any]:
    """Evaluate a pipeline on a dataset without any optimization.

    Parameters
    ----------
    pipeline
        A :class:`~tpcp.Pipeline` to evaluate on the given data.
    dataset
        A :class:`~tpcp.Dataset` containing all information.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
    n_jobs
        Number of jobs to run in parallel.
        One job is created per datapoint.
        The default (`None`) means 1 job at the time, hence, no parallel computing.
    verbose
        The verbosity level (larger number -> higher verbosity).
        At the moment this only effects `Parallel`.
    pre_dispatch
        The number of jobs that should be pre dispatched.
        For an explanation see the documentation of :class:`~joblib.Parallel`.
    progress_bar
        True/False to enable/disable a `tqdm` progress bar.
    """
    scoring_args = {"n_jobs": n_jobs, "verbose": verbose, "pre_dispatch": pre_dispatch, "progress_bar": progress_bar}
    if isinstance(scoring, Scorer):
        for arg, value in scoring_args.items():
            if isinstance(value, _Default):
                continue
            raise ValueError(
                "You passed a explicit Scorer object for the scoring parameter. In this case, we expect "
                f"multiprocessing parameters ({list(scoring_args.keys())}) to be configured directly on the "
                f"Scorer instance. However, you specified {arg}={value} by passing it directly "
                "to the validate function. Instead, pass multiprocessing parameters to your Scorer during "
                "initialization or by using `set_params`."
            )
        scoring = _validate_scorer(scoring)
    else:
        scoring = _validate_scorer(scoring)
        scoring.set_params(
            **{arg: value.get_value() if isinstance(value, _Default) else value for arg, value in scoring_args.items()}
        )

    results = _score(
        pipeline.clone(),
        dataset,
        scoring,
        pipeline.get_params(),
        return_data_labels=True,
        return_times=True,
    )
    results = _aggregate_final_results([results])

    # Fix the formatting of all the score results
    scores = _prefix_para_dict(_normalize_score_results(results.pop("scores", [])), "agg__")
    single_scores = _prefix_para_dict(_normalize_score_results(results.pop("single__scores", [])), "single__")
    results = {**results, **single_scores, **scores}

    return results
