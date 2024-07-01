"""Helper to validate/evaluate pipelines and Optimize."""
from collections.abc import Iterator
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
from joblib import Parallel
from sklearn.model_selection import BaseCrossValidator
from tqdm.auto import tqdm

from tpcp import Dataset, Pipeline
from tpcp._base import _Default
from tpcp._optimize import BaseOptimize
from tpcp._utils._general import _aggregate_final_results, _normalize_score_results, _passthrough
from tpcp._utils._score import _optimize_and_score, _score
from tpcp.parallel import delayed
from tpcp.validate._cross_val_helper import TpcpSplitter
from tpcp.validate._scorer import Scorer, _validate_scorer


def cross_validate(
    optimizable: BaseOptimize,
    dataset: Dataset,
    *,
    scoring: Optional[Callable] = None,
    cv: Optional[Union[TpcpSplitter, int, BaseCrossValidator, Iterator]] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    optimize_params: Optional[dict[str, Any]] = None,
    pre_dispatch: Union[str, int] = "2*n_jobs",
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
        If scoring is `None` the default `score` method of the optimizable is used instead.
    cv
        The cross-validation strategy to use.
        For simple use-cases the same input as for the sklearn cross-validation function are supported.
        For further inputs check the `sklearn` `documentation
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>`_.

        For more complex usecases like grouping or stratification, the :class:`~tpcp.TpcpSplitter` can be used.
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

        test_score / test_{scorer-name}
            The aggregated value of a score over all data-points.
            If a single score is used for scoring, then the generic name "score" is used.
            Otherwise, multiple columns with the name of the respective scorer exist.
        test_single_score / test_single_{scorer-name}
            The individual scores per datapoint per fold.
            This is a list of values with the `len(train_set)`.
        test_data_labels
            A list of data labels of the train set in the order the single score values are provided.
            These can be used to associate the `single_score` values with a certain data-point.
        train_score / train_{scorer-name}
            Results for train set of each fold.
        train_single_score / train_single_{scorer-name}
            Results for individual data points in the train set of each fold
        train_data_labels
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
    scoring = _validate_scorer(scoring, optimizable.pipeline)

    cv = cv if isinstance(cv, TpcpSplitter) else TpcpSplitter(base_splitter=cv)

    splits = list(cv.split(dataset))

    pbar = partial(tqdm, total=len(splits), desc="CV Folds") if progress_bar else _passthrough

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch, return_as="generator")
    with parallel:
        results = list(
            pbar(
                parallel(
                    delayed(_optimize_and_score)(
                        # We clone the estimator to make sure that all the folds are
                        # independent, and that it is pickle-able.
                        optimizable.clone(),
                        scoring,
                        dataset[train],
                        dataset[test],
                        optimize_params=optimize_params,
                        hyperparameters=None,
                        pure_parameters=None,
                        return_train_score=return_train_score,
                        return_times=True,
                        return_data_labels=True,
                        return_optimizer=return_optimizer,
                        error_info=f"This error occurred in fold {i}.",
                    )
                    for i, (train, test) in enumerate(splits)
                )
            )
        )
    assert results is not None  # For the typechecker
    results = _aggregate_final_results(results)

    # Fix the formatting of all the score results
    score_results = _reformat_scores(
        ["test_scores", "test_single_scores", "train_scores", "train_single_scores"], results
    )
    results.update(score_results)

    return results


def validate(
    pipeline: Pipeline,
    dataset: Dataset,
    *,
    scoring: Optional[Union[Callable, Scorer]] = None,
    n_jobs: Optional[int] = _Default(None),
    verbose: int = _Default(0),
    pre_dispatch: Union[str, int] = _Default("2*n_jobs"),
    progress_bar: bool = _Default(True),
):
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
        If scoring is `None` the default `score` method of the optimizable is used instead.
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
    # iterate over args that will be passed to Scorer
    for arg, value in scoring_args.items():
        # when a Scorer instance is provided, the respective arguments were already set
        if isinstance(scoring, Scorer) and not isinstance(value, _Default):
            raise ValueError(  # noqa: TRY004
                "You passed a explicit Scorer object for the scoring parameter. In this case, we expect "
                f"multiprocessing parameters ({list(scoring_args.keys())}) to be configured directly on the "
                f"Scorer instance. However, you specified {arg}={value} by passing it directly "
                "to the validate function. Instead, pass multiprocessing parameters to your Scorer during "
                "initialization or by using `set_params`."
            )

        # extract the value from _Default instances
        if isinstance(value, _Default):
            scoring_args[arg] = value.get_value()

    scoring = _validate_scorer(scoring, pipeline)

    scoring.set_params(**scoring_args)

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
    score_results = _reformat_scores(["scores", "single_scores"], results)
    results.update(score_results)

    return results


def _propagate_values(
    name: str, propagate_values: bool, values: Optional[list[Union[str, tuple[str, ...]]]], train_idx: list[int]
):
    if propagate_values is False or values is None:
        return {}
    return {name: np.array(values)[train_idx]}


def _reformat_scores(score_names: list[str], score_results: dict[str, Any]):
    # extract subtypes (e.g., precision, recall) of scores from nested score dictionaries
    reformatted_results = {}
    for name in score_names:
        if name in score_results:
            score = score_results.pop(name)
            prefix = ""
            if "_" in name:
                prefix = name.rsplit("_", 1)[0] + "_"
            score = _normalize_score_results(score, prefix)
            # We use a new dict here, as it is unsafe to append a dict you are iterating over
            reformatted_results.update(score)
    return reformatted_results
