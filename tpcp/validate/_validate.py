"""Helper to validate/evaluate pipelines and Optimize."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from joblib import delayed
from sklearn.model_selection import BaseCrossValidator, check_cv
from tqdm.auto import tqdm

from tpcp import Dataset
from tpcp._optimize import BaseOptimize
from tpcp._utils._general import _aggregate_final_results, _normalize_score_results
from tpcp._utils._multiprocess import TqdmParallel
from tpcp._utils._score import _ERROR_SCORE_TYPE, _optimize_and_score
from tpcp.validate._scorer import _validate_scorer


def cross_validate(
    optimizable: BaseOptimize,
    dataset: Dataset,
    *,
    groups: Optional[List[Union[str, Tuple[str, ...]]]] = None,
    scoring: Optional[Callable] = None,
    cv: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    optimize_params: Optional[Dict[str, Any]] = None,
    propagate_groups: bool = True,
    pre_dispatch: Union[str, int] = "2*n_jobs",
    return_train_score: bool = False,
    return_optimizer: bool = False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
    progress_bar: bool = True,
):
    """Evaluate a pipeline on a dataset using cross validation.

    This function follows as much as possible the interface of :func:`~sklearn.model_selection.cross_validate`.
    If the tpcp documentation is missing some information, the respective documentation of sklearn might be helpful.

    Parameters
    ----------
    optimizable
        A optimizable class instance like :class:`~tpcp.GridSearch`/:class:`~tpcp.GridSearchCV` or a
        :class:`~tpcp.Pipeline` wrapped in an `Optimize` object (:class:`~tpcp.OptimizablePipeline`).
    dataset
        A :class:`~tpcp.Dataset` containing all information.
    groups
        Group labels for samples used by the cross validation helper, in case a grouped CV is used (e.g.
        :class:`~sklearn.model_selection.GroupKFold`).
        Check the documentation of the :class:`~tpcp.dataset.Dataset` class and the respective example for
        information on how to generate group labels for tpcp datasets.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
        If scoring is `None` the default `score` method of the optimizable is used instead.
    cv
        An integer specifying the number of folds in a K-Fold cross validation or a valid cross validation helper.
        The default (`None`) will result in a 5-fold cross validation.
        For further inputs check the `sklearn`
        `documentation
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>`_.
    n_jobs
        Number of jobs to run in parallel.
        One job is created per CV fold.
        The default (`None`) means 1 job at the time, hence, no parallel computing.
    verbose
        The verbosity level (larger number -> higher verbosity).
        At the moment this only effects `Parallel`.
    optimize_params
        Additional parameter that are forwarded to the `optimize` method.
    propagate_groups
        In case your optimizable is a cross validation based optimize (e.g. :class:`~tpcp.GridSearchCv`) and you are
        using a grouped cross validation, you probably want to use the same grouped CV for the outer and the inner
        cross validation. If `propagate_groups` is True, the group labels belonging to the training of each fold are
        passed to the `optimize` method of the optimizable. This only has an effect if `groups` are specified.
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
    error_score
        Value to assign to the score if an error occurs during scoring.
        If set to ‘raise’, the error is raised.
        If a numeric value is given, a Warning is raised.
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
    cv_checked: BaseCrossValidator = check_cv(cv, None, classifier=True)

    scoring = _validate_scorer(scoring, optimizable.pipeline)

    optimize_params = optimize_params or {}
    if propagate_groups is True and "groups" in optimize_params:
        raise ValueError(
            "You can not use `propagate_groups` and specify `groups` in `optimize_params`. "
            "The latter would overwrite the prior. "
            "Most likely you only want to use `propagate_groups`."
        )

    splits = list(cv_checked.split(dataset, groups=groups))
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    pbar: Optional[tqdm] = None
    if progress_bar:
        pbar = tqdm(desc="CV Folds", total=len(splits))
    parallel = TqdmParallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch, pbar=pbar)
    results = parallel(
        delayed(_optimize_and_score)(
            optimizable.clone(),
            dataset,
            scoring,
            train,
            test,
            optimize_params={**_propagate_groups(propagate_groups, groups, train), **optimize_params},
            hyperparameters=None,
            pure_parameters=None,
            return_train_score=return_train_score,
            return_times=True,
            return_data_labels=True,
            return_optimizer=return_optimizer,
            error_score=error_score,
        )
        for train, test in splits
    )
    assert results is not None  # For the typechecker
    results = _aggregate_final_results(results)
    score_results = {}
    # Fix the formatting of all the score results
    for name in ["test_scores", "test_single_scores", "train_scores", "train_single_scores"]:
        if name in results:
            score = results.pop(name)
            prefix = ""
            if "_" in name:
                prefix = name.rsplit("_", 1)[0] + "_"
            score = _normalize_score_results(score, prefix)
            # We use a new dict here, as it is unsafe to append a dict you are iterating over
            score_results.update(score)

    results.update(score_results)
    return results


def _propagate_groups(
    propagate_groups: bool, groups: Optional[List[Union[str, Tuple[str, ...]]]], train_idx: List[int]
):
    if propagate_groups is False or groups is None:
        return {}
    return {"groups": np.array(groups)[train_idx]}
