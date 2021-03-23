"""This is a modified version of sklearn fit and score functionality.

The original code is licenced under BSD-3: https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

import numbers
import time
from typing import Dict, Optional

import numpy as np
from joblib import Memory
from sklearn import clone

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import BaseOptimize
from gaitmap.future.pipelines._pipelines import SimplePipeline
from gaitmap.future.pipelines._scorer import GaitmapScorer, _ERROR_SCORE_TYPE


def _score(
    pipeline: SimplePipeline,
    dataset: Dataset,
    scorer: GaitmapScorer,
    parameters: Optional[Dict],
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
):
    """Set parameters and return score.

    Parameters
    ----------
    pipeline
        An instance of a gaitmap pipeline
    dataset
        An instance of a gaitmap dataset with multiple data points.
    scorer
        A scorer that calculates a score by running the pipeline on each data point and then aggregates the results.
    parameters : dict of valid parameters for the pipeline
        The parameters that should be set for the pipeline before scoring
    return_parameters
        If the parameter value that was inputted should be added to the result dict
    return_data_labels
        If the names of the data points should be added to the result dict
    return_times
        If the time required to score the dataset should be added to the result dict
    error_score
        The value that should be used if scoring fails for a specific data point.
        This can be any numeric value (including nan and inf) or "raises".
        If it is "raises", the scoring error is raised instead of ignored.
        In all other cases a warning is displayed.
        Note, that if the value is set to np.nan, the aggregated value over multiple data points will also be nan,
        if scoring fails for a single data point.

    Returns
    -------
    result : dict with the following attributes
        scores : dict of scorer name -> float
            Calculated scores
        scores_single : dict of scorer name -> np.ndarray
            Calculated scores for each individual data point
        score_time : float
            Time required to score the dataset
        data : List
            List of data point labels used
        parameters : dict or None
            The parameters that have been evaluated.

    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been spelled correctly.)"
        )

    if parameters is not None:
        # clone after setting parameters in case any parameters are estimators (like pipeline steps).
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        pipeline = pipeline.set_params(**cloned_parameters)

    start_time = time.time()
    agg_scores, single_scores = scorer(pipeline, dataset, error_score)
    score_time = time.time() - start_time

    result = {"scores": agg_scores, "single_scores": single_scores}
    if return_times:
        result["score_time"] = score_time
    if return_data_labels:
        result["data_labels"] = dataset.groups
    if return_parameters:
        result["parameters"] = parameters
    return result


def _optimize_and_score(
    optimizer: BaseOptimize,
    dataset: Dataset,
    scorer: GaitmapScorer,
    train: np.ndarray,
    test: np.ndarray,
    hyperparameters: Optional[Dict],
    pure_parameters: Optional[Dict],
    return_train_score=False,
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
    memory: Optional[Memory] = None,
):
    """Optimize and score the optimized pipeline on the train and test data, respectively.

    This method is aware of the differences between hyperparameters and normal (pure) parameters.
    This can be used to cache the results of training, as the training results should only depend on hyperparameters,
    but not on normal parameters.
    The provided `memory` instance is used to perform this caching step.

    Note, that caching in this context should only be performed in the context of a single call to e.g. GridSearchCV
    and the cache should be deleted afterwards to avoid cache leak.
    Therefore, the cachedir should ideally be set to a random tmp dir by the caller.
    """
    if memory is None:
        memory = Memory(None)
    if hyperparameters is not None:
        # clone after setting parameters in case any parameters are estimators (like pipeline steps).
        cloned_hyperparameters = {}
        for k, v in hyperparameters.items():
            cloned_hyperparameters[k] = clone(v, safe=False)
        hyperparameters = cloned_hyperparameters
    if pure_parameters is not None:
        cloned_pure_parameters = {}
        for k, v in hyperparameters.items():
            cloned_pure_parameters[k] = clone(v, safe=False)
        pure_parameters = cloned_pure_parameters

    train_set = dataset[train]
    test_set = dataset[test]

    # We do not set all paras right away, we first create a cached optimize function that only has the hyper
    # parameters as input.
    # This allows to cache the train results, if the _optimize_and_score is called multiple times with the same hyper
    # parameters.
    # To be sure that nothing "bad" happens here, we also pass in the pipeline itself to invalidate the cache,
    # in case a completly different pipeline/algorithm is optimized.
    # Ideally the `memory` object used here should only be used once in the context of e.g. a GridSearchCV.
    # TODO: Throw error if optimization modifies pure parameter
    def cachable_optimize(opti: BaseOptimize, hyperparas: Dict, data: Dataset) -> BaseOptimize:
        return opti.set_params(**hyperparas).optimize(data)

    start_time = time.time()
    optimize_func = memory.cache(cachable_optimize)
    optimizer = optimize_func(optimizer, hyperparameters, train_set)
    optimize_time = time.time() - start_time

    # Now we set the remaining paras
    optimized_pipeline = optimizer.optimized_pipeline_.set_params(**pure_parameters)

    agg_scores, single_scores = scorer(optimized_pipeline, test_set, error_score)
    score_time = time.time() - optimize_time

    result = {"scores": agg_scores, "single_scores": single_scores}
    if return_train_score:
        train_agg_scores, train_single_scores = scorer(optimized_pipeline, train_set, error_score)
        train_score_time = time.time() - score_time
        result["train_scores"] = train_agg_scores
        result["train_single_scores"] = train_single_scores
    if return_times:
        result["score_time"] = score_time
        result["optimize_time"] = optimize_time
        if return_train_score:
            result["train_score_time"] = train_score_time
    if return_data_labels:
        result["train_data_labels"] = train_set.groups
        result["test_data_labels"] = test_set.groups
    if return_parameters:
        result["parameters"] = {**hyperparameters, **pure_parameters}
    return result
