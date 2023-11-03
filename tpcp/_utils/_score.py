"""Score and fit results.

This is a modified version of sklearn fit and score functionality.

The original code is licenced under BSD-3: https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional, Union

from joblib import Memory
from typing_extensions import TypedDict

from tpcp._base import clone
from tpcp._hash import custom_hash
from tpcp._utils._general import _get_nested_paras
from tpcp.exceptions import OptimizationError, TestError

if TYPE_CHECKING:
    from tpcp._dataset import Dataset
    from tpcp._optimize import BaseOptimize
    from tpcp._pipeline import Pipeline
    from tpcp.validate import Scorer

_SCORE_TYPE = Union[dict[str, float], float]  # pylint: disable=invalid-name
_AGG_SCORE_TYPE = Union[dict[str, float], float]  # pylint: disable=invalid-name
_SINGLE_SCORE_TYPE = Union[dict[str, list[float]], Optional[list[float]]]  # pylint: disable=invalid-name


class _ScoreResults(TypedDict, total=False):
    """Type representing results of _score."""

    scores: _AGG_SCORE_TYPE
    single_scores: _SINGLE_SCORE_TYPE
    score_time: float
    data_labels: list[Union[str, tuple[str, ...]]]
    parameters: Optional[dict[str, Any]]


class _OptimizeScoreResults(TypedDict, total=False):
    """Type representing results of _score_and_optimize."""

    test_scores: _AGG_SCORE_TYPE
    test_single_scores: _SINGLE_SCORE_TYPE
    train_scores: _AGG_SCORE_TYPE
    train_single_scores: _SINGLE_SCORE_TYPE
    score_time: float
    optimize_time: float
    train_data_labels: list[Union[str, tuple[str, ...]]]
    test_data_labels: list[Union[str, tuple[str, ...]]]
    parameters: Optional[dict[str, Any]]
    optimizer: BaseOptimize


def _score(
    pipeline: Pipeline,
    dataset: Dataset,
    scorer: Scorer,
    parameters: Optional[dict[str, Any]],
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    error_info: Optional[str] = None,
) -> _ScoreResults:
    """Set parameters and return score.

    Parameters
    ----------
    pipeline
        An instance of a tpcp pipeline
    dataset
        An instance of a tpcp dataset with multiple data points.
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

    Returns
    -------
    result : dict with the following attributes
        scores : dict of scorer name -> float
            Calculated scores
        scores_single : dict of scorer name -> np.ndarray
            Calculated scores for each individual data point
        score_time : float
            Time required to score the dataset
        data : list
            List of data point labels used
        parameters : dict or None
            The parameters that have been evaluated.

    """
    if parameters is not None:
        # clone after setting parameters in case any parameters are estimators (like pipeline steps).
        parameters = _clone_parameter_dict(parameters)

        pipeline = pipeline.set_params(**parameters)

    try:
        start_time = time.time()
        agg_scores, single_scores = scorer(pipeline, dataset)
        score_time = time.time() - start_time
    except Exception as e:  # noqa: BLE001
        raise TestError(
            f"Testing the algorithm on the dataset failed with the error above.\n{error_info or ''}\n\n"
            f"The test-set is:\n{[d.group_labels for d in dataset]}"
        ) from e

    result: _ScoreResults = {"scores": agg_scores, "single_scores": single_scores}
    if return_times:
        result["score_time"] = score_time
    if return_data_labels:
        result["data_labels"] = dataset.group_labels
    if return_parameters:
        result["parameters"] = parameters
    return result


def _optimize_and_score(
    optimizer: BaseOptimize,
    scorer: Scorer,
    train_set: Dataset,
    test_set: Dataset,
    *,
    optimize_params: Optional[dict] = None,
    hyperparameters: Optional[dict] = None,
    pure_parameters: Optional[dict] = None,
    return_train_score=False,
    return_optimizer=False,
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    memory: Optional[Memory] = None,
    error_info: Optional[str] = None,
) -> _OptimizeScoreResults:
    """Optimize and score the optimized pipeline on the train and test data, respectively.

    This method is aware of the differences between hyperparameters and normal (pure) parameters.
    This can be used to cache the results of training, as the training results should only depend on hyperparameters,
    but not on normal parameters.
    The provided `memory` instance is used to perform this caching step.

    Note, that caching in this context should only be performed in the context of a single call to e.g. GridSearchCV
    and the cache should be deleted afterwards to avoid cache leak.
    Therefore, the cachedir should ideally be set to a random tmp dir by the caller.

    Note: In the past, we only provided the index to the optimize method and not the subsets of the data directly.
    We changed that, because in cases, where the data index is non-deterministic, the order of the dataset might
    change between the point where we calculate the split (in the caller) and the point where we extract the respective
    data points (here).
    This might even happen in a different Python process (in the case of multiprocessing) where the dataset object was
    recreated.
    Hence, it might be that the dataobject recreated after pickling and unpickling is not the same as the original one.
    To avoid this entirely, we split and extract the subset right in the same process in the caller and only pass
    the subset to this method.
    Really badly written Datasets might still get past this, but let's hope this rarely happens.
    Note, that the Dataset object itself has a safeguard against non-deterministic indices, but this is not a guarantee.
    Let's better be safe and reduce the surface for bugs even further here.

    """
    if memory is None:
        memory = Memory(None)
    # clone after setting parameters in case any parameters are estimators (like pipeline steps).
    hyperparameters = _clone_parameter_dict(hyperparameters)
    pure_parameters = _clone_parameter_dict(pure_parameters)

    optimize_params_clean: dict = optimize_params or {}

    try:
        start_time = time.time()
        optimizer = _cached_optimize(
            optimizer, train_set, hyperparameters, pure_parameters, memory, optimize_params_clean
        )
        optimize_time = time.time() - start_time
    except Exception as e:  # noqa: BLE001
        raise OptimizationError(
            f"The optimization on the trainset failed with the error above.\n{error_info or ''}\n\n"
            f"This optimization used the following trainset:\n{train_set}"
        ) from e

    # Now we set the remaining paras.
    # Because, we need to set the parameters on the optimized pipeline and not the input pipeline we strip the
    # naming prefix.
    striped_paras = _get_nested_paras(pure_parameters, "pipeline")
    optimizer.optimized_pipeline_.set_params(**striped_paras)
    # We also set the parameters of the input pipeline to make it seem that all parameters were set from the
    # beginning.
    optimizer = optimizer.set_params(**pure_parameters)

    try:
        agg_scores, single_scores = scorer(optimizer.optimized_pipeline_, test_set)
        score_time = time.time() - optimize_time - start_time
    except Exception as e:  # noqa: BLE001
        raise TestError(
            f"Testing the optimized algorithm on the test-set failed with the error above.\n{error_info or ''}\n\n"
            f"The test-set is:\n{test_set}"
        ) from e

    result: _OptimizeScoreResults = {"test_scores": agg_scores, "test_single_scores": single_scores}
    if return_train_score:
        try:
            train_agg_scores, train_single_scores = scorer(optimizer.optimized_pipeline_, train_set)
        except Exception as e:  # noqa: BLE001
            raise TestError(
                "Running the optimized algorithm on the train-set to calculate the train error failed with the error "
                f"above.\n{error_info or ''}\n\n"
                f"The train-set is:\n{[d.group_labels for d in test_set]}"
            ) from e
        result["train_scores"] = train_agg_scores
        result["train_single_scores"] = train_single_scores
    if return_times:
        result["score_time"] = score_time
        result["optimize_time"] = optimize_time
    if return_data_labels:
        # Note we always return the train data attribute as it is interesting information independent of the train
        # score and has 0 runtime impact.
        result["train_data_labels"] = train_set.group_labels
        result["test_data_labels"] = test_set.group_labels
    if return_optimizer:
        # This is the actual trained optimizer. This means that `optimizer.optimized_pipeline_` contains the actual
        # instance of the trained pipeline.
        result["optimizer"] = optimizer
    if return_parameters:
        result["parameters"] = {**hyperparameters, **pure_parameters}
    return result


def _cached_optimize(
    optimizer: BaseOptimize, data: Dataset, hyperparameters: dict, pure_parameters: dict, memory: Memory, optimize_paras
):
    """Set parameters and optimize a pipeline and cache the optimization result.

    This method will cache the training as long as the hyperparameters stay the same.
    Changing the pure parameters will not invalidate the cache.

    """

    # We do not set all paras right away, we first create a cached optimize function that only has the hyper
    # parameters as input.
    # This allows to cache the train results, if the _optimize_and_score is called multiple times with the same hyper
    # parameters.
    # To be sure that nothing "bad" happens here, we also pass in the pipeline class itself to invalidate the cache,
    # in case a completely different pipeline/algorithm is optimized.
    # Ideally the `memory` object used here should only be used once.
    # E.g. for a single a GridSearchCV.
    def cachable_optimize(
        opti: type[BaseOptimize], hyperparas: dict[str, Any], data: Dataset, optimize_params: dict
    ) -> BaseOptimize:
        _ = opti
        return optimizer.set_params(**hyperparas).optimize(data, **optimize_params)

    optimize_func = memory.cache(cachable_optimize)
    # Optimization must never modify pure parameters, or we have a problem.
    # We check that by calculating the hash of all pure parameters before and after the optimization.
    opti_paras = optimizer.get_params()
    pure_para_subset = {k: opti_paras[k] for k in pure_parameters}
    pure_para_hash = custom_hash(pure_para_subset)
    pipeline_pure_para_hash = custom_hash(_get_nested_paras(pure_para_subset, "pipeline"))

    # This is the actual call to train the optimizer:
    optimizer = optimize_func(type(optimizer), hyperparameters, data, optimize_paras)

    opti_paras = optimizer.get_params()
    optimized_pipeline_paras = optimizer.optimized_pipeline_.get_params()
    # We check that the pure parameters on the optimize object haven't changed and that the pure parameters belonging
    # to the pipeline have not changed in the `optimized_pipeline`.
    # Note, that the first case will never happen with tpcp native Optimizers, but could happen for custom
    # optimizers.
    if pipeline_pure_para_hash != custom_hash(
        {k: optimized_pipeline_paras[k] for k in _get_nested_paras(pure_parameters, "pipeline")}
    ) or pure_para_hash != custom_hash({k: opti_paras[k] for k in pure_parameters}):
        raise ValueError(
            "Optimizing the pipeline modified a parameter marked as `pure`. "
            "This must not happen. "
            "Double check your optimize implementation and the list of pure parameters."
        )

    return optimizer


def _clone_parameter_dict(param_dict: Optional[dict]) -> dict:
    cloned_param_dict = {}
    if param_dict is not None:
        for k, v in param_dict.items():
            cloned_param_dict[k] = clone(v, safe=False)
    return cloned_param_dict
