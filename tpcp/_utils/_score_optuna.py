import numbers
import time
from typing import Optional, Dict, Any, Tuple, Callable

import numpy as np
from optuna import Trial
from optuna.trial import TrialState

from tpcp import Pipeline, Dataset
from tpcp._utils._score import ScoreResults, _clone_parameter_dict
from tpcp._utils._score import _ERROR_SCORE_TYPE
from tpcp.validate import Scorer


def _score(
    pipeline: Pipeline,
    dataset: Dataset,
    scorer: Scorer,
    parameters: Optional[Dict[str, Any]],
    trial: Trial,
    transform_score: Callable,
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
) -> ScoreResults:
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
        parameters = _clone_parameter_dict(parameters)

        pipeline = pipeline.set_params(**parameters)

    start_time = time.time()
    agg_scores, single_scores, state = scorer.call_optuna(pipeline, dataset, error_score, trial, transform_score)
    score_time = time.time() - start_time

    result: ScoreResults = {"scores": agg_scores, "single_scores": single_scores, "state": state}
    if return_times:
        result["score_time"] = score_time
    if return_data_labels:
        result["data_labels"] = dataset.groups
    if return_parameters:
        result["parameters"] = parameters
    return result