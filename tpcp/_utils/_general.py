"""Some helper to work with the format the results of GridSearches and CVs."""
from __future__ import annotations

import copy
import numbers
from typing import Optional

import numpy as np


def _passthrough(arg):
    """Pass through function. Return the input."""
    return arg


def _noop(*_, **__):
    """No operation."""


def _aggregate_final_results(results: list) -> dict:
    """Aggregate the list of dict to dict of np ndarray/list.

    Modified based on sklearn.model_selection._validation._aggregate_score_dicts


    Parameters
    ----------
    results : list of dict
        List of dicts of the results for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------
    >>> results = [{"a": 1, "b": 10}, {"a": 2, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 10}]
    >>>
    >>> _aggregate_final_results(results)  # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}

    """
    return {
        key: np.asarray([score[key] for score in results])
        if isinstance(results[0][key], numbers.Number)
        else [score[key] for score in results]
        for key in results[0]
    }


def _normalize_score_results(scores: list, prefix="", single_score_key="score"):
    """Create a scoring dictionary based on the type of `scores`."""
    if scores[0] is None:
        # This is the case, when we have a custom aggregator that sets RETURN_RAW_SCORES to False.
        return None
    if isinstance(scores[0], dict):
        # multimetric scoring
        return {prefix + k: v for k, v in _aggregate_final_results(scores).items()}
    # single
    return {prefix + single_score_key: scores}


def _prefix_para_dict(params_dict: Optional[dict], prefix="pipeline__") -> Optional[dict]:
    """Add a prefix to all parameter names in the dictionary.

    This can be helpful to adjust a parameter grid that was originally created for a pipeline to work on a wrapper like
    `Optimize` using the `__` naming convention for nested objects.
    """
    if not params_dict:
        return None
    return {prefix + k: v for k, v in params_dict.items()}


def _get_nested_paras(param_dict: Optional[dict], nested_object_name="pipeline") -> dict:
    """Get the parameters belonging to a nested object and remove the suffix.

    If the parameter of a double nested object are required, use `level_1__level_1`.
    """
    if not param_dict:
        return {}
    return {k.split("__", 1)[1]: v for k, v in param_dict.items() if k.startswith(f"{nested_object_name}__")}


def _split_hyper_and_pure_parameters(
    param_dict: list[dict], pure_parameters: Optional[list[str]]
) -> list[tuple[Optional[dict], Optional[dict]]]:
    """Split a list of parameters in hyperparameters and pure parameters.

    For each dictionary in the list, this separates the pure parameters (names provided in input) from all
    hyperparameters (remaining parameters).
    If either the none of the pure parameters is present in a parameter dict or all parameters are pure parameters,
    the pure or the hyperparameters are `None`.

    Returns
    -------
    split_parameters
        List of tuples `(hyper, pure)` for each of the para dicts in the input list.

    """
    if pure_parameters is None:
        return [(c, None) for c in param_dict]
    split_param_dict = []
    for c in param_dict:
        # We need to copy, otherwise, we remove elements from the actual parameter list that is passed as input.
        c = copy.copy(c)  # noqa: PLW2901
        tmp = {}
        for k in list(c.keys()):
            if k in pure_parameters:
                tmp[k] = c.pop(k)
        split_param_dict.append((c or None, tmp or None))
    return split_param_dict
