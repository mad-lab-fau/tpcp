"""Some helper to work with the format the results of GridSearches and CVs."""
from __future__ import annotations

import copy
import numbers
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple, Union

import joblib
import numpy as np

import tpcp._base

if TYPE_CHECKING:
    from tpcp._base import Algo

    from tpcp.base import BaseAlgorithm, BaseTpcpObject

_EMPTY = object()
_DEFAULT_PARA_NAME = "__TPCP_DEFAULT"


def _aggregate_final_results(results: List) -> Dict:
    """Aggregate the list of dict to dict of np ndarray/list.

    Modified based on sklearn.model_selection._validation._aggregate_score_dicts


    Parameters
    ----------
    results : list of dict
        List of dicts of the results for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------
    >>> results = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3}, {'a': 10, 'b': 10}]
    >>>
    >>> _aggregate_final_results(results)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}

    """
    return {
        key: np.asarray([score[key] for score in results])
        if isinstance(results[0][key], numbers.Number)
        else [score[key] for score in results]
        for key in results[0]
    }


def _normalize_score_results(scores: List, prefix="", single_score_key="score"):
    """Create a scoring dictionary based on the type of `scores`."""
    if isinstance(scores[0], dict):
        # multimetric scoring
        return {prefix + k: v for k, v in _aggregate_final_results(scores).items()}
    # single
    return {prefix + single_score_key: scores}


def _prefix_para_dict(params_dict: Optional[Dict], prefix="pipeline__") -> Optional[Dict]:
    """Add a prefix to all parameter names in the dictionary.

    This can be helpful to adjust a parameter grid that was originally created for a pipeline to work on a wrapper like
    `Optimize` using the `__` naming convention for nested objects.
    """
    if not params_dict:
        return None
    return {prefix + k: v for k, v in params_dict.items()}


def _get_nested_paras(param_dict: Optional[Dict], nested_object_name="pipeline") -> Dict:
    """Get the parameters belonging to a nested object and remove the suffix.

    If the parameter of a double nested object are required, use `level_1__level_1`.
    """
    if not param_dict:
        return {}
    return {k.split("__", 1)[1]: v for k, v in param_dict.items() if k.startswith(f"{nested_object_name}__")}


def _clone_parameter_dict(param_dict: Optional[Dict]) -> Dict:
    cloned_param_dict = {}
    if param_dict is not None:
        for k, v in param_dict.items():
            cloned_param_dict[k] = clone(v, safe=False)
    return cloned_param_dict


def _split_hyper_and_pure_parameters(
    param_dict: List[Dict], pure_parameters: Optional[List[str]]
) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """Split a list of parameters in hyper parameters and pure parameters.

    For each dictionary in the list, this separates the pure parameters (names provided in input) from all hyper
    parameters (remaining parameters).
    If either the none of the pure parameters is present in a parameter dict or all parameters are pure parameters,
    the pure or the hyper parameters are `None`.

    Returns
    -------
    split_parameters
        List of tuples `(hyper, pure)` for each of the para dicts in the input list.

    """
    if pure_parameters is None:
        return [(c, None) for c in param_dict]
    split_param_dict = []
    for c in param_dict:
        c = copy.copy(c)  # Otherwise we remove elements from the actual parameter list that is passed as input.
        tmp = {}
        for k in list(c.keys()):
            if k in pure_parameters:
                tmp[k] = c.pop(k)
        split_param_dict.append((c or None, tmp or None))
    return split_param_dict


def _check_safe_run(pipeline: BaseAlgorithm, old_method: Callable, *args, **kwargs):
    """Run the pipeline and check that run behaved as expected."""
    before_paras = pipeline.get_params()
    before_paras_hash = joblib.hash(before_paras)
    output: BaseAlgorithm = old_method(pipeline, *args, **kwargs)
    after_paras = pipeline.get_params()
    after_paras_hash = joblib.hash(after_paras)
    if not before_paras_hash == after_paras_hash:
        raise ValueError(
            f"Running `{old_method.__name__}` of {type(pipeline)} did modify the parameters of the algorithm. "
            "This must not happen to make sure individual runs of the pipeline are independent.\n\n"
            "This usually happens, when you use an algorithm object or other mutable objects as a parameter to your "
            "pipeline. "
            "In this case, make sure you call `algo_object.clone()` or more general `clone(mutable_input) on the "
            f"within the `{old_method.__name__}` method before modifying the mutable or running the nested algorithm."
        )
    if not isinstance(output, type(pipeline)):
        raise ValueError(
            f"The `{old_method.__name__}` method of {type(pipeline)} must return `self` or in rare cases a new "
            f"instance of {type(pipeline)}. "
            f"But the return value had the type {type(output)}."
        )
    if not output._action_is_applied:
        raise ValueError(
            f"Running the `{old_method.__name__}` method of {type(pipeline)} did not set any results on the output. "
            f"Make sure the `{old_method.__name__}` method sets the result values as expected as class attributes and "
            f"all names of result attributes have a trailing `_` to mark them as such."
        )
    return output


def safe_action(action_method: Callable):
    """Apply a set of runtime checks to a custom action method to prevent implementation errors.

    The following things are checked:

        - The action method must return `self` (or at least an instance of the algorithm or pipeline)
        - The action method must set result attributes on the pipeline
        - All result attributes must have a trailing `_` in their name
        - The action method must not modify the input parameters of the pipeline

    In general we recommend to just apply this decorator to all custom action methods.
    The runtime overhead is usually small enough to not make a difference.

    The only execption are custom pipelines.
    For pipelines, the `safe_run` method can be used as shortcut to call `run` with the same runtime checks,
    this decorator provides.

    Examples
    --------
    >>> from tpcp import BaseAlgorithm, safe_action
    >>> class MyAlgorithm(BaseAlgorithm):
    ...     _action_method = ("detect",)
    ...
    ...     @safe_action
    ...     def detect(self):
    ...         ...
    ...         return self

    """
    if getattr(action_method, "__is_safe_action", False):
        # It seems like the decorator was already applied and we do not want to apply it multiple times and run
        # duplicated checks.
        return action_method

    @wraps(action_method)
    def safe_wrapped(self: BaseAlgorithm, *args, **kwargs):
        if action_method.__name__ not in self._action_method:
            raise ValueError(
                f"The `safe_action` decorator can only be applied to the action methods ({self._get_action_methods()} "
                f"for {type(self)}) of an algorithm or methods. "
                f"To register an action method add the following to the class definition of {type(self)}:\n\n"
                f"`    _action_method = ({action_method.__name__},)`\n\n"
                "Or append it to the tuple, if it already exists."
            )
        return _check_safe_run(self, action_method, *args, **kwargs)

    safe_wrapped.__is_safe_action = True
    return safe_wrapped


def clone(
    algorithm: Union[BaseTpcpObject, List[BaseTpcpObject], Set[BaseTpcpObject], Tuple[BaseTpcpObject]],
    *,
    safe: bool = False,
):
    """Construct a new algorithm object with the same parameters.

    This is a modified version from sklearn and the original was published under a BSD-3 license and the original file
    can be found here: https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/base.py#L31

    The method creates a copy of tpcp algorithms and pipelines, without any results attached.
    I.e. it is equivalent to creating a new instance with the same parameter.
    For all objects that are not tpcp objects (or lists of them), a deepcopy is created.

    .. warning :: This function will not clone sklearn models as expected!
                  `sklearn.clone` will remove the trained model from sklearn object by creating a new instance.
                  This clone method, will deepcopy sklearn models.
                  This means fitted models will be copied and are still available afterwards.
                  For more information have a look at the documenation about "inputs and results" in tpcp.
                  TODO: Link

    Parameters
    ----------
    algorithm : {list, tuple, set} of algorithm instance or a single algorithm instance
        The algorithm or group of algorithms to be cloned.
    safe : bool, default=False
        If safe is False, clone will fall back to a deep copy on objects that are not algorithms.

    """
    if algorithm is _EMPTY:
        return _EMPTY
    # XXX: not handling dictionaries
    if isinstance(algorithm, (list, tuple, set, frozenset)):
        return type(algorithm)([clone(a, safe=safe) for a in algorithm])  # noqa: to-many-function-args
    # Compared to sklearn, we check specifically for _BaseSerializable and not just if `get_params` is defined on the
    # object.
    # Due to the way algorithms/pipelines in tpcp work, they need to inherit from _BaseSerializable.
    # Therefore, we check explicitly for that, as we do not want to accidentally treat an sklearn algo (or similar) as
    # algorithm
    if not isinstance(algorithm, tpcp.base.BaseTpcpObject):
        if not safe:
            return copy.deepcopy(algorithm)
        raise TypeError(
            f"Cannot clone object '{repr(algorithm)}' (type {type(algorithm)}): "
            "it does not seem to be a compatible algorithm class algorithm as it does not inherit from "
            "_BaseSerializable or BaseAlgorithm method."
        )

    klass = algorithm.__class__
    new_object_params = algorithm.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError(
                f"Cannot clone object {algorithm}, as the constructor either does not set or modifies parameter {name}"
            )
    return new_object


def default(algo: Algo) -> Algo:
    """Wrap nested algorithm arguments to mark them as default value.

    This is required, as algorithms by default are mutable.
    Hence, when one algo instance is used as default parameter for another algo instance, we have a mutable default,
    which is bad.

    We handle that by cloning default values on init.
    To mark a parameter to be cloned on init, it needs to be wrapped with this function.
    """
    setattr(algo, _DEFAULT_PARA_NAME, True)
    return algo
