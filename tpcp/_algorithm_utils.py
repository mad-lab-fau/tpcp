"""Basic utilities to work with algorithm and pipeline objects."""

from __future__ import annotations

import types
import warnings
from functools import wraps
from inspect import isclass
from pickle import PicklingError
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from typing_extensions import Concatenate, ParamSpec

from tpcp._base import _get_annotated_fields_of_type
from tpcp._hash import custom_hash
from tpcp._parameters import _ParaTypes
from tpcp.exceptions import PotentialUserErrorWarning

if TYPE_CHECKING:
    from tpcp import Algorithm, OptimizableAlgorithm, OptimizablePipeline
    from tpcp._algorithm import AlgorithmT

    OptimizableT = TypeVar("OptimizableT", OptimizablePipeline, OptimizableAlgorithm)

ACTION_METHOD_INDICATOR = "__tpcp_action_method"
OPTIMIZE_METHOD_INDICATOR = "__tpcp_optimize_method"


P = ParamSpec("P")


def get_action_method(instance: Algorithm, method_name: Optional[str] = None) -> Callable:
    """Get the action method for an Algorithm.

    If method_name is None, the primary action method is returned (the one listed first in `Algorithm._action_methods`).
    Otherwise, the action method belonging to the respective name is returned.
    """
    method_names = get_action_methods_names(instance)
    if method_name is not None:
        if method_name not in method_names:
            raise ValueError(
                "`method_name` must be one of the specified action methods of the algorithm. "
                f"Valid ones are {method_names}"
            )
    else:
        method_name = method_names[0]
    return getattr(instance, method_name)


def get_action_methods_names(instance_or_cls: Union[Type[Algorithm], Algorithm]) -> Tuple[str, ...]:
    """Get the names of all action methods of a class.

    This basically returns `instance_or_cls._action_method`, but ensures that the return type is a tuple.
    """
    method_names = instance_or_cls._action_methods
    if isinstance(method_names, str):
        method_names = (method_names,)
    if not isinstance(method_names, tuple) and len(method_names) == 0:
        if isclass(instance_or_cls):
            instance_or_cls = cast(Type[Algorithm], instance_or_cls)
            name = instance_or_cls.__name__
        else:
            name = type(instance_or_cls).__name__
        raise ValueError(f"`_action_methods` of {name} must either be a string or a tuple of strings.")
    return method_names


def get_action_params(instance: Algorithm) -> Dict[str, Any]:
    """Get all "Action Params" / "Other Parameters" of the Algorithm.

    Action params are all parameters passed as input to the action method.

    Note: We do not magically set these values on the algorithm instance. Instead, the developer of the algorithms must
    implement the algorithm to follow this convention.

    In general, this function is not that useful but might be used for debugging purposes.

    Returns
    -------
    params : dict
        Parameter names mapped to their values.

    """
    params = instance.get_params()
    attrs = {
        v: getattr(instance, v)
        for v in vars(instance)
        if not v.endswith("_") and not v.startswith("_") and v not in params
    }
    return attrs


def get_results(instance: Algorithm) -> Dict[str, Any]:
    """Get all Results of the Algorithm.

    "Results" or "Attributes" are all values considered results of the algorithm.
    They are indicated by a trailing "_" in their name.
    The values are only populated after the action method of the algorithm was called.

    Returns
    -------
    params : dict
        Parameter names mapped to their values.

    Raises
    ------
    AttributeError
        If one or more of the attributes are not retrievable from the instance.
        This usually indicates that the action method was not called yet.

    """
    all_attributes = dir(instance)
    attrs = {
        v: getattr(instance, v)
        for v in all_attributes
        if v.endswith("_") and not v.startswith("__") and not isinstance(getattr(instance, v), types.MethodType)
    }
    return attrs


def is_action_applied(instance: Algorithm) -> bool:
    """Check if the action method was already called/results were generated."""
    if len(get_results(instance)) == 0:
        return False
    return True


def _check_safe_run(algorithm: AlgorithmT, old_method: Callable, *args: Any, **kwargs: Any) -> AlgorithmT:
    """Run the pipeline and check that run behaved as expected."""
    before_paras = algorithm.get_params()
    before_paras_hash = custom_hash(before_paras)
    output: AlgorithmT
    if hasattr(old_method, "__self__"):
        # In this case the method is already bound and we do not need to pass the algo as first argument
        output = old_method(*args, **kwargs)
    else:
        output = old_method(algorithm, *args, **kwargs)
    after_paras = algorithm.get_params()
    after_paras_hash = custom_hash(after_paras)
    if not before_paras_hash == after_paras_hash:
        raise ValueError(
            f"Running `{old_method.__name__}` of {type(algorithm).__name__} did modify the parameters of the "
            "algorithm. "
            "This must not happen to make sure individual runs of the algorithm/pipeline are independent.\n\n"
            "This usually happens when you use an algorithm object or other mutable objects as a parameter to your "
            "algorithm/pipeline. "
            "In this case, make sure you call `algo_object.clone()` or more general `clone(mutable_input)` on the "
            f"object within the `{old_method.__name__}` method before modifying the mutable or running the nested "
            "algorithm."
        )
    if not isinstance(output, type(algorithm)):
        raise ValueError(
            f"The `{old_method.__name__}` method of {type(algorithm).__name__} must return `self` or in rare cases a "
            f"new instance of {type(algorithm).__name__}. "
            f"But the return value had the type {type(output)}."
        )
    if not is_action_applied(output):
        raise ValueError(
            f"Running the `{old_method.__name__}` method of {type(algorithm).__name__} did not set any results on the "
            "output. "
            f"Make sure the `{old_method.__name__}` method sets the result values as expected as class attributes and "
            f"all names of result attributes have a trailing `_` to mark them as such."
        )
    return output


def make_action_safe(
    action_method: Callable[Concatenate[AlgorithmT, P], AlgorithmT]
) -> Callable[Concatenate[AlgorithmT, P], AlgorithmT]:
    """Mark a method as an "action" and apply a set of runtime checks to prevent implementation errors.

    This decorator marks a method as action.
    Each algorithm is expected to have at least one action method.
    For pipelines this action method is called "run".
    This means, when implementing a custom action or run method, it must always be wrapped in this decorator.

    Besides registering the method, the following things are checked at runtime:

        - The action method must return `self` (or at least an instance of the algorithm or pipeline)
        - The action method must set result attributes on the pipeline
        - All result attributes must have a trailing `_` in their name
        - The action method must not modify the input parameters of the pipeline

    In general, we recommend to just apply this decorator to all custom action methods.
    The runtime overhead is usually small enough to not make a difference.

    Examples
    --------
    >>> from tpcp import Algorithm, make_action_safe
    >>> class MyAlgorithm(Algorithm):
    ...
    ...     @make_action_safe
    ...     def detect(self, data, sampling_rate_hz):
    ...         ...
    ...         return self

    """
    if getattr(action_method, ACTION_METHOD_INDICATOR, False) is True:
        # It seems like the decorator was already applied and we do not want to apply it multiple times and run
        # duplicated checks.
        return action_method

    @wraps(action_method)
    def safe_wrapped(self: AlgorithmT, *args: P.args, **kwargs: P.kwargs) -> AlgorithmT:
        if action_method.__name__ not in get_action_methods_names(self):
            warnings.warn(
                "The `make_action_safe` decorator should only be applied to an action method "
                f"({get_action_methods_names(self)} for {type(self)}) of an algorithm or methods. "
                f"To register an action method add the following to the class definition of {type(self)}:\n\n"
                f"`    _action_methods = ({action_method.__name__},)`\n\n"
                "Or append it to the tuple, if it already exists.",
                PotentialUserErrorWarning,
            )
        return _check_safe_run(self, action_method, *args, **kwargs)

    setattr(safe_wrapped, ACTION_METHOD_INDICATOR, True)
    return safe_wrapped


def _get_nested_opti_paras(algorithm: Algorithm, opti_para_names: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paras = algorithm.get_params(deep=True)
    optimizable_paras = {}
    other_paras = {}
    for p, v in paras.items():
        if p in opti_para_names:
            optimizable_paras[p] = v
        # For each optimizable parameter, we also add all children, as they are also allowed to change,
        # if the parent is allowed to.
        elif any(p.startswith(o + "__") for o in opti_para_names):
            optimizable_paras[p] = v
        else:
            other_paras[p] = v
    # We need to exclude "parent" objects, when a nested para is marked as optimizable
    # Because, if the nested para changes, the parent para will change as well, and we can not do anything about it.
    for p in optimizable_paras:
        parent_name = p.rsplit("__", 1)[0]
        other_paras.pop(parent_name, None)

    return optimizable_paras, other_paras


def _check_safe_optimize(algorithm: OptimizableT, old_method: Callable, *args: Any, **kwargs: Any) -> OptimizableT:
    # record the hash of the pipeline to make an educated guess if the optimization works
    opti_para_names = _get_annotated_fields_of_type(algorithm, _ParaTypes.OPTI)
    optimizable_paras, other_paras = _get_nested_opti_paras(algorithm, opti_para_names)
    if len(optimizable_paras) == 0:
        raise ValueError(
            f"No parameter of {type(algorithm).__name__} was marked as optimizable. "
            "Mark at least one; parameter with the `OptiPara`/`OptimizablePara` annotation to use "
            "`self_optimize`."
        )
    before_hash_optimizable = custom_hash(optimizable_paras)
    before_hash_other = custom_hash(other_paras)
    optimized_algorithm: OptimizableT
    if hasattr(old_method, "__self__"):
        # In this case the method is already bound and we do not need to pass the algo as first argument
        optimized_algorithm = old_method(*args, **kwargs)
    else:
        optimized_algorithm = old_method(algorithm, *args, **kwargs)
    if not isinstance(optimized_algorithm, type(algorithm)):
        raise ValueError(
            "Calling `self_optimize` did not return an instance of the algorithm/pipeline itself! "
            "Normally, this method should return `self`."
        )
    # We calculate the hash afterwards twice.
    # Once directly after the optimization and once after cloning.
    # The first hash records any changes to the object.
    # The second hash only records changes to the parameters, because everything else is removed by clone.
    # Hence, if we see differences between the hashes, other things besides the parameters are changed.
    after_hash = custom_hash(optimized_algorithm)
    after_hash_after_clone = custom_hash(optimized_algorithm.clone())
    if after_hash_after_clone != after_hash:
        raise RuntimeError(
            "Optimizing seems to have changed class attributes that are not parameters (i.e. not provided in the "
            "`__init__`). "
            "This can lead to unexpected issues!"
        )
    # Now we need to check, which parameters have been modified.
    # We only expect/allow parameters that are marked as "Optimizable".
    # Therefore, we calculate the hash of all other parameters and check if they have changed.
    # We also consider parameter changed, that did not exist or were completely removed.
    # Most of the complicated magic here is in _get_nested_opti_paras.
    # It takes care of including and excluding the correct parameters in the other list, even if nested paras are
    # marked as "Optimizable"
    after_optimizable_paras, after_other_paras = _get_nested_opti_paras(algorithm, opti_para_names)
    after_hash_optimizable = custom_hash(after_optimizable_paras)
    after_hash_other = custom_hash(after_other_paras)
    if before_hash_other != after_hash_other:
        # In this case we raise an error anyway, so lets go deep:
        removed_paras = set(other_paras) - set(after_other_paras)
        added_paras = set(after_other_paras) - set(other_paras)
        changed_paras = []
        for k in set(other_paras) - set(removed_paras):
            if custom_hash(other_paras[k]) != custom_hash(after_other_paras[k]):
                changed_paras.append(k)
        changed_paras = sorted(changed_paras)
        changed_paras.extend([f"{p} (removed)" for p in sorted(removed_paras)])
        changed_paras.extend([f"{p} (added)" for p in sorted(added_paras)])
        raise RuntimeError(
            "Optimizing the pipeline has modified the following parameters, that were not marked as optimizable: "
            f"{changed_paras}. "
            "Double check the implementation of `self_optimize` and either mark the changing parameters as "
            "optimizable by adding `OptiPara`/`OptimizableParameter` as type annotation or make sure that they are not "
            "accidentally modified in your implementation."
        )
    if before_hash_optimizable == after_hash_optimizable:
        # If the hash didn't change the object didn't change.
        # Something might have gone wrong.
        warnings.warn(
            "Optimizing the algorithm doesn't seem to have changed any of the parameters marked as optimizable "
            f"({optimizable_paras}). "
            "This could indicate an implementation error of the `self_optimize` method.",
            PotentialUserErrorWarning,
        )
    return optimized_algorithm


def make_optimize_safe(
    self_optimize_method: Callable[Concatenate[OptimizableT, P], OptimizableT]
) -> Callable[Concatenate[OptimizableT, P], OptimizableT]:
    """Apply a set of runtime checks to a custom `self_optimize` method to prevent implementation errors.

    The following things are checked:

        - The `self_optimize` method must return `self` (or at least an instance of the algorithm or pipeline).
        - The `self_optimize` method must only modify input parameters of the pipeline and not any other attributes.
        - The `self_optimize` method should modify at least one of the input parameters (this doesn't raise an error,
          but just a warning).

    In general, we recommend to just apply this decorator to all custom `self_optimize` methods.
    The runtime overhead is usually small enough to not make a difference.

    The only execption are custom pipelines that you only optimize using the :class:`~tpcp.optimize.Optimize` wrapper.
    This wrapper will apply the same runtime checks anyway.
    However, it doesn't hurt to apply it as decorator as well.
    We make sure that the cheks will still only be performed once.

    Examples
    --------
    >>> from tpcp import OptimizableAlgorithm, make_optimize_safe
    >>> class MyAlgorithm(OptimizableAlgorithm):
    ...     def __init__(self, para_1: int = 4):
    ...         self.para_1 = para_1
    ...
    ...     @make_optimize_safe
    ...     def self_optimize(self, train_data, **kwargs):
    ...         # find a better value for para_1 based on the provided trainings data
    ...         better_value_for_para_1 = 5
    ...         self.para_1 = better_value_for_para_1
    ...         return self

    """
    if getattr(self_optimize_method, OPTIMIZE_METHOD_INDICATOR, False) is True:
        # It seems like the decorator was already applied, and we do not want to apply it multiple times and run
        # duplicated checks.
        return self_optimize_method

    @wraps(self_optimize_method)
    def safe_wrapped(self: OptimizableT, *args: P.args, **kwargs: P.kwargs) -> OptimizableT:
        if self_optimize_method.__name__ != "self_optimize":
            warnings.warn(
                "The `make_optimize_safe` decorator is only meant for the `self_optimize` method, but you applied it "
                f"to the `{self_optimize_method.__name__}` method.",
                PotentialUserErrorWarning,
            )
        try:
            return _check_safe_optimize(self, self_optimize_method, *args, **kwargs)
        except PicklingError as e:
            raise ValueError(
                "We had trouble hashing your class instance."
                "This is required to run the safety checks for the optimize method. "
                "This usually happens, if your pipeline or algorithm or one of its parameters is based "
                "on a dynamically defined class (e.g. a class defined within a function). "
                "Try defining your classes on a module level. "
                "If this is not possible for you, you need to disable the safety checks."
            ) from e

    setattr(safe_wrapped, OPTIMIZE_METHOD_INDICATOR, True)
    return safe_wrapped
