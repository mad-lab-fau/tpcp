"""Private base classes for tpcp.

These classes are in a separate module to avoid circular imports.
In basically all cases, you do not need them.
"""
from __future__ import annotations

import inspect
from collections import defaultdict
from functools import wraps
from inspect import Parameter
from typing import Any, DefaultDict, Dict, List, Tuple, TypeVar

import tpcp._utils._general as gen_utils
from tpcp._utils._general import _DEFAULT_PARA_NAME
from tpcp.exceptions import MutableDefaultsError

Algo = TypeVar("Algo", bound="_BaseTpcpObject")


class _BaseTpcpObject:
    def __init_subclass__(cls, safe=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if safe is True:
            # This allows users to disable all "metaclass" magic and this is also used for library internal base
            # objects to reduce the number of operations that are performed on start up.
            return
        # For the type checker
        assert issubclass(cls, _BaseTpcpObject)
        init_defaults = cls._get_init_defaults()
        if init_defaults:
            listed_mutables = {k: (v.default, _is_dangerous_mutable(v)) for k, v in init_defaults.items()}
            # Mutables not wrapped in default
            unwrapped_mutables = {k: v[0] for k, v in listed_mutables.items() if v[1] is True}
            # Mutables that are wrapped in default
            wrapped_mutables = {k: v[0] for k, v in listed_mutables.items() if v[1] is False}
            if len(unwrapped_mutables) > 0:
                raise MutableDefaultsError(
                    f"The class {cls.__name__} contains mutable objects as default values ({unwrapped_mutables}). "
                    "This can lead to unexpected and unpleasant issues! "
                    "To solve this issue wrap your mutable default arguments explicitly with `default` "
                    "(or `df`). "
                    "This will enforce a clone/copy of the respective input, whenever a new instance of "
                    "your Algorithm or Pipeline is created. "
                    "\n"
                    "Note, that we do not check for all cases of mutable objects. "
                    f"At the moment, we check only for {_get_dangerous_mutable_types()}. "
                    "To learn more about this topic, check TODO: LINK."
                )
            # Finally we wrap the init to replace the marked mutables on each class init.
            # We only do that when there are actual arguments that need to be wrapped.
            if wrapped_mutables and cls.__init__ is not object.__init__:
                cls.__init__ = _replace_defaults_wrapper(cls.__init__)

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """Get parameter names for the estimator.

        The parameters of an algorithm are defined based on its `__init__` method.
        All parameters of this method are considered parameters of the algorithm.

        Notes
        -----
        Adopted based on `sklearn BaseEstimator._get_param_names`.

        Returns
        -------
        param_names
            List of parameter names of the algorithm

        """
        parameters = list(cls._get_init_defaults().values())
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "tpcp-algorithms and pipeline should always specify their parameters in the signature of their "
                    f"__init__ (no varargs). {cls} doesn't follow this convention."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    @classmethod
    def _get_init_defaults(cls) -> Dict[str, inspect.Parameter]:
        # fetch the constructor or the original constructor before deprecation wrapping if any
        init = cls.__init__
        if init is object.__init__:
            # No explicit constructor to introspect
            return {}

        # introspect the constructor arguments to find the model parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        defaults = {k: p for k, p in init_signature.parameters.items() if p.name != "self" and p.kind != p.VAR_KEYWORD}
        return defaults

    def _get_params_without_nested_class(self) -> Dict[str, Any]:
        return {k: v for k, v in self.get_params().items() if not isinstance(v, _BaseTpcpObject)}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this algorithm.

        Parameters
        ----------
        deep
            Only relevant if object contains nested algorithm objects.
            If this is the case and deep is True, the params of these nested objects are included in the output using a
            prefix like `nested_object_name__` (Note the two "_" at the end)

        Returns
        -------
        params
            Parameter names mapped to their values.

        """
        # Basically copied from sklearn
        out: Dict[str, Any] = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and isinstance(value, _BaseTpcpObject):
                deep_items = value.get_params(deep=True).items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self: Algo, **params: Any) -> Algo:
        """Set the parameters of this Algorithm.

        To set parameters of nested objects use `nested_object_name__para_name=`.
        """
        # Basically copied from sklearn
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params: DefaultDict[str, Any] = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(f"`{key}` is not a valid parameter name for {self.__class__.__name__}.")

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self

    def clone(self: Algo) -> Algo:
        """Create a new instance of the class with all parameters copied over.

        This will create a new instance of the class itself and all nested objects
        """
        return gen_utils.clone(self, safe=True)


def _get_dangerous_mutable_types() -> Tuple[type]:
    # TODO: Update this list or even make it a white list?
    return (_BaseTpcpObject,)


def _is_dangerous_mutable(para: Parameter):
    """Check if a parameter is one of the mutable objects "considered" dangerous.

    If None, it is not a mutable from the list
    If True, it is a mutable not wrapped in default
    If False, it is a mutable from the list, but not dangerous, because it is wrapped in `default`.
    """
    val = para.default
    if not isinstance(val, _get_dangerous_mutable_types()):
        return None
    if getattr(val, _DEFAULT_PARA_NAME, None):
        return False
    return True


def _replace_defaults_wrapper(old_init):
    """Decorate an init to create new instances of mutable defaults.

    This should only be used in combiantion with `default` and will be applied as part of `__init_subclass`.
    Direct usage of this decorator should not be required.
    """

    @wraps(old_init)
    def new_init(self, *args, **kwargs):
        # call the old init.
        old_init(self, *args, **kwargs)
        assert isinstance(self, _BaseTpcpObject)  # For the type checker
        # Check if any of the initial values has a "default parameter flag".
        # If yes we replace it with a clone (in case of a tpcp object) or a deepcopy in case of other objects.
        for k, v in self.get_params(deep=False).items():
            if getattr(v, _DEFAULT_PARA_NAME, None):
                cloned_object = gen_utils.clone(v)
                # In case we made a deepcopy, we need to remove the default param.
                try:
                    delattr(cloned_object, _DEFAULT_PARA_NAME)
                except AttributeError:
                    pass
                setattr(self, k, cloned_object)

    return new_init
