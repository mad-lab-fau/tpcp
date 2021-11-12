"""Private base classes for tpcp.

These classes are in a separate module to avoid circular imports.
In basically all cases, you do not need them.
"""
from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Type, TypeVar

import tpcp._utils._general as gen_utils

Algo = TypeVar("Algo", bound="_BaseTpcpObject")


class _BaseTpcpObject:
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

    @classmethod
    def _get_subclasses(cls: Type[Algo]):
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def _find_subclass(cls: Type[Algo], name: str) -> Type[Algo]:
        for subclass in _BaseTpcpObject._get_subclasses():
            if subclass.__name__ == name:
                return subclass
        raise ValueError(f"No algorithm class with name {name} exists")

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
