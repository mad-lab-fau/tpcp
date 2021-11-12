"""Private base classes for tpcp.

These classes are in a separate module to avoid circular imports.
In basically all cases, you do not need them.
"""
from __future__ import annotations

import inspect
import json
import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Type, TypeVar, Union

import numpy as np
import pandas as pd
from joblib import Memory

from tpcp._utils._general import clone

BaseType = TypeVar("BaseType", bound="_BaseSerializable")


class _CustomEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: method-hidden
        if isinstance(o, _BaseSerializable):
            return o._to_json_dict()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return dict(_obj_type="Array", array=o.tolist())
        if isinstance(o, pd.DataFrame):
            return dict(_obj_type="DataFrame", df=o.to_json(orient="split"))
        if isinstance(o, pd.Series):
            return dict(_obj_type="Series", df=o.to_json(orient="split"))
        if isinstance(o, Memory):
            warnings.warn(
                "Exporting `joblib.Memory` objects to json is not supported. "
                "The value will be replaced by `None` and caching needs to be reactivated after loading the "
                "object again. "
                "This can be using `instance.set_params(memory=Memory(...))`"
            )
            return None
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def _custom_deserialize(json_obj):
    if "_tpcp_obj" in json_obj:
        return _BaseSerializable._find_subclass(json_obj["_tpcp_obj"])._from_json_dict(json_obj)
    return json_obj


class _BaseSerializable:
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
    def _get_subclasses(cls: Type[BaseType]):
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def _find_subclass(cls: Type[BaseType], name: str) -> Type[BaseType]:
        for subclass in _BaseSerializable._get_subclasses():
            if subclass.__name__ == name:
                return subclass
        raise ValueError(f"No algorithm class with name {name} exists")

    @classmethod
    def _from_json_dict(cls: Type[BaseType], json_dict: Dict) -> BaseType:
        params = json_dict["params"]
        input_data = {k: params[k] for k in cls._get_param_names() if k in params}
        instance = cls(**input_data)
        return instance

    def _get_params_without_nested_class(self) -> Dict[str, Any]:
        return {k: v for k, v in self.get_params().items() if not isinstance(v, _BaseSerializable)}

    def _to_json_dict(self) -> Dict[str, Any]:
        json_dict: Dict[str, Union[str, Dict[str, Any]]] = {
            "_tpcp_obj": self.__class__.__name__,
            "params": self.get_params(deep=False),
        }
        return json_dict

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
            if deep and isinstance(value, _BaseSerializable):
                deep_items = value.get_params(deep=True).items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self: BaseType, **params: Any) -> BaseType:
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

    def clone(self: BaseType) -> BaseType:
        """Create a new instance of the class with all parameters copied over.

        This will create a new instance of the class itself and all nested objects
        """
        return clone(self, safe=True)

    def to_json(self) -> str:
        """Export the current object parameters as json.

        For details have a look at the this :ref:`example <algo_serialize>`.

        You can use the `from_json` method of any tpcp algorithm to load the object again.

        .. warning:: This will only export the Parameters of the instance, but **not** any results!

        """
        final_dict = self._to_json_dict()
        return json.dumps(final_dict, indent=4, cls=_CustomEncoder)

    @classmethod
    def from_json(cls: Type[BaseType], json_str: str) -> BaseType:
        """Import an tpcp object from its json representation.

        For details have a look at the this :ref:`example <algo_serialize>`.

        You can use the `to_json` method of a class to export it as a compatible json string.

        Parameters
        ----------
        json_str
            json formatted string

        """
        instance = json.loads(json_str, object_hook=_custom_deserialize)
        return instance
