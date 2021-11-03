"""Base classes for all algorithms and pipelines."""
from __future__ import annotations
import inspect
import json
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Type, TypeVar, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Memory

if TYPE_CHECKING:
    from tpcp.pipelines import SimplePipeline
    from tpcp.dataset import Dataset

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
        # fetch the constructor or the original constructor before deprecation wrapping if any
        init = cls.__init__
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "tpcp-algorithms and pipeline should always specify their parameters in the signature of their "
                    f"__init__ (no varargs). {cls} with constructor {init_signature} doesn't follow this convention."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

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
        cloned_dict = self.get_params(deep=False)
        for k, v in cloned_dict.items():
            if isinstance(v, _BaseSerializable):
                cloned_dict[k] = v.clone()
        return self.__class__(**cloned_dict)

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


class BaseAlgorithm(_BaseSerializable):
    """Base class for all algorithms.

    All type-specific algorithm classes should inherit from this class and need to

    1. overwrite `_action_method` with the name of the actual action method of this class type
    2. implement a stub for the action method

    Attributes
    ----------
    _action_method
        The name of the action method used by the Childclass

    """

    _action_method: str

    @property
    def _action_is_applied(self) -> bool:
        """Check if the action method was already called/results were generated."""
        if len(self.get_attributes()) == 0:
            return False
        return True

    def _get_action_method(self) -> Callable:
        """Get the action method as callable.

        This is intended to be used by wrappers, that do not know the Type of an algorithm
        """
        return getattr(self, self._action_method)

    def get_other_params(self) -> Dict[str, Any]:
        """Get all "Other Parameters" of the Algorithm.

        "Other Parameters" are all parameters set outside of the `__init__` that are not considered results.
        This usually includes the "data" and all other parameters passed to the action method.

        Returns
        -------
        params
            Parameter names mapped to their values.

        """
        params = self.get_params()
        attrs = {
            v: getattr(self, v) for v in vars(self) if not v.endswith("_") and not v.startswith("_") and v not in params
        }
        return attrs

    def get_attributes(self) -> Dict[str, Any]:
        """Get all Attributes of the Algorithm.

        "Attributes" are all values considered results of the algorithm.
        They are indicated by a trailing "_" in their name.
        The values are only populated after the action method of the algorithm was called.

        Returns
        -------
        params
            Parameter names mapped to their values.

        Raises
        ------
        AttributeError
            If one or more of the attributes are not retrievable from the instance.
            This usually indicates that the action method was not called yet.

        """
        all_attributes = dir(self)
        attrs = {
            v: getattr(self, v)
            for v in all_attributes
            if v.endswith("_") and not v.startswith("__") and not isinstance(getattr(self, v), types.MethodType)
        }
        return attrs


class BaseOptimize(BaseAlgorithm):
    """Base class for all optimizer."""

    pipeline: SimplePipeline

    dataset: Dataset

    optimized_pipeline_: SimplePipeline

    _action_method = "optimize"

    def optimize(self, dataset: Dataset, **optimize_params):
        """Apply some form of optimization on the the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self, datapoint: Dataset):
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.run(datapoint)

    def safe_run(self, datapoint: Dataset):
        """Call the safe_run method of the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.safe_run(datapoint)

    def score(self, datapoint: Dataset):
        """Execute score on the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.score(datapoint)