"""Public base classes for all algorithms and pipelines."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Callable, Dict

from tpcp._base import _BaseTpcpObject
from tpcp._meta import AlgorithmMeta

if TYPE_CHECKING:
    from tpcp.dataset import Dataset
    from tpcp.pipelines import SimplePipeline


class BaseTpcpObject(_BaseTpcpObject, metaclass=AlgorithmMeta):
    """Baseclass for all tpcp objects."""


class BaseAlgorithm(BaseTpcpObject):
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
