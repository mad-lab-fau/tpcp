"""Public base classes for all algorithms and pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, TypeVar, Union

from tpcp._algorithm_utils import make_optimize_safe
from tpcp._base import BaseTpcpObject
from tpcp._parameters import Parameter

if TYPE_CHECKING:
    from tpcp import Dataset, Pipeline

Algorithm_ = TypeVar("Algorithm_", bound="Algorithm")
BaseOptimize_ = TypeVar("BaseOptimize_", bound="BaseOptimize")


class Algorithm(BaseTpcpObject, _skip_validation=True):
    """Base class for all algorithms.

    All type-specific algorithm classes should inherit from this class and need to

    1. overwrite `_action_method` with the name of the actual action method of this class type
    2. implement a stub for the action method

    Attributes
    ----------
    _action_method
        The name of the action method used by the Childclass

    """

    _action_methods: Union[Tuple[str, ...], str]


class OptimizableAlgorithm(Algorithm, _skip_validation=True):
    """Base class for algorithms with distinct parameter optimization."""

    @make_optimize_safe
    def self_optimize(self: Algorithm_, *args: Any, **kwargs: Any) -> Algorithm_:
        """Optimize the input parameter of the algorithm using any logic.

        This method can be used to adapt the input parameters (values provided in the init) based on any data driven
        heuristic.

        Note that the optimizations must only modify the input parameters that are marked as
        `optiparas`/`optimizable_parameters`.

        Returns
        -------
        self
            The class instance with optimized input parameters.

        """
        raise NotImplementedError()


class BaseOptimize(Algorithm, _skip_validation=True):
    """Base class for all optimizer."""

    _action_methods: Union[Tuple[str, ...], str] = "optimize"

    pipeline: Parameter[Pipeline]

    dataset: Dataset

    optimized_pipeline_: Pipeline

    def optimize(self: BaseOptimize_, dataset: Dataset, **optimize_params: Any) -> BaseOptimize_:
        """Apply some form of optimization on the the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self: BaseOptimize_, datapoint: Dataset) -> BaseOptimize_:
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.run(datapoint)
