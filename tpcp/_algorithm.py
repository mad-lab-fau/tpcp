"""Public base classes for all algorithms and pipelines."""

from __future__ import annotations

from typing import Any, Tuple, TypeVar, Union

from tpcp._algorithm_utils import make_optimize_safe
from tpcp._base import BaseTpcpObject

Algorithm_ = TypeVar("Algorithm_", bound="Algorithm")


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
