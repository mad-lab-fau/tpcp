"""Public base classes for all algorithms and pipelines."""

from __future__ import annotations

from typing import ClassVar, Tuple, TypeVar, Union

from tpcp._base import BaseTpcpObject

AlgorithmT = TypeVar("AlgorithmT", bound="Algorithm")


class Algorithm(BaseTpcpObject):
    """Base class for all algorithms.

    All type-specific algorithm classes should inherit from this class and need to

    1. overwrite `_action_method` with the name of the actual action method of this class type
    2. implement a stub for the action method

    If you want to create an optimizable algorithm, add a `self_optimize` or (`self_optimize_with_info`) method to your
    class.
    We do not provide a separate base class for that, as we can make no assumptions about the call signature of your
    custom `self_optimize` method.
    If you need an "optimizable" version for a group of algorithms you are working with, create a custom
    `OptimizableAlgorithm` class or `OptimizableAlgorithmMixing` that is specific to your algorithm.

    Attributes
    ----------
    _action_methods
        The name(s) of the action method used by the child class

    """

    _action_methods: ClassVar[Union[Tuple[str, ...], str]]
