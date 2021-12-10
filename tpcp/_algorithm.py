"""Public base classes for all algorithms and pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

from tpcp._algorithm_utils import make_optimize_safe
from tpcp._base import Algo, BaseTpcpObject, _has_all_defaults, _get_init_defaults

if TYPE_CHECKING:
    from tpcp import Dataset, Pipeline


class Optimizable:
    """Mixin class to mark an object as optimizable."""

    def self_optimize(self: Algo, dataset: Dataset, **kwargs) -> Algo:
        """Optimize the input parameter of the pipeline or algorithm using any logic.

        This method can be used to adapt the input parameters (values provided in the init) based on any data driven
        heuristic.

        Note that the optimizations must only modify the input parameters (aka `self.clone` should retain the
        optimization results).

        Parameters
        ----------
        dataset
            An instance of a :class:`tpcp.dataset.Dataset` containing one or multiple data points that can
            be used for training.
            The structure of the data and the available reference information will depend on the dataset.
        kwargs
            Additional parameter required for the optimization process.

        Returns
        -------
        self
            The class instance with optimized input parameters.

        """
        raise NotImplementedError()  # pragma: no cover


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

    def __init_subclass__(cls, _allow_non_defaults: bool = False, _skip_validation: bool = False, **kwargs):
        """Initialize all algorithm subclasses.

        Compared to all the normal checks, we also check, if all the input parameters have sensible defaults.
        This is something we expect from algorithms, so that they can bve run without providing any parameters
        explicitly.
        """
        super().__init_subclass__(**kwargs)

        if _skip_validation is not True and _allow_non_defaults is not True:
            fields = _get_init_defaults(cls)
            _has_all_defaults(fields, cls)


class OptimizableAlgorithm(Algorithm, _skip_validation=True):
    """Base class for algorithms with distinct parameter optimization."""

    @make_optimize_safe
    def self_optimize(self, *args, **kwargs):
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

    dataset: Dataset

    optimized_pipeline_: Pipeline

    def __init_subclass__(cls, _allow_non_defaults: bool = True, **kwargs):
        """Initialize all Optimizer Subclasses."""
        # Optimizers usually have the pipeline they optimize as first parameter.
        # Therefore, this parameter is allowed to have no default.
        super().__init_subclass__(_allow_non_defaults=_allow_non_defaults, **kwargs)

    def optimize(self, dataset: Dataset, **optimize_params):
        """Apply some form of optimization on the the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self, datapoint: Dataset):
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.run(datapoint)
