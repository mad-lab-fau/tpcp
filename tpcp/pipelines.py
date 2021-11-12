"""Base Classes for custom pipelines."""
from typing import Dict, TypeVar, Union

from tpcp._utils._general import _check_safe_run
from tpcp.base import BaseAlgorithm
from tpcp.dataset import Dataset

Self = TypeVar("Self", bound="SimplePipeline")


class SimplePipeline(BaseAlgorithm):
    """Baseclass for all custom pipelines.

    To create your own custom pipeline, subclass this class and implement `run`.
    """

    dataset_single: Dataset

    _action_method = "run"

    def run(self: Self, datapoint: Dataset) -> Self:
        """Run the pipeline.

        Note, that it is usually preferred to use `safe_run` on custom pipelines instead of `run`, as `safe_run` can
        catch certain implementation errors of the run method.

        Parameters
        ----------
        datapoint
            An instance of a :class:`tpcp.dataset.Dataset` containing only a single datapoint.
            The structure of the data will depend on the dataset.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        raise NotImplementedError()  # pragma: no cover

    def safe_run(self: Self, datapoint: Dataset) -> Self:
        """Run the pipeline with some additional checks.

        It is preferred to use this method over `run`, as it can catch some simple implementation errors of custom
        pipelines.

        The following things are checked:

        - The run method must return `self` (or at least an instance of the pipeline)
        - The run method must set result attributes on the pipeline
        - All result attributes must have a trailing `_` in their name
        - The run method must not modify the input parameters of the pipeline

        Parameters
        ----------
        datapoint
            An instance of a :class:`tpcp.dataset.Dataset` containing only a single datapoint.
            The structure of the data will depend on the dataset.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        return _check_safe_run(self, datapoint)

    def score(self, datapoint: Dataset) -> Union[float, Dict[str, float]]:
        """Calculate performance of the pipeline on a datapoint with reference information.

        This is an optional method and does not need to be implemented in many cases.
        Usually stand-a-lone functions are better suited as scorer.

        A typical score method will call `self.run(datapoint)` and then compare the results with reference values
        also available on the dataset.

        Parameters
        ----------
        datapoint
            An instance of a :class:`tpcp.dataset.Dataset` containing only a single datapoint.
            The structure of the data and the available reference information will depend on the dataset.

        Returns
        -------
        score
            A float or dict of float quantifying the quality of the pipeline on the provided data.
            A higher score is always better.

        """
        raise NotImplementedError()  # pragma: no cover


class OptimizablePipeline(SimplePipeline):
    """Pipeline with custom ways to optimize and/or train input parameters.

    OptimizablePipelines are expected to implement a concrete way to train internal models or optimize parameters.
    This should not be a reimplementation of GridSearch or similar methods.
    For this :class:`tpcp.pipelines.GridSearch` should be used directly.

    It is important that `self_optimize` only modifies input parameters of the pipeline.
    This means, if a parameter is optimized, by `self_optimize` it should be named in the `__init__` and should be
    exportable when calling `pipeline.get_params`.
    It is also possible to optimize nested parameters.
    For example, if the input of the pipeline is an algorithm or another pipeline on its own, all parameters of these
    objects can be modified as well.
    In any case, you should make sure that all optimized parameters are still there if you call `.clone()` on the
    optimized pipeline.
    """

    def self_optimize(self: Self, dataset: Dataset, **kwargs) -> Self:
        """Optimize the input parameter of the pipeline using any logic.

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
