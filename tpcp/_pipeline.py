"""Base Classes for custom pipelines."""

from typing import Any, ClassVar, Generic, TypeVar

from typing_extensions import Self

from tpcp import NOTHING
from tpcp._algorithm import Algorithm
from tpcp._algorithm_utils import ACTION_METHOD_INDICATOR, _check_safe_run, make_action_safe
from tpcp._dataset import DatasetT

PipelineT = TypeVar("PipelineT", bound="Pipeline")
OptimizablePipelineT = TypeVar("OptimizablePipelineT", bound="OptimizablePipeline")


class Pipeline(Algorithm, Generic[DatasetT]):
    """Baseclass for all custom pipelines.

    To create your own custom pipeline, subclass this class and implement `run`.
    """

    _action_methods: ClassVar[tuple[str, str]] = ("safe_run", "run")

    datapoint: DatasetT

    @make_action_safe
    def run(self, datapoint: DatasetT) -> Self:
        """Run the pipeline.

        .. note::
            It is usually preferred to use `safe_run` on custom pipelines instead of `run`, as `safe_run` can
            catch certain implementation errors of the run method.

        Parameters
        ----------
        datapoint
            An instance of a :class:`tpcp.Dataset` containing only a single datapoint.
            The structure of the data will depend on the dataset.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        raise NotImplementedError()  # pragma: no cover

    def safe_run(self, datapoint: DatasetT) -> Self:
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
            An instance of a :class:`tpcp.Dataset` containing only a single datapoint.
            The structure of the data will depend on the dataset.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        run_method = self.run
        if getattr(run_method, ACTION_METHOD_INDICATOR, False) is True:
            return run_method(datapoint)
        return _check_safe_run(self, run_method, datapoint)


class OptimizablePipeline(Pipeline[DatasetT]):
    """Pipeline with custom ways to optimize and/or train input parameters.

    OptimizablePipelines are expected to implement a concrete way to train internal models or optimize parameters.
    This should not be a reimplementation of GridSearch or similar methods.
    For this :class:`tpcp.pipelines.GridSearch` should be used directly.

    It is important that `self_optimize` only modifies input parameters of the pipeline that are marked as
    `OptimizableParameter`.
    This means, if a parameter is optimized, by `self_optimize` it should be named in the `__init__`, should be
    exportable when calling `pipeline.get_params` and should be annotated using the `OptimizableParameter` type hint on
    class level.
    For the sake of documentation (and potential automatic checks) in the future, it also makes sense to add the
    `HyperParameter` type annotation to all parameters that act as hyper parameters for the optimization performed in
    `self_optimize`.
    To learn more about parameter annotations check this `example <optimize_pipelines>`_ and this `
    guide <optimization>`_ in the docs.

    It is also possible to optimize nested parameters.
    For example, if the input of the pipeline is an algorithm or another pipeline on its own, all parameters of these
    objects can be modified as well.
    In any case, you should make sure that all optimized parameters are still there if you call `.clone()` on the
    optimized pipeline.
    """

    def self_optimize(self, dataset: DatasetT, **kwargs) -> Self:
        """Optimize the input parameters of the pipeline or algorithm using any logic.

        This method can be used to adapt the input parameters (values provided in the init) based on any data driven
        heuristic.

        .. note::
            The optimizations must only modify the input parameters (aka `self.clone` should retain the optimization
            results).
            If you need to return further information, implement `self_optimize_with_info` instead.

        Parameters
        ----------
        dataset
            An instance of a :class:`tpcp.Dataset` containing one or multiple data points that can
            be used for training.
            The structure of the data and the available reference information will depend on the dataset.
        kwargs
            Additional parameters required for the optimization process.

        Returns
        -------
        self
            The class instance with optimized input parameters.

        """
        try:
            # This seems hacky, but is used to avoid infinite recursion
            setattr(type(self), "__optimize_not_implemented__", True)
            out = self.self_optimize_with_info(dataset, **kwargs)[0]
            delattr(type(self), "__optimize_not_implemented__")
        except NotImplementedError as e:
            raise NotImplementedError() from e
        return out

    def self_optimize_with_info(self, dataset: DatasetT, **kwargs) -> tuple[Self, Any]:
        """Optimize the input parameters of the pipeline or algorithm using any logic.

        This is equivalent to `self_optimize`, but allows you to return additional information as a second return value.
        If you implement this method, there is no need to implement `self_optimize` as well.

        Parameters
        ----------
        dataset
            An instance of a :class:`tpcp.Dataset` containing one or multiple data points that can
            be used for training.
            The structure of the data and the available reference information will depend on the dataset.
        kwargs
            Additional parameters required for the optimization process.

        Returns
        -------
        self
            The class instance with optimized input parameters.
        info
            An arbitrary piece of information

        """
        try:
            if getattr(type(self), "__optimize_not_implemented__", False):
                delattr(type(self), "__optimize_not_implemented__")
                raise NotImplementedError()
        except NotImplementedError as e:
            raise NotImplementedError() from e
        return self.self_optimize(dataset, **kwargs), NOTHING
