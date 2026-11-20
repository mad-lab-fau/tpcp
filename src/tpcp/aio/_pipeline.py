"""Base classes for composing task-returning actions."""

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, ParamSpec, Self, TypeVar

from tpcp import NOTHING
from tpcp._algorithm import Algorithm as SyncAlgorithm
from tpcp._algorithm_utils import _check_run_result
from tpcp._async_execution import BoundTask, _LocalWorkflow
from tpcp._config import get_config
from tpcp._dataset import DatasetT
from tpcp._hash import custom_hash

P = ParamSpec("P")
R = TypeVar("R")


class Algorithm(SyncAlgorithm):
    """Declare action names with _action_methods and bind their async implementation with action()."""

    def action(self, method: Callable[P, Awaitable[R]], /, *args: P.args, **kwargs: P.kwargs) -> BoundTask[R]:
        """Bind a coroutine method that applies results to this instance on every execution."""
        return _LocalWorkflow(method)(*args, **kwargs)


class Pipeline(Algorithm, Generic[DatasetT]):
    """Implement run(datapoint) to return a bound task."""

    _action_methods: ClassVar = ("safe_run", "run")

    def run(self, datapoint: DatasetT) -> BoundTask[Self]:
        """Bind execution on one datapoint."""
        raise NotImplementedError

    def safe_run(self, datapoint: DatasetT) -> BoundTask[Self]:
        """Check parameters and action results around the awaited run."""
        return self.action(self._safe_run, datapoint)

    async def _safe_run(self, datapoint):
        if not get_config()["action_safety_checks"]:
            return await self.run(datapoint)
        before = custom_hash(self.get_params(deep=False))
        output = await self.run(datapoint)
        return _check_run_result(self, self.run, before, output)


class OptimizablePipeline(Pipeline[DatasetT]):
    """Implement either optimization action. The other delegates to it."""

    def self_optimize(self, dataset: DatasetT, **kwargs) -> BoundTask[Self]:
        """Bind training and return the fitted pipeline."""
        return self.action(self._without_info, dataset, kwargs)

    async def _without_info(self, dataset, kwargs):
        if type(self).self_optimize_with_info is OptimizablePipeline.self_optimize_with_info:
            raise NotImplementedError
        fitted, _ = await self.self_optimize_with_info(dataset, **kwargs)
        return fitted

    def self_optimize_with_info(self, dataset: DatasetT, **kwargs) -> BoundTask[tuple[Self, Any]]:
        """Bind training and return the fitted pipeline and optimization information."""
        return self.action(self._with_info, dataset, kwargs)

    async def _with_info(self, dataset, kwargs):
        return await self.self_optimize(dataset, **kwargs), NOTHING
