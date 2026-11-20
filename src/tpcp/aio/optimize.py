"""Asynchronous optimization entry points."""

from typing import Any, TypeVar

from tpcp._dataset import DatasetT
from tpcp.aio.parallel import BoundTask
from tpcp.optimize import GridSearch

_OptimizerT = TypeVar("_OptimizerT", bound=GridSearch)


def optimize(optimizer: _OptimizerT, dataset: DatasetT, **kwargs: Any) -> BoundTask[_OptimizerT]:
    """Bind a grid search as a lazy task."""
    return optimizer.async_optimize(dataset, **kwargs)


__all__ = ["GridSearch", "optimize"]
