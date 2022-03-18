"""Baseclass for optimizers.

This is in a separate file to avoid circular imports.
"""
from typing import Any, Dict, Generic, Tuple, TypeVar, Union

from typing_extensions import Self

from tpcp import Algorithm, Parameter
from tpcp._dataset import Dataset_
from tpcp._pipeline import Pipeline_

BaseOptimize_ = TypeVar("BaseOptimize_", bound="BaseOptimize")


class BaseOptimize(Algorithm, Generic[Pipeline_, Dataset_], _skip_validation=True):
    """Base class for all optimizer."""

    _action_methods: Union[Tuple[str, ...], str] = "optimize"

    pipeline: Parameter[Pipeline_]

    dataset: Dataset_

    optimized_pipeline_: Pipeline_

    def optimize(self, dataset: Dataset_, **optimize_params: Any) -> Self:
        """Apply some form of optimization on the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self, datapoint: Dataset_) -> Pipeline_:
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.run(datapoint)

    def safe_run(self, datapoint: Dataset_) -> Pipeline_:
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.safe_run(datapoint)

    def score(self, datapoint: Dataset_) -> Union[float, Dict[str, float]]:
        """Run score of the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.score(datapoint)
