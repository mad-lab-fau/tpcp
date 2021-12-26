"""Baseclass for optimizers.

This is in a separate file to avoid circular imports.
"""
from typing import Any, Dict, Tuple, TypeVar, Union

from tpcp import Algorithm, Dataset, Parameter, Pipeline

BaseOptimize_ = TypeVar("BaseOptimize_", bound="BaseOptimize")


class BaseOptimize(Algorithm, _skip_validation=True):
    """Base class for all optimizer."""

    _action_methods: Union[Tuple[str, ...], str] = "optimize"

    pipeline: Parameter[Pipeline]

    dataset: Dataset

    optimized_pipeline_: Pipeline

    def optimize(self: BaseOptimize_, dataset: Dataset, **optimize_params: Any) -> BaseOptimize_:
        """Apply some form of optimization on the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self, datapoint: Dataset) -> Pipeline:
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.run(datapoint)

    def safe_run(self, datapoint: Dataset) -> Pipeline:
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.safe_run(datapoint)

    def score(self, datapoint: Dataset) -> Union[float, Dict[str, float]]:
        """Run score of the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.score(datapoint)
