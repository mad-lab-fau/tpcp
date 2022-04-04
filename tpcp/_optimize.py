"""Baseclass for optimizers.

This is in a separate file to avoid circular imports.
"""
from typing import Any, Dict, Generic, Tuple, Union

from typing_extensions import Self

from tpcp import Algorithm, Parameter
from tpcp._dataset import DatasetT
from tpcp._pipeline import PipelineT


class BaseOptimize(Algorithm, Generic[PipelineT, DatasetT], _skip_validation=True):
    """Base class for all optimizer."""

    _action_methods: Union[Tuple[str, ...], str] = "optimize"

    pipeline: Parameter[PipelineT]

    dataset: DatasetT

    optimized_pipeline_: PipelineT

    def optimize(self, dataset: DatasetT, **optimize_params: Any) -> Self:
        """Apply some form of optimization on the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self, datapoint: DatasetT) -> PipelineT:
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.run(datapoint)

    def safe_run(self, datapoint: DatasetT) -> PipelineT:
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.safe_run(datapoint)

    def score(self, datapoint: DatasetT) -> Union[float, Dict[str, float]]:
        """Run score of the optimized pipeline.

        This is a wrapper to contain API compatibility with `Pipeline`.
        """
        return self.optimized_pipeline_.score(datapoint)
