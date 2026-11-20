"""Baseclass for optimizers.

This is in a separate file to avoid circular imports.
"""

from typing import Any, ClassVar, Generic, Self

from tpcp import Algorithm, Parameter
from tpcp._dataset import DatasetT
from tpcp._pipeline import PipelineT


class _Optimizer(Algorithm, Generic[PipelineT, DatasetT]):
    """Shared configuration and fitted state for optimizer classes."""

    _action_methods: ClassVar[tuple[str, ...] | str] = "optimize"

    pipeline: Parameter[PipelineT]

    dataset: DatasetT

    optimized_pipeline_: PipelineT


class BaseOptimize(_Optimizer[PipelineT, DatasetT]):
    """Base class for synchronous optimizers."""

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

    def optimize(self, dataset: DatasetT, **optimize_params: Any) -> Self:
        """Apply some form of optimization on the input parameters of the pipeline."""
        raise NotImplementedError()
