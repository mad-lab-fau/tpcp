"""Task-returning optimizers for local Optuna studies."""

from typing import Any, Self

from tpcp._async_execution import BoundTask, _LocalWorkflow, _optimization_resource, _run_local_call
from tpcp._dataset import DatasetT
from tpcp._pipeline import PipelineT
from tpcp.aio.optimize import BaseOptimize
from tpcp.aio.validate import _require_sync_pipeline, _serial_scorer
from tpcp.optimize.optuna import _CustomOptunaOptimizeConfig, _OptunaSearchConfig, _StudyOptimizer

__all__ = ["CustomOptunaOptimize", "OptunaSearch"]


class _Study(_StudyOptimizer[PipelineT, DatasetT], BaseOptimize[PipelineT, DatasetT]):
    def optimize(self, dataset: DatasetT, *, resource=None, **kwargs: Any) -> BoundTask[Self]:
        """Bind a fresh serial study execution using the configured storage."""
        return _LocalWorkflow(self._execute_study)(dataset, resource or _optimization_resource.get(), kwargs)

    async def _execute_study(self, dataset, resource, kwargs):
        _require_sync_pipeline(self.pipeline)
        if resource != "cpu":
            raise ValueError("Local Optuna studies require CPU execution.")
        if self.n_jobs != 1:
            raise ValueError("Async Optuna studies require n_jobs=1.")
        # Optuna owns live storage, callbacks and the sampler. Each await runs the study.
        return await _run_local_call(self._run_study, dataset, **kwargs)


class CustomOptunaOptimize(_CustomOptunaOptimizeConfig[PipelineT, DatasetT], _Study[PipelineT, DatasetT]):
    """Implement create_objective to run a serial study in a local thread."""


class OptunaSearch(_OptunaSearchConfig[PipelineT, DatasetT], _Study[PipelineT, DatasetT]):
    """Bind an Optuna parameter search with live study storage."""

    def _study_scorer(self):
        return _serial_scorer(self.pipeline, self.scoring)
