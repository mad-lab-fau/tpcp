from typing import Any, Callable, Dict, Optional, Sequence, TypeVar, Union

from optuna import Study, Trial, create_study, samplers
from optuna.structs import FrozenTrial
from optuna.study.study import ObjectiveFuncType

from tpcp import Dataset, OptimizablePipeline, Parameter, Pipeline, clone
from tpcp._optimize import BaseOptimize
from tpcp.optimize import Optimize

CustomOptunaOptimize_ = TypeVar("CustomOptunaOptimize_", bound="CustomOptunaOptimize")


class CustomOptunaOptimize(BaseOptimize):
    pipeline: Parameter[Pipeline]
    study: Study

    optimized_pipeline_: Pipeline
    study_: Study

    def __init__(
        self,
        pipeline: Pipeline,
        study: Study,
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        return_optimized: bool = True,
    ) -> None:  # noqa: super-init-not-called
        self.pipeline = pipeline
        self.timeout = timeout
        self.n_trials = n_trials
        self.study = study
        self.return_optimized = return_optimized

    @property
    def search_results_(self) -> Dict[str, Sequence[Any]]:
        return self.study_.trials_dataframe().to_dict()

    @property
    def best_params_(self) -> Dict[str, Any]:
        """Parameters of the best trial in the :class:`~optuna.study.Study`."""
        return self.study_.best_params

    @property
    def best_score_(self) -> float:
        """Mean cross-validated score of the best estimator."""
        return self.study_.best_value

    @property
    def best_trial_(self) -> FrozenTrial:
        """Best trial in the :class:`~optuna.study.Study`."""
        return self.study_.best_trial

    def optimize(self: CustomOptunaOptimize_, dataset: Dataset, **_: Any) -> CustomOptunaOptimize_:
        self.dataset = dataset

        objective = self._create_objective(self.pipeline, dataset=dataset)

        self.study_ = self.study

        # TODO: expose remaining parameters
        self.study_.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)

        if self.return_optimized:
            self.optimized_pipeline_ = self.refit(clone(self.pipeline), dataset, self.study_)
        return self

    def create_objective(self) -> Callable[[Trial, Pipeline, Dataset], Union[float, Sequence[float]]]:
        raise NotImplementedError()

    def _create_objective(self, pipeline: Pipeline, dataset: Dataset) -> ObjectiveFuncType:
        inner_objective = self.create_objective()

        def objective(trial: Trial):
            inner_pipe = clone(pipeline)
            return inner_objective(trial, inner_pipe, dataset)

        return objective

    def refit(self, pipeline: Pipeline, dataset: Dataset, study: Study) -> Pipeline:
        # Pipeline that will be passed here is already cloned, so no need to clone again.
        pipeline_with_best_params = pipeline.set_params(**study.best_params)
        if isinstance(pipeline_with_best_params, OptimizablePipeline):
            return Optimize(pipeline_with_best_params).optimize(dataset).optimized_pipeline_
        return pipeline_with_best_params
