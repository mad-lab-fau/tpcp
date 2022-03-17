from typing import Any, Callable, Dict, Optional, Sequence, TypeVar, Union

import numpy as np
from optuna import Study, Trial
from optuna.structs import FrozenTrial
from optuna.study.study import ObjectiveFuncType

from tpcp import Dataset, OptimizablePipeline, Parameter, Pipeline, clone
from tpcp._optimize import BaseOptimize
from tpcp.optimize import Optimize

CustomOptunaOptimize_ = TypeVar("CustomOptunaOptimize_", bound="CustomOptunaOptimize")


class CustomOptunaOptimize(BaseOptimize):
    """Base class for custom Optuna wrapper.


    Example
    -------

    >>> from tpcp.validate import Scorer
    >>> from optuna import create_study
    >>> from optuna import samplers
    >>>
    >>> class MyOptunaOptimizer(CustomOptunaOptimize):
    ...     def create_objective(self):
    ...         def objective(trial: Trial, pipeline: Pipeline, dataset: Dataset):
    ...             trial.suggest_float("my_pipeline_para", 0, 3)
    ...             mean_score = Scorer(lambda dp: pipeline.score(dp))
    ...             return mean_score
    ...         return objective
    >>>
    >>> study = create_study(sampler=samplers.RandomSampler())
    >>> opti = MyOptunaOptimizer(pipeline=MyPipeline(), study=study, n_trials=10)
    >>> opti = opti.optimize(MyDataset())

    """
    pipeline: Parameter[Pipeline]
    study: Parameter[Study]

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
        """Detailed results of the study.

        This basically contains the same information as `self.study_.trials_dataframe()`, with some small modifications:

        - columns starting with "params_" are renamed to "param_"
        - a new column called "params" containing all parameters as dict is added
        - "value" is renamed to score"
        - the score of pruned trials is set to `np.nan`

        These changes are made to make the output comparable to the output of :class:`~tpcp.optimize.GridSearch` and
        :class:`~tpcp.optimize.GridSearchCV`.
        """
        def rename_param_columns(name: str):
            param_start = "params_"
            if name.startswith(param_start):
                return "param_" + name[len(param_start) :]
            return name

        base_df = (
            self.study_.trials_dataframe()
            .drop("number", axis=1)
            .rename(columns={"value": "score"})
            .rename(columns=rename_param_columns)
        )
        base_df["params"] = [t.params for t in self.study_.trials]

        # If a trial is pruned we set the score to nan.
        # This is clearer than showing the last step value.
        # If this is required people should report them as user values in their objective function
        base_df.loc[base_df["state"] == "PRUNED", "score"] = np.nan

        return base_df.to_dict()

    @property
    def best_params_(self) -> Dict[str, Any]:
        """Parameters of the best trial in the :class:`~optuna.study.Study`."""
        return self.study_.best_params

    @property
    def best_score_(self) -> float:
        """Best score reached in the study."""
        return self.study_.best_value

    @property
    def best_trial_(self) -> FrozenTrial:
        """Best trial in the :class:`~optuna.study.Study`."""
        return self.study_.best_trial

    def optimize(self: CustomOptunaOptimize_, dataset: Dataset, **_: Any) -> CustomOptunaOptimize_:
        """Optimize the objective over the dataset and find the best parameter combination.

        This method calls `self.create_objective` to obtain the objective function that should be optimized.

        Parameters
        ----------
        dataset
            The dataset used for optimization.

        """
        self.dataset = dataset

        objective = self._create_objective(self.pipeline, dataset=dataset)

        self.study_ = self.study

        # TODO: expose remaining parameters
        self.study_.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        if self.return_optimized:
            self.optimized_pipeline_ = self.return_optimized_pipeline(clone(self.pipeline), dataset, self.study_)
        return self

    def create_objective(self) -> Callable[[Trial, Pipeline, Dataset], Union[float, Sequence[float]]]:
        """Return the objective function that should be optimized.

        This method should be implemented by a child class and return a objective function that is compatible with
        Optuna.
        However, compared to a normal Optuna objective function, the function should expect a pipeline and a dataset
        object as additional inputs to do the optimization.
        """
        raise NotImplementedError()

    def _create_objective(self, pipeline: Pipeline, dataset: Dataset) -> ObjectiveFuncType:
        inner_objective = self.create_objective()

        def objective(trial: Trial):
            inner_pipe = clone(pipeline)
            return inner_objective(trial, inner_pipe, dataset)

        return objective

    def return_optimized_pipeline(self, pipeline: Pipeline, dataset: Dataset, study: Study) -> Pipeline:
        """Return the pipeline with the best parameters of a study.

        This either just returns the pipeline with the best parameters set, or if the pipeline is a subclass of
        `OptimizablePipeline` it attempts a re-optimization of the pipeline using the provided dataset.

        This functionality is a sensible default, but it is expected to overwrite this method in custom subclasses,
        if specific behaviour is needed.

        Don't call this function on its own! It is only expected to be called internally by optimize.
        """
        # Pipeline that will be passed here is already cloned, so no need to clone again.
        pipeline_with_best_params = pipeline.set_params(**study.best_params)
        if isinstance(pipeline_with_best_params, OptimizablePipeline):
            return Optimize(pipeline_with_best_params).optimize(dataset).optimized_pipeline_
        return pipeline_with_best_params
