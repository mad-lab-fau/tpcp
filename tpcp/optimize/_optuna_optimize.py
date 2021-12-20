from typing import TypeVar, Any, Callable, Union, Optional, Dict

import numpy as np
from optuna import Trial, Study, create_study, samplers

from tpcp import Dataset, Pipeline, clone
from tpcp._optimize import BaseOptimize
from tpcp._utils._score import _SCORE_CALLABLE, _ERROR_SCORE_TYPE
from tpcp.validate import Scorer

OptunaSearch_ = TypeVar("OptunaSearch_", bound="OptunaSearch")


class _SimpleObjective:
    def __init__(
        self,
        pipeline: Pipeline,
        search_space_creator: Callable[[Trial], Dict[str, Any]],
        scoring: Scorer,
        error_score: _ERROR_SCORE_TYPE,
        score_name: Optional[str],
        dataset: Dataset,
    ):
        self.pipeline = pipeline
        self.search_space_creator = search_space_creator
        self.scoring = scoring
        self.error_score = error_score
        self.score_name = score_name
        self.dataset = dataset

    def __call__(self, trial: Trial):
        pipeline = clone(self.pipeline)
        params = self.search_space_creator(trial)

        pipeline = pipeline.set_params(**params)
        # TODO: Replace with `_score` and then use `trail.set_user_attr` to store all addition information.
        agg_scores, _ = self.scoring(pipeline, data=self.dataset, error_score=self.error_score)
        if (self.score_name is False and isinstance(agg_scores, dict)) or (
            isinstance(self.score_name, str) and not isinstance(agg_scores, dict)
        ):
            raise ValueError()
        if self.score_name:
            return agg_scores[self.score_name]
        else:
            return agg_scores


class OptunaSearch(BaseOptimize):
    def __init__(
        self,
        pipeline: Pipeline,
        search_space_creator: Callable[[Trial], Dict[str, Any]],
        *,
        scoring: Optional[Union[_SCORE_CALLABLE, Scorer]] = None,
        study: Optional[Study] = None,
        n_trials: int = 10,
        return_optimized: Union[bool, str] = True,
        error_score: _ERROR_SCORE_TYPE = np.nan,
    ) -> None:
        self.pipeline = pipeline
        self.search_space_creator = search_space_creator
        self.scoring = scoring
        self.study = study
        self.n_trials = n_trials
        self.return_optimized = return_optimized
        self.error_score = error_score

    def optimize(self: OptunaSearch_, dataset: Dataset, **optimize_params: Any) -> OptunaSearch_:
        if self.study is None:
            # TODO: Random seed hadnling
            sampler = samplers.TPESampler()

            self.study_ = create_study(direction="maximize", sampler=sampler)
        else:
            self.study_ = self.study

        objective = _SimpleObjective(
            self.pipeline, self.search_space_creator, self.scoring, self.error_score, self.return_optimized, dataset
        )

        # TODO: Maybe switch to ask and tell interface here. This should allow for easier multiprocessing with a
        #  custom loop
        # TODO: Check if we can use proper mutliprocessing here
        self.study_.optimize(objective, n_jobs=1, n_trials=self.n_trials)

        self.best_params_ = self.study_.best_params
        if self.return_optimized:
            self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        return self



class OptunaSearchCV(BaseOptimize):
    pass
