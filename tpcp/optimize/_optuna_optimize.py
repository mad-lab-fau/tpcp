from typing import TypeVar, Any, Callable, Union, Optional, Dict

import numpy as np
from optuna import Trial, Study, create_study, samplers
from optuna.trial import TrialState

from tpcp import Dataset, Pipeline, clone
from tpcp._optimize import BaseOptimize
from tpcp._utils._score import _SCORE_CALLABLE, _ERROR_SCORE_TYPE
from tpcp._utils._score_optuna import _score
from tpcp.validate import Scorer
from tpcp.validate._scorer import _validate_scorer

OptunaSearch_ = TypeVar("OptunaSearch_", bound="OptunaSearch")


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
        #  TODO: Replace return optimized with rank function or similar
        self.return_optimized = return_optimized
        self.error_score = error_score

    def optimize(self: OptunaSearch_, dataset: Dataset, **optimize_params: Any) -> OptunaSearch_:
        if self.study is None:
            # TODO: Random seed hadnling
            sampler = samplers.TPESampler()

            self.study_ = create_study(direction="maximize", sampler=sampler)
        else:
            self.study_ = self.study

        scoring = _validate_scorer(self.scoring, self.pipeline)

        if isinstance(self.return_optimized, str):
            transform_score = lambda x: x[self.return_optimized]
        else:
            transform_score = lambda x: x

        # TODO: Check if we can use proper mutliprocessing here
        for _ in range(self.n_trials):
            trial = self.study_.ask()
            params = self.search_space_creator(trial)
            p = clone(self.pipeline).set_params(**params)
            result = _score(
                pipeline=p,
                dataset=self.dataset,
                scorer=scoring,
                parameters=params,
                error_score=self.error_score,
                trial=trial,
                transform_score=transform_score,
                return_parameters=True,
                return_data_labels=True,
                return_times=True
            )
            state = result["state"]
            if state == TrialState.COMPLETE:
                self.study_.tell(trial, transform_score(result["scores"]))
            else:
                self.study_.tell(trial, None, state=state)

        self.best_params_ = self.study_.best_params
        if self.return_optimized:
            self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        return self


class OptunaSearchCV(BaseOptimize):
    pass
