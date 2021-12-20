from concurrent.futures import Future
from typing import TypeVar, Any, Callable, Union, Optional, Dict

import numpy as np
from joblib import delayed
from optuna import Trial, Study, create_study, samplers
from optuna.trial import TrialState

from tpcp import Dataset, Pipeline, clone
from tpcp._optimize import BaseOptimize
from tpcp._utils._multiprocess import CustomLokyPool
from tpcp._utils._score import _SCORE_CALLABLE, _ERROR_SCORE_TYPE
from tpcp._utils._score_optuna import _score
from tpcp.optimize._optimize import _format_gs_results
from tpcp.validate import Scorer
from tpcp.validate._scorer import _validate_scorer

OptunaSearch_ = TypeVar("OptunaSearch_", bound="OptunaSearch")


class OptunaSearch(BaseOptimize):
    gs_results_: Dict[str, Any]
    best_params_: Dict[str, Any]
    multimetric_: bool

    def __init__(
        self,
        pipeline: Pipeline,
        search_space_creator: Callable[[Trial], Dict[str, Any]],
        *,
        scoring: Optional[Union[_SCORE_CALLABLE, Scorer]] = None,
        study: Optional[Study] = None,
        n_trials: int = 10,
        n_jobs: Optional[int] = None,
        return_optimized: Union[bool, str] = True,
        error_score: _ERROR_SCORE_TYPE = np.nan,
    ) -> None:
        self.pipeline = pipeline
        self.search_space_creator = search_space_creator
        self.scoring = scoring
        self.study = study
        self.n_trials = n_trials
        self.n_jobs = n_jobs
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

        pool = CustomLokyPool(n_jobs=self.n_jobs)
        results = []
        tested_params = []
        with pool:
            for _ in range(self.n_trials):
                trial = self.study_.ask()
                params = self.search_space_creator(trial)
                p = clone(self.pipeline).set_params(**params)
                task = delayed(_score)(
                    pipeline=p,
                    dataset=self.dataset,
                    scorer=scoring,
                    parameters=params,
                    error_score=self.error_score,
                    trial=trial,
                    transform_score=transform_score,
                    return_parameters=True,
                    return_data_labels=True,
                    return_times=True,
                )

                def callback(result: Future):
                    result = result.result()
                    state = result["state"]
                    if state == TrialState.COMPLETE:
                        self.study_.tell(trial, transform_score(result["scores"]))
                    else:
                        self.study_.tell(trial, None, state=state)
                    results.append(result)
                    tested_params.append(result["parameters"])

                pool.submit(task, callback)

        first_test_score = results[0]["scores"]
        self.multimetric_ = isinstance(first_test_score, dict)

        # We reuse the parameter formatting from GridSearch
        results = _format_gs_results(
            tested_params,
            results,
        )

        # TODO: Should we check that the best params reported by optuna are the same as our?
        if self.return_optimized:
            self.best_params_ = self.study_.best_params
            self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        self.gs_results_ = results

        return self


class OptunaSearchCV(BaseOptimize):
    pass
