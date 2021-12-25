from concurrent.futures import Future, wait, FIRST_COMPLETED
from typing import TypeVar, Any, Callable, Union, Optional, Dict

import numpy as np
from joblib import delayed
from joblib._parallel_backends import ImmediateResult
from optuna import Trial, Study, create_study, samplers
from optuna.trial import TrialState
from tqdm import tqdm

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
        progress_bar: bool = True,
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
        self.progress_bar = progress_bar

    def optimize(self: OptunaSearch_, dataset: Dataset, **optimize_params: Any) -> OptunaSearch_:

        self.dataset = dataset

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

        pbar: Optional[tqdm] = None
        if self.progress_bar:
            pbar = tqdm(total=self.n_trials, desc="Para Combos")

        n_jobs = self.n_jobs or 1
        pool = CustomLokyPool(n_jobs=n_jobs, pbar=pbar)
        results = []
        tested_params = []

        def create_task():
            trial = self.study_.ask()
            params = self.search_space_creator(trial)
            p = clone(self.pipeline).set_params(**params)
            task = delayed(_score)(
                pipeline=p,
                dataset=dataset,
                scorer=scoring,
                parameters=params,
                error_score=self.error_score,
                trial=trial,
                transform_score=transform_score,
                return_parameters=True,
                return_data_labels=True,
                return_times=True,
            )
            return task, trial

        # TODO: Warning here, if there are many workers, but small number of trials? This might not result in what
        #  your expect
        n_initial = min(pool.n_jobs, self.n_trials)

        running_params = {}
        with pool:
            # First we start n trials based on the number of n workers
            for _ in range(n_initial):
                task, trial = create_task()
                running_params[pool.submit(task)] = trial

            # Then we wait until one of the tasks completes and check if we want to start a new one.
            while running_params:
                done, running = wait(running_params, return_when=FIRST_COMPLETED)
                for ft in done:
                    result = ft.get()
                    finished_trail = running_params[ft]
                    state = result["state"]
                    if state == TrialState.COMPLETE:
                        self.study_.tell(finished_trail, transform_score(result["scores"]))
                    else:
                        self.study_.tell(finished_trail, None, state=state)
                    results.append(result)
                    tested_params.append(result["parameters"])
                    del running_params[ft]
                if len(results) + len(running) < self.n_trials:
                    task, trial = create_task()
                    running_params[pool.submit(task)] = trial

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
