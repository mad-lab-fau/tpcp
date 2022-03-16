r"""
.. _custom_optuna_optimizer:

Custom Optuna
============
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
from optuna import Study, Trial, samplers
from optuna.pruners import BasePruner
from optuna.study.study import ObjectiveFuncType, create_study
from optuna.trial import FrozenTrial

from examples.datasets.datasets_final_ecg import ECGExampleData
from tpcp.optimize._optuna_optimize import CustomOptunaOptimize

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path(".").resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)


import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector
from tpcp import Dataset, Parameter, Pipeline, cf


class MyPipeline(Pipeline):
    algorithm: Parameter[QRSDetector]

    r_peak_positions_: pd.Series

    def __init__(self, algorithm: QRSDetector = cf(QRSDetector())):
        self.algorithm = algorithm

    def run(self, datapoint: ECGExampleData):
        # Note: We need to clone the algorithm instance, to make sure we don't leak any data between runs.
        algo = self.algorithm.clone()
        algo.detect(datapoint.data, datapoint.sampling_rate_hz)

        self.r_peak_positions_ = algo.r_peak_positions_
        return self


pipe = MyPipeline()


from examples.algorithms.algorithms_qrs_detection_final import match_events_with_reference


def f1_score(pipeline: MyPipeline, datapoint: ECGExampleData) -> float:
    # We use the `safe_run` wrapper instead of just run. This is always a good idea.
    # We don't need to clone the pipeline here, as GridSearch will already clone the pipeline internally and `run`
    # will clone it again.
    pipeline = pipeline.safe_run(datapoint)
    tolerance_s = 0.02  # We just use 20 ms for this example
    matches_events, _ = match_events_with_reference(
        pipeline.r_peak_positions_.to_numpy(),
        datapoint.r_peak_positions_.to_numpy(),
        tolerance=tolerance_s * datapoint.sampling_rate_hz,
    )
    n_tp = len(matches_events)
    f1_score = (2 * n_tp) / (len(pipeline.r_peak_positions_) + len(datapoint.r_peak_positions_))
    return f1_score


class MinDatapointPerformancePruner(BasePruner):
    def __init__(self, min_performance: float):
        self.min_performance = min_performance

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        step = trial.last_step

        if step is not None:
            score = trial.intermediate_values[step]
            if score < self.min_performance:
                return True
        return False


class OptunaSearchEarlyStopping(CustomOptunaOptimize):
    def __init__(
        self,
        pipeline: Pipeline,
        study: Optional[Study],
        create_search_space: Callable[[Trial], Dict[str, Any]],
        score_function: Callable[[Pipeline, Dataset], float],
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        return_optimized: bool = True,
    ) -> None:
        self.create_search_space = create_search_space
        self.score_function = score_function
        super().__init__(pipeline, study, n_trials=n_trials, timeout=timeout, return_optimized=return_optimized)

    def create_objective(self) -> Callable[[Trial, Pipeline, Dataset], Union[float, Sequence[float]]]:
        def objective(trial: Trial, pipeline: Pipeline, dataset: Dataset) -> float:
            paras = self.create_search_space(trial)
            pipeline = pipeline.set_params(**paras)

            single_scores: List[float] = []
            for step, dp in enumerate(dataset):
                score = self.score_function(pipeline, dp)
                # We will append the value to all scores independent of early stopping or not. This allows us to see
                # the value in the results.
                single_scores.append(score)

                trial.report(score, step)

                if trial.should_prune():
                    # We report the single_scores before ending the trial
                    trial.set_user_attr("single_scores", single_scores)
                    raise optuna.TrialPruned(dp.groups[0], score)

            trial.set_user_attr("single_scores", single_scores)

            return float(np.mean(single_scores))

        return objective


def create_search_space(trial: Trial):
    parameters = dict(
        algorithm__min_r_peak_height_over_baseline=trial.suggest_float(
            "algorithm__min_r_peak_height_over_baseline", 0.1, 2, step=0.1
        ),
        algorithm__high_pass_filter_cutoff_hz=trial.suggest_float(
            "algorithm__high_pass_filter_cutoff_hz", 0.1, 2, step=0.1
        ),
    )
    return parameters

sampler = samplers.TPESampler(seed=42)
study = create_study(direction="maximize", sampler=sampler, pruner=MinDatapointPerformancePruner(0.3))

opti = OptunaSearchEarlyStopping(
    pipe,
    study,
    create_search_space=create_search_space,
    score_function=f1_score,
    n_trials=10,
)

opti.optimize(example_data)

print(pd.DataFrame(opti.search_results_))
