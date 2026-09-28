r"""
.. _pure_parameters:

Pure parameters
===============

Some pipeline parameters change the output of
:meth:`~tpcp.OptimizablePipeline.run` without changing what
:meth:`~tpcp.OptimizablePipeline.self_optimize` learns. Marking these as
:class:`~tpcp.PureParameter` lets :class:`~tpcp.optimize.GridSearchCV`
train once per fold and training-dependent parameter combination, then score
each value of the pure parameter with the trained pipeline.

Here we train the QRS detector from :ref:`gridsearch_cv` and apply a separate
minimum-distance rule to its detected peaks. The filter cutoff affects the
learned detector. The minimum distance only removes peaks after detection, so
it cannot affect training.
"""

from pathlib import Path
from typing import Self

import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from tpcp import (
    OptimizableParameter,
    OptimizablePipeline,
    Parameter,
    PureParameter,
    cf,
)
from tpcp.optimize import GridSearchCV

from examples.algorithms.algorithms_qrs_detection_final import (
    OptimizableQrsDetector,
    match_events_with_reference,
    precision_recall_f1_score,
)
from examples.datasets.datasets_final_ecg import ECGExampleData

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
dataset = ECGExampleData(data_path)[:4]

# %%
# A pipeline with an action-only parameter
# ----------------------------------------
# Use four recordings to keep this example quick. Each CV fold trains on two.
# `min_peak_distance_s` is a top-level pipeline parameter. It is used only in
# `run`, after the detector has found its candidate peaks. Neither
# `self_optimize` nor anything it calls reads this parameter.
#
# We record each real training call to make the benefit of caching visible.
# The example runs with one job, so the list is shared with the pipeline
# instances that GridSearchCV clones.
training_calls = []


class QrsPipeline(OptimizablePipeline[ECGExampleData]):
    algorithm: Parameter[OptimizableQrsDetector]
    algorithm__min_r_peak_height_over_baseline: OptimizableParameter[float]
    min_peak_distance_s: PureParameter[float]

    r_peak_positions_: pd.Series

    def __init__(
        self,
        algorithm: OptimizableQrsDetector = cf(OptimizableQrsDetector()),
        min_peak_distance_s: float = 0.3,
    ):
        self.algorithm = algorithm
        self.min_peak_distance_s = min_peak_distance_s

    def self_optimize(self, dataset: ECGExampleData) -> Self:
        training_calls.append(self.algorithm.high_pass_filter_cutoff_hz)
        ecg_data = [d.data["ecg"] for d in dataset]
        r_peaks = [d.r_peak_positions_["r_peak_position"] for d in dataset]
        self.algorithm = self.algorithm.clone().self_optimize(
            ecg_data, r_peaks, dataset.sampling_rate_hz
        )
        return self

    def run(self, datapoint: ECGExampleData) -> Self:
        detected = self.algorithm.clone().detect(
            datapoint.data["ecg"], datapoint.sampling_rate_hz
        )
        min_distance_samples = (
            self.min_peak_distance_s * datapoint.sampling_rate_hz
        )
        kept_peaks = []
        for peak in detected.r_peak_positions_:
            if not kept_peaks or peak - kept_peaks[-1] >= min_distance_samples:
                kept_peaks.append(peak)
        self.r_peak_positions_ = pd.Series(kept_peaks)
        return self


# %%
# Score every parameter combination
# ---------------------------------
# A larger minimum distance can suppress an incorrect extra detection, but
# it may also remove a real heartbeat.
# We score the filtered peaks against the labeled reference peaks.
def score(pipeline: QrsPipeline, datapoint: ECGExampleData) -> float:
    detected = pipeline.safe_run(datapoint).r_peak_positions_.to_numpy()
    reference = datapoint.r_peak_positions_["r_peak_position"].to_numpy()
    matches = match_events_with_reference(
        detected, reference, tolerance=0.02 * datapoint.sampling_rate_hz
    )
    return precision_recall_f1_score(matches)[2]


filter_cutoffs = [0.25, 0.5, 1.0]
peak_distances = [0.3, 0.5, 0.8, 1.1]
parameters = ParameterGrid(
    {
        "algorithm__high_pass_filter_cutoff_hz": filter_cutoffs,
        "min_peak_distance_s": peak_distances,
    }
)

search = GridSearchCV(
    pipeline=QrsPipeline(),
    parameter_grid=parameters,
    scoring=score,
    cv=KFold(n_splits=2),
    pure_parameters=True,
    return_optimized=False,
    progress_bar=False,
).optimize(dataset)

# The pure parameter still changes the output and therefore the score.
# GridSearchCV evaluates every combination.
results = pd.DataFrame(search.cv_results_)
results[
    [
        "param__algorithm__high_pass_filter_cutoff_hz",
        "param__min_peak_distance_s",
        "mean__test__agg__score",
    ]
]

# %%
# Count the training calls
# ------------------------
# There are 3 filter cutoffs, 4 peak distances, and 2 folds: 24 scores.
# Only the filter cutoff affects training, so the detector is trained
# 3 * 2 = 6 times. Without `pure_parameters=True`, it would train 24 times.
# `return_optimized=False` avoids an additional final training call.
print(f"Parameter combinations: {len(parameters)}")
print(f"Training calls: {len(training_calls)}")
assert len(training_calls) == len(filter_cutoffs) * 2

# Use `PureParameter` only when changing the parameter cannot change the
# result of `self_optimize`, including calls made from within it. Marking a
# training-dependent parameter as pure would reuse the wrong trained result.
