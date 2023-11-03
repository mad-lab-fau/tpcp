r"""
.. _opti_info:

Optimization Info
=================

Tpcp is focused on "running" pipelines and less on the "optimization" step.
This is great for traditional algorithms and algorithms will complex return values, as you can easily store multiple
parameters as attributes on the object during `run`.

However, for optimization, you are limited to modifying input parameters.
This works well in many cases, but sometimes, you need additional information from the optimization.
For example, you might want to extract the loss decay of an iterative learning algorithms.
This information is something that you wouldn't want to store in the input parameters (usually).

For these cases tpcp provides the `self_optimize_with_info` method.
This is basically identical to `self_optimize`, but is expected to provide two return values: the optimized instance
AND an arbitrary additional object containing any information you like.
Methods that get optimizable pipelines as input (e.g. :class:`~tpcp.optimize.Optimize` are aware of these method and
will call `self_optimize_with_info` if available and store the additional info as result objects.

The :class:`~tpcp.OptimizablePipeline` base-class is implemented in a way that you only need to worry about
implementing either the `self_optimize_with_info` or the `self_optimize` method.
The other will be available automatically (the additional info will be `NOTHING`, if the method is not implemented).

If you are implementing a new Algorithm (instead of a pipeline), we don't provide this additional support,
but it is relatively simple to implement yourself.

In the following we will show how all of this works by expanding the QRS detection algorithm implemented in the other
examples to return additional information from the optimization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from typing_extensions import Self

from examples.algorithms.algorithms_qrs_detection_final import (
    QRSDetector,
    match_events_with_reference,
)
from examples.datasets.datasets_final_ecg import ECGExampleData
from tpcp import HyperParameter, OptimizableParameter, OptimizablePipeline, Parameter, cf, make_optimize_safe
from tpcp.optimize import Optimize

# %%
# In the algorithm class below, we basically reimplemented the `OptimizableQrsDetector` from the algorithm example.
# However, instead of the `self_optimize` method, we implemented the `self_optimize_with_info` method and added
# additional information from the threshold selection process to the output of the optimization.
#
# To ensure interface compatibility with other algorithms, we also provided a `self_optimize` method, that simply calls
# `self_optimize_with_info` under the hood.


class OptimizableQrsDetectorWithInfo(QRSDetector):
    min_r_peak_height_over_baseline: OptimizableParameter[float]
    r_peak_match_tolerance_s: HyperParameter[float]

    def __init__(
        self,
        max_heart_rate_bpm: float = 200.0,
        min_r_peak_height_over_baseline: float = 1.0,
        r_peak_match_tolerance_s: float = 0.01,
        high_pass_filter_cutoff_hz: float = 1,
    ):
        self.r_peak_match_tolerance_s = r_peak_match_tolerance_s
        super().__init__(
            max_heart_rate_bpm=max_heart_rate_bpm,
            min_r_peak_height_over_baseline=min_r_peak_height_over_baseline,
            high_pass_filter_cutoff_hz=high_pass_filter_cutoff_hz,
        )

    @make_optimize_safe
    def self_optimize_with_info(
        self, ecg_data: list[pd.Series], r_peaks: list[pd.Series], sampling_rate_hz: float
    ) -> tuple[Self, dict[str, np.ndarray]]:
        all_labels = []
        all_peak_heights = []
        for d, p in zip(ecg_data, r_peaks):
            filtered = self._filter(d.to_numpy().flatten(), sampling_rate_hz)
            # Find all potential peaks without the height threshold
            potential_peaks = self._search_strategy(filtered, sampling_rate_hz, use_height=False)
            # Determine the label for each peak, by matching them with our ground truth
            labels = np.zeros(potential_peaks.shape)
            matches = match_events_with_reference(
                events=np.atleast_2d(potential_peaks).T,
                reference=np.atleast_2d(p.to_numpy().astype(int)).T,
                tolerance=self.r_peak_match_tolerance_s * sampling_rate_hz,
            )
            tp_matches = matches[(~np.isnan(matches)).all(axis=1), 0].astype(int)
            labels[tp_matches] = 1
            labels = labels.astype(bool)
            all_labels.append(labels)
            all_peak_heights.append(filtered[potential_peaks])
        all_labels = np.hstack(all_labels)
        all_peak_heights = np.hstack(all_peak_heights)
        # We "brute-force" a good cutoff by testing a bunch of thresholds and then calculating the Youden Index for
        # each.
        fpr, tpr, thresholds = roc_curve(all_labels, all_peak_heights)
        youden_index = tpr - fpr
        # The best Youden index gives us a balance between sensitivity and specificity.
        self.min_r_peak_height_over_baseline = thresholds[np.argmax(youden_index)]

        # Here we create the additional infor object:
        additional_info = {"all_youden_index": youden_index, "all_thresholds": thresholds}
        return self, additional_info

    def self_optimize(self, ecg_data: list[pd.Series], r_peaks: list[pd.Series], sampling_rate_hz: float) -> Self:
        return self.self_optimize_with_info(ecg_data=ecg_data, r_peaks=r_peaks, sampling_rate_hz=sampling_rate_hz)[0]


# %%
# To use this algorithm in an optimization, we need a pipeline to wrap it.
# Below we can find a reimplementation of the pipline from the "Optimizable Pipeline" example.
#
# However, instead of implementing `self_optimize` method, we implemented the `self_optimize_with_info` method and
# also called the `self_optimize_with_info` of our algorithm under the hood.
#
# Note, that for pipelines, we don't need to implement a dummy `self_optimize` method.
# Our baseclass already takes care of that.


class MyPipeline(OptimizablePipeline[ECGExampleData]):
    algorithm: Parameter[OptimizableQrsDetectorWithInfo]
    algorithm__min_r_peak_height_over_baseline: OptimizableParameter[float]

    r_peak_positions_: pd.Series

    def __init__(self, algorithm: OptimizableQrsDetectorWithInfo = cf(OptimizableQrsDetectorWithInfo())):
        self.algorithm = algorithm

    @make_optimize_safe
    def self_optimize_with_info(self, dataset: ECGExampleData, **kwargs):
        ecg_data = [d.data["ecg"] for d in dataset]
        r_peaks = [d.r_peak_positions_["r_peak_position"] for d in dataset]
        # Note: We need to clone the algorithm instance, to make sure we don't leak any data between runs.
        algo = self.algorithm.clone()
        # Here we call the `self_optimize_with_info` method!
        self.algorithm, additional_data = algo.self_optimize_with_info(ecg_data, r_peaks, dataset.sampling_rate_hz)
        return self, additional_data

    def run(self, datapoint: ECGExampleData):
        # Note: We need to clone the algorithm instance, to make sure we don't leak any data between runs.
        algo = self.algorithm.clone()
        algo.detect(datapoint.data["ecg"], datapoint.sampling_rate_hz)

        self.r_peak_positions_ = algo.r_peak_positions_
        return self


# %%
# Let's test this class!
#
# However, first we need some test data
from pathlib import Path

from sklearn.model_selection import train_test_split

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)

train_set, test_set = train_test_split(example_data, train_size=0.7, random_state=0)
# We only want a single dataset in the test set
test_set = test_set[0]

# %%
# With the train data we can try out the optimization
optimized_pipe, info = MyPipeline().self_optimize_with_info(train_set)
info

# %%
optimized_pipe

# %%
# But we can also just call the auto-generated `self_optimize` method and don't get the info:
optimized_pipe = MyPipeline().self_optimize(train_set)
optimized_pipe

# %%
# However, in most cases, we should just use the `Optimize` wrapper.
# It will call the `self_optimize_with_info` method if available (you can force it to use `self_optimize` using the
# `optimize_with_info` parameter) and then provide the additional info as attribute
optimizer = Optimize(MyPipeline()).optimize(train_set)

optimizer.optimization_info_
# %%
optimizer.optimized_pipeline_


# %%
# As `Optimize` is aware of this and stores the info as a result attribute, the information is also available in the
# output of a cross validation.

# %%
# Further Notes
# -------------
# Sometimes it might be a good idea to provide separate implementation of `self_optimize` and `self_optimize_with_info`.
# This might be required, when collecting and calculating the additional info creates a relevant computational overhead.
# However, you should make sure, that the two methods return the same optimization result otherwise.
