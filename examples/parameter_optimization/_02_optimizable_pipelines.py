r"""
.. _optimize_pipelines:

Optimizable Pipelines
=====================

Some algorithms can actively be "trained" to improve their performance or adapt it to a certain dataset.
In `tpcp` we use the term "optimize" instead of "train", as not all algorithms are based on "machine learning" in the
traditional sense.
We consider all algorithms/pipelines "optimizable" if they have parameters and models that can be adapted and optimized
using an algorithm specific optimization method.
Algorithms that can **only** be optimized by brute force (e.g. via GridSearch) are explicitly excluded from this group.
For more information about the conceptional idea behind this, see the guide on
:ref:`algorithm evaluation <algorithm_evaluation>`.

In this example we will implement an optimizable pipeline around the `OptimizableQrsDetector` we developed in
:ref:`custom_algorithms_qrs_detection`.
As optimization might depend on the dataset and pre-processing, we need to write a wrapper around the `self_optimize`
method of the `OptimizableQrsDetector` on a pipeline level.
However, in general this should be really straight forward, as most of the complexity is already implemented on
algorithm level.

This example shows how such a pipeline should be implemented and how it can be optimized using
:class:`~tpcp.optimize.Optimize`.
"""

# %%
# The Pipeline
# ------------
# Our pipeline will implement all the logic on how our algorithms are applied to the data and how algorithms should
# be optimized based on train data.
#
# An optimizable pipeline usually needs the following things:
#
# 1. It needs to be a subclass of :class:`~tpcp.OptimizablePipeline`.
# 2. It needs to have a `run` method that runs all the algorithmic steps and stores the results as class attributes.
#    The `run` method should expect only a single data point (in our case a single recording of one sensor) as input.
# 3. It needs to have an `self_optimize` method, that performs a data-driven optimization of one or more input
#    parameters.
#    This method is expected to return `self` and is only allowed to modify parameters marked as `OptimizableParameter`
#    using the class-level typehints (more below)
# 4. A `init` that defines all parameters that should be adjustable. Note, that the names in the function signature of
#    the `init` method, **must** match the corresponding attribute names (e.g. `max_cost` -> `self.max_cost`).
#    If you want to adjust multiple parameters that all belong to the same algorithm (and your algorithm is
#    implemented as a subclass of :class:`~tpcp.Algorithm`, it can be convenient to just pass the algorithm as a
#    parameter.
#    However, keep potential issues with mutable defaults in mind (:ref:`more info <mutable_defaults>`).
# 5. At least one of the input parameters must be marked as `OptimizableParameter` in the class-level typehints.
#    If parameters are nested tpcp objects you can use the `__` syntax to mark nested values as optimizable.
#    Note, that you always need to mark the parameters you want to optimize in the current pipeline.
#    Annotations in nested objects are ignored.
#    The more precise you are with these annotations, the more help the runtime checks in tpcp can provide.
# 6. (Optionally) Mark parameters as `PureParameter` using the type annotations. This can be used by
#    :class:`~tpcp.optimize.GridSearchCV` to apply some performance optimizations.
#    However, be careful with that!
#    In our case, there are no `PureParameters`, as all (nested) input parameters change the output of the
#    `self_optimize` method.
#

import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import OptimizableQrsDetector
from examples.datasets.datasets_final_ecg import ECGExampleData
from tpcp import OptimizableParameter, OptimizablePipeline, Parameter, cf, make_optimize_safe


class MyPipeline(OptimizablePipeline[ECGExampleData]):
    algorithm: Parameter[OptimizableQrsDetector]
    algorithm__min_r_peak_height_over_baseline: OptimizableParameter[float]

    r_peak_positions_: pd.Series

    def __init__(self, algorithm: OptimizableQrsDetector = cf(OptimizableQrsDetector())):
        self.algorithm = algorithm

    @make_optimize_safe
    def self_optimize(self, dataset: ECGExampleData, **kwargs):
        ecg_data = [d.data["ecg"] for d in dataset]
        r_peaks = [d.r_peak_positions_["r_peak_position"] for d in dataset]
        # Note: We need to clone the algorithm instance, to make sure we don't leak any data between runs.
        algo = self.algorithm.clone()
        self.algorithm = algo.self_optimize(ecg_data, r_peaks, dataset.sampling_rate_hz)
        return self

    def run(self, datapoint: ECGExampleData):
        # Note: We need to clone the algorithm instance, to make sure we don't leak any data between runs.
        algo = self.algorithm.clone()
        algo.detect(datapoint.data["ecg"], datapoint.sampling_rate_hz)

        self.r_peak_positions_ = algo.r_peak_positions_
        return self


pipe = MyPipeline()


# %%
# Comparison
# ----------
# To see the effect of the optimization, we will compare the output of the optimized pipeline with the output of the
# default pipeline.
# As it is not the goal of this example to perform any form of actual evaluation of a model, we will just compare the
# number of identified R-peaks to show, that the optimization had an impact on the output.
#
# For a fair comparison, we must use some train data to optimize the pipeline and then compare the outputs only on a
# separate test set.
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
(train_set.group_labels, test_set.group_labels)

# %%
# The Baseline
# ------------
# For our baseline, we will use the pipeline, but will not apply the optimization.
# This means, the pipeline will use the default threshold.
pipeline = MyPipeline()

# We use the `safe_run` wrapper instead of just run. This is always a good idea.
results = pipeline.safe_run(test_set)
print("The default `min_r_peak_height_over_baseline` is", pipeline.algorithm.min_r_peak_height_over_baseline)
print("Number of R-Peaks:", len(results.r_peak_positions_))


# %%
# Optimization
# ------------
# To optimize the pipeline, we will **not** call `self_optimize` directly, but use the
# :class:`~tpcp.optimize.Optimize` wrapper.
# It has the same interface as other optimization methods like :class:`~tpcp.optimize.GridSearch`.
# Further, it makes some checks to catch potential implementation errors of our `self_optimize` method.
#
# Note, that the optimize method will perform all optimizations on a copy of the pipeline.
# The means the pipeline object used as input will not be modified.
from tpcp.optimize import Optimize

# Remember we only optimize on the `train_set`.
optimized_pipe = Optimize(pipeline).optimize(train_set)
optimized_results = optimized_pipe.safe_run(test_set)
print("The optimized `min_r_peak_height_over_baseline` is", optimized_results.algorithm.min_r_peak_height_over_baseline)
print("Number of R-Peaks:", len(optimized_results.r_peak_positions_))

# %%
# We can see that training has drastically modified the threshold and increased the number of R-peaks we detected.
# To figure out, if all the new R-peaks are actually correct, we would need to make a more extensive evaluation.
#
#
# Final Notes
# -----------
# In this example we only modified a threshold of the algorithm.
# However, the concept of optimization can be expanded to anything imaginable (e.g. templates, ML-models, NN-models).
#
