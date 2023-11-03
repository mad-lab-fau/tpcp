r"""
.. _cross_validation:

Cross Validation
================

Whenever using some sort of trainable algorithm it is important to clearly separate the training and the testing data to
get an unbiased result.
Usually this is achieved by a train-test split.
However, if you don't have that much data, there is always a risk that one random train-test split, will provide
better (or worse) results than another.
In these cases it is a good idea to use cross-validation.
In this procedure, you perform multiple train-test splits and average the results over all "folds".
For more information see our :ref:`evaluation guide <algorithm_evaluation>` and the `sklearn guide on cross
validation <https://scikit-learn.org/stable/modules/cross_validation.html>`_.

In this example, we will learn how to use the :func:`~tpcp.validate.cross_validate` function implemented in
tcpc.
For this, we will redo the example on :ref:`optimizable pipelines <optimize_pipelines>` but we will perform the final
evaluation via cross-validation.
If you want to have more information on how the dataset and pipeline is built, head over to this example.
Here we will just copy the code over.
"""
# %%
# Dataset
from pathlib import Path

from examples.datasets.datasets_final_ecg import ECGExampleData

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)

# %%
# Pipeline
import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import OptimizableQrsDetector
from tpcp import OptimizableParameter, OptimizablePipeline, Parameter, cf


class MyPipeline(OptimizablePipeline):
    algorithm: Parameter[OptimizableQrsDetector]
    algorithm__min_r_peak_height_over_baseline: OptimizableParameter[float]

    r_peak_positions_: pd.Series

    def __init__(self, algorithm: OptimizableQrsDetector = cf(OptimizableQrsDetector())):
        self.algorithm = algorithm

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
        algo.detect(datapoint.data, datapoint.sampling_rate_hz)

        self.r_peak_positions_ = algo.r_peak_positions_
        return self


# %%
# The Scorer
# ----------
# The scorer is identical to the scoring function used in the other examples.
# The F1-score is still the most important parameter for our comparison.
from examples.algorithms.algorithms_qrs_detection_final import match_events_with_reference, precision_recall_f1_score


def score(pipeline: MyPipeline, datapoint: ECGExampleData):
    # We use the `safe_run` wrapper instead of just run. This is always a good idea.
    # We don't need to clone the pipeline here, as GridSearch will already clone the pipeline internally and `run`
    # will clone it again.
    pipeline = pipeline.safe_run(datapoint)
    tolerance_s = 0.02  # We just use 20 ms for this example
    matches = match_events_with_reference(
        pipeline.r_peak_positions_.to_numpy(),
        datapoint.r_peak_positions_.to_numpy(),
        tolerance=tolerance_s * datapoint.sampling_rate_hz,
    )
    precision, recall, f1_score = precision_recall_f1_score(matches)
    return {"precision": precision, "recall": recall, "f1_score": f1_score}


# %%
# Data Splitting
# --------------
# Before performing a cross validation, we need to decide on the number of folds and type of splits.
# In `tpcp` we support all cross validation iterators provided in
# `sklearn <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`__.
#
# To keep the runtime low for this example, we are going to use a 3-fold CV.
from sklearn.model_selection import KFold

cv = KFold(n_splits=3)

# %%
# Cross Validation
# ----------------
# Now we have all the pieces for the final cross validation.
# First we need to create instances of our data and pipeline.
# Then we need to wrap our pipeline instance into an :class:`~tpcp.optimize.Optimize` wrapper.
# Finally, we can call `tpcp.validate.cross_validate`.
from tpcp.optimize import Optimize
from tpcp.validate import cross_validate

pipe = MyPipeline()
optimizable_pipe = Optimize(pipe)

results = cross_validate(
    optimizable_pipe, example_data, scoring=score, cv=cv, return_optimizer=True, return_train_score=True
)
result_df = pd.DataFrame(results)
result_df

# %%
# Understanding the Results
# -------------------------
# The cross validation provides a lot of outputs (some of them can be disabled using the function parameters).
# To simplify things a little, we will split the output into four parts:
#
# The main output are the test set performance values.
# Each row corresponds to performance in respective fold.
performance = result_df[["test_precision", "test_recall", "test_f1_score"]]
performance

# %%
# The final generalization performance you would report is usually the average over all folds.
# The STD can also be interesting, as it tells you how stable your optimization is and if your splits provide
# comparable data distributions.
generalization_performance = performance.agg(["mean", "std"])
generalization_performance

# %%
# If you need more insight into the results (e.g. when the std of your results is high), you can inspect the
# individual score for each data point.
# In this example this is only a list with a single element per score, as we only had a single datapoint per fold.
# In a real scenario, this will be a list of all datapoints.
# Inspecting this list can help to identify potential issues with certain parts of your dataset.
# To link the performance values to a specific datapoint, you can look at the `test_data_labels` field.
single_performance = result_df[
    ["test_single_precision", "test_single_recall", "test_single_f1_score", "test_data_labels"]
]
single_performance

# %%
# Even further insight is provided by the train results (if activated in parameters).
# These are the performance results on the train set and can indicate if the training provided meaningful results and
# can also indicate over-fitting, if the performance of the test set is much worse than the performance on the train
# set.
train_performance = result_df[
    [
        "train_precision",
        "train_recall",
        "train_f1_score",
        "train_single_precision",
        "train_single_recall",
        "train_single_f1_score",
        "train_data_labels",
    ]
]
train_performance

# %%
# The final level of debug information is provided via the timings (note the long runtime in fold 0 can be explained
# by the jit-compiler used in `BarthDtw`) ...
timings = result_df[["score_time", "optimize_time"]]
timings

# %%
# ... and the optimized pipeline object.
# This is the actual trained object generated in this fold.
# You can apply it to other data for testing or inspect the actual object for further debug information that might be
# stored on it.
optimized_pipeline = result_df["optimizer"][0]
optimized_pipeline

# %%
optimized_pipeline.optimized_pipeline_.get_params()

# %%
# Further Notes
# -------------
# We also support grouped cross validation.
# Check the :ref:`dataset guide <custom_dataset_basics>` on how you can group the data before cross-validation or
# generate data labels to be used with `GroupedKFold`.
#
# `Optimize` is just an example of an optimizer that can be passed to cross validation.
# You can pass any `tpcp` optimizer like `GridSearch` or `GridSearchCV` or custom optimizer that implement the
# `tpcp.optimize.BaseOptimize` interface.
