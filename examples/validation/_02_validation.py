r"""
.. _validation:

Validation
================

TODO: when to use validation, when cv? what is the difference bw them?
Whenever using some sort of trainable algorithm it is important to clearly separate the training and the testing data to
get an unbiased result.
Usually this is achieved by a train-test split.
However, if you don't have that much data, there is always a risk that one random train-test split, will provide
better (or worse) results than another.
In these cases it is a good idea to use cross-validation.
In this procedure, you perform multiple train-test splits and average the results over all "folds".
For more information see our :ref:`evaluation guide <algorithm_evaluation>` and the `sklearn guide on cross
validation <https://scikit-learn.org/stable/modules/cross_validation.html>`_.

In this example, we will learn how to use the :func:`~tpcp.validate.validate` function implemented in
tcpc.
For this, we will reuse the pipeline from the example on :ref:` gridsearch <grid_search>` and the data from the example
on :ref:` optimizable pipelines <optimizable_pipelines`.
If you want to have more information on how the dataset and pipeline is built, head over to these examples.
Here we will just copy the code over.
"""
# %%
# Dataset
from pathlib import Path

import numpy as np

from examples.datasets.datasets_final_ecg import ECGExampleData

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path(".").resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)

# %%
# Pipeline
import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector
from tpcp import Parameter, Pipeline, cf


class MyPipeline(Pipeline[ECGExampleData]):
    algorithm: Parameter[QRSDetector]

    r_peak_positions_: pd.Series

    def __init__(self, algorithm: QRSDetector = cf(QRSDetector())):
        self.algorithm = algorithm

    def run(self, datapoint: ECGExampleData):
        # Note: We need to clone the algorithm instance, to make sure we don't leak any data between runs.
        algo = self.algorithm.clone()
        algo.detect(datapoint.data["ecg"], datapoint.sampling_rate_hz)

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
# Validation
# ----------------
# Now we have all the pieces for the final validation.
# First we need to create instances of our data and pipeline.
# Finally, we can call `tpcp.validate.validate`.
from tpcp.optimize import Optimize
from tpcp.validate import validate

pipe = MyPipeline()

results = validate(pipe, example_data, scoring=score)
result_df = pd.DataFrame(results)
result_df

# TODO: add example on parallelization
# TODO
# %%
# Understanding the Results
# -------------------------
# The cross validation provides a lot of outputs (some of them can be disabled using the function parameters).
# To simplify things a little, we will split the output into four parts:
#
# The main output are the dataset performance values.
performance = result_df[["precision", "recall", "f1_score"]]
performance

# %%
# If you need more insight into the results, you can inspect the
# individual score for each data point.
# In this example this is only a list with a single element per score, as we only had a single datapoint.
# In a real scenario, this will be a list of all datapoints.
# Inspecting this list can help to identify potential issues with certain parts of your dataset.
# To link the performance values to a specific datapoint, you can look at the `data_labels` field.
single_performance = result_df[["single_precision", "single_recall", "single_f1_score", "data_labels"]]
single_performance

# %%
# The final level of debug information is provided via the timings.
timings = result_df[["score_time"]]
timings

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
