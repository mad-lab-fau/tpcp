r"""
.. _validation_example:

Validation
================

Whenever using some sort of algorithm that has fixed parameters already, for example from previous work, and you simply
want to test its performance on your data, you can use validation.
Note that this is not the correct approach if you need to optimize parameters, e.g., when training or evaluating
a newly developed algorithm.
In this case, you should use :ref:`cross validation <cross_validation>` instead.

In this example, we will learn how to use the :func:`~tpcp.validate.validate` function implemented in tpcp.
For this, we will reuse the pipeline and data from the example on :ref:`gridsearch <grid_search>`.
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
# Finally, we can call :func:`~tpcp.validate.validate`.
from tpcp.validate import validate

pipe = MyPipeline()

results = validate(pipe, example_data, scoring=score)
result_df = pd.DataFrame(results)
result_df


# %%
# Understanding the Results
# -------------------------
# The validation provides a lot of outputs.
# To simplify things a little, we will split the output into three parts:
#
# The main output are the means of the performance values over all datapoints.
# They are all prefixed with `agg__` to make it easy to filter them out within the results.
# Note that if you want to use different aggregation methods, you can create and pass a custom scorer to
# :func:`~tpcp.validate.validate`. See the example on :ref:`custom scorers <custom_scorer>` for further details.
performance = result_df.filter(like="agg__")
performance

# %%
# If you need more insight into the results, you can inspect the
# individual score for each data point given in a list. In this example, we had 12 data points.
# Thus, we retrieve have 12 values for each score.
# These values are all prefixed with `single__`.
# Inspecting this list can help to identify potential issues with certain parts of your dataset.
# To link the performance values to a specific datapoint, you can look at the `data_labels` field.
single_performance = result_df.filter(like="single__")
single_performance

# %%
# It is often quite handy to explode this dataframe and combine it with the data labels.
# This way, you can easily identify the datapoints that are causing issues.
exploded_results = (
    single_performance.explode(single_performance.columns.to_list())
    .rename_axis("fold")
    .set_index(result_df["data_labels"].explode(), append=True)
)
exploded_results

# %%
# The final level of debug information is provided via the timings.
timings = result_df.filter(like="debug__")
timings

# %%
# Further Notes
# -------------
# For large amounts of data, we also support parallel processing of data points. This can be enabled by setting the
# `n_jobs` parameter in the :func:`~tpcp.validate.validate` to the number of parallel workers you want to use.
# Furthermore, you can configure the verbosity level and the number of pre-dispatched batches using the `verbose` and
# `pre_dispatch` parameter, respectively.
# For more details, check the documentation of the utilized
# `joblib.Parallel <https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html>` class.
