"""
.. _custom_scorer:

Custom Scorer
=============

Scorer or scoring functions are used in tpcp whenever we need to rank any form of output.
For examples, after a GridSearch, we want to know which pipeline is the best.
This is done by a function, that takes a pipeline and a datapoint as an input and returns one or multiple score.
These scores are then averaged over all datapoints provided.

However, sometimes this is not exactly what we want.
In this case, you need to create a custom scorer or custom aggregator to also control how scores are averaged over all
datapoints.

Four general usecases arise for custom scorers:

1. You actually don't want to score anything, but just want to collect some metadata, or pass results out of the method
   unchanged for later analysis. This can be easily done using :func:`~tpcp.validate.no_agg` (See first example below)
2. You can properly calculate a performance value on a single datapoint, but you don't want to take the mean over all
   datapoints, but rather use a different aggregation metrics (e.g. median, ...).
   This can be done by using the existing :class:`~tpcp.validate.FloatAggregator` class with a new function (See second
   and third example below)
3. Similar to 3, but you require additional information passed through the aggregation function. This could be the
   datapoints itself (e.g. to calculate a Macro Average) or some other metadata required for the aggregation.
   This can be done by inheriting from the :class:`~tpcp.validate.Aggregator` class and implementing the `aggregate`
   method (See fourth example below).
4. You want to calculate a score, that can not be first aggregated on a datapoint level.
   For example, you are detecting events in a dataset and you want to calculate the F1 score across all events of a
   dataset, without first aggregating the F1 score on a datapoint level.

"""
from collections.abc import Sequence
from pathlib import Path

# %%
# Setup
# -----
# We will simply reuse the pipline from the general QRS detection example.
# For all of our custom scorer, we will use this pipeline and apply it to all datapoints of the ECG example dataset.
from examples.algorithms.algorithms_qrs_detection_final import (
    match_events_with_reference,
)
from examples.datasets.datasets_final_ecg import ECGExampleData

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)

import pandas as pd
from joblib.memory import Memory

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector, precision_recall_f1_score
from examples.datasets.datasets_final_ecg import ECGExampleData
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
# We set up a global cache for our pipeline to speed up the repeated evaluation we do below.
from tpcp.caching import global_disk_cache

global_disk_cache(memory=Memory("./.cache"), restore_in_parallel_process=True, action_method_name="run")(MyPipeline)

pipe = MyPipeline()

# %%
# No Aggregation
# --------------
# Sometimes you might want to return data from a score function that should not be aggregated.
# This could be arbitrary metadata or scores will value that can not be averaged.
# In this case you can simply use the :func:`~tpcp.validate.no_agg` aggregator.
# This will return only the single values and no aggregated items.
#
# In the example below, we will calculate the precision, recall and f1-score for each datapoint and in addition return
# the number of labeled reference values as "metadata".
# This metadata will not be aggregated, but still be available in the single results.
#
# .. note:: At the moment we don't support returning only no-aggregated from a scorer.
#           At least one value must be aggregated, so that it can be used to rank results.
#           If you really need this (e.g. in combination with :func:`~tpcp.validate.validate`), you can return a dummy
#           value that is not used in the aggregation.
from tpcp.validate import no_agg


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
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "n_labels": no_agg(len(datapoint.r_peak_positions_)),
    }


# %%
# We can see that the n_labels is not contained in the aggregated results.
from tpcp.validate import Scorer

no_agg_agg, no_agg_single = Scorer(score)(pipe, example_data)
no_agg_agg

# %%
# But we can still access the value in the single results.
no_agg_single["n_labels"]

# %%
# Custom Median Scorer
# --------------------
# If we want to change the way the scores are aggregated, we can use a custom aggregator.
# For simple cases, this does not require to implement a new class, but we can use the
# :class:`~tpcp.validate.FloatAggregator` directly.
# It assumes that we have a function that takes a sequence of floats and returns a float.
#
# Aggregators are simply instances of the :class:`~tpcp.validate.Aggregator` classes.
# So we can create a new instance of the :class:`~tpcp.validate.FloatAggregator` with a new function.
#
# Below we simply use the median as an example.
import numpy as np

from tpcp.validate import FloatAggregator

median_agg = FloatAggregator(np.median)

# %%
# Then we reuse the score function from before and wrap the F1-score with the median aggregator.
# For all other values, the default aggregator will be used (which is the mean).


# .. warning:: Note, that you score function must return the same aggregator for a scores across all datapoints.
#              If not, we will raise an error!
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
    return {"precision": precision, "recall": recall, "f1_score": f1_score, "median_f1_score": median_agg(f1_score)}


# %%
median_results_agg, median_results_single = Scorer(score)(pipe, example_data)
median_results_agg

# %%
assert median_results_agg["median_f1_score"] == np.median(median_results_single["f1_score"])
assert median_results_agg["f1_score"] == np.mean(median_results_single["f1_score"])

# %%
# .. note:: We could also change the default aggregator for all scores by using the `default_aggregator` parameter of
#           the :class:`~tpcp.validate.Scorer` class (See the next example).
# Let's start with the first way.
all_median_results_agg, all_median_results_single = Scorer(score, default_aggregator=median_agg)(pipe, example_data)
median_results_agg
# %%
# We can see via the log-printing that the aggregator was called 3 times (once per score).
assert all_median_results_agg["f1_score"] == np.median(all_median_results_single["f1_score"])
assert all_median_results_agg["precision"] == np.median(all_median_results_single["precision"])

# %%
# Multi-Return Aggregator
# -----------------------
# Sometimes an aggregator needs to return multiple values.
# We can easily do that, by returning a dict from the `aggregate` method or in case of the `FloatAggregator` by passing
# a function that returns a dict.
#
# As example, we will calculate the mean and standard deviation of the returned scores in one aggregation.
# This could be applied individually to each score (as seen in the previous example) or to all scores at once using
# the `default_aggregator` parameter.
# We will demonstrate the latter here.


def mean_and_std(vals: Sequence[float]):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


mean_and_std_agg = FloatAggregator(mean_and_std)


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


multi_agg_agg, multi_agg_single = Scorer(score, default_aggregator=mean_and_std_agg)(pipe, example_data)

# %%
# When multiple values are returned, the names are concatenated with the names of the scores using `__`.
multi_agg_agg

# %%
# Macro Aggregation
# -----------------
# In some datasets (in particular, when we have multiple recordings per participant), we might want to calculate a
# single performance value for each participant and then average these values.
# Fundamentally, this is a little tricky with tpcp, as all of our processing happens per datapoint, and each datapoint
# is usually one recording, to simplify the pipeline structures.
#
# Hence, we need to shift some of the aggregation complexity into our scoring function.
# As this is a little complicated and such a common usecase, tpcp provides a helper class for this:
# :class:`~tpcp.validate.MacroFloatAggregator`.
# It allows us to define an initial grouping based on the dataset index columns and define how values are aggregated
# first per group and then across all groups.
from tpcp.validate import MacroFloatAggregator

macro_average_patient_group = MacroFloatAggregator(groupby="patient_group", group_agg=np.mean, final_agg=np.mean)

# %%
# We will apply this aggregation to the F1-score:


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
    return {"precision": precision, "recall": recall, "f1_score": macro_average_patient_group(f1_score)}


# %%
# We can see that we now get the "single" aggregation values per group (`f1_score__{group}`) and the final aggregated
# values (`f1_score__macro`).
macro_agg, macro_single = Scorer(score)(pipe, example_data)
macro_agg

# %%
# The raw values are still available in the single results.
macro_single["f1_score"]

# %%
# So far we did not need to implement a fully custom aggregation, as we `tpcp` could provide helper funcs for typical
# usecases.
# However, if you need to do more complicated things with your score values, or pass other things than floats to your
# scores, you will need a custom aggregator as shown in the next example.

# %%
# Fully Custom Aggregation
# ------------------------
# In the next example, we want to aggregate on a "lower" level than a single datapoint.
# In the previous example, where we wanted to aggregate first on a "higher" level than a single datapoint.
# In this case we could provide tpcp-helper, as the higher levels were defined by the used `Dataset`.
# Hence, we could make some assumptions about how the passed data will look like.
#
# However, if you want to go more granular as a single datapoint, we can not know what datastructures you are dealing
# with.
# Therefore, you need to create a completely custom aggregation by subclassing :class:`~tpcp.validate.Aggregator`.
#
# Below we show an example, where we calculate the precision, recall and f1-score without aggregating on a datapoint
# level, but rather first combining all predictions and references across all datapoints before calculating the
# precision, recall and f1-score.
#
# There are no restrictions on the data you can pass from the scorer.
# Only the aggregator needs to be able to handle the values and then return a float or a dict with float values.
#
# In this example, we will use a custom aggregator to calculate the precision, recall and f1-score without
# aggregating on a datapoint level first.
# For that we return the raw `matches` from the score function and wrap them into an aggregator that concatenates all
# of them, before throwing them into the `precision_recall_f1_score` function.
#
# Note, that the actual aggregation is an instance of our custom class, NOT the class itself.
from tpcp.validate import Aggregator


class SingleValuePrecisionRecallF1(Aggregator[np.ndarray]):
    def aggregate(self, /, values: Sequence[np.ndarray], **_) -> dict[str, float]:
        print("SingleValuePrecisionRecallF1 Aggregator called")
        precision, recall, f1_score = precision_recall_f1_score(np.vstack(values))
        return {"precision": precision, "recall": recall, "f1_score": f1_score}


single_value_precision_recall_f1_agg = SingleValuePrecisionRecallF1()


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
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_sample": single_value_precision_recall_f1_agg(matches),
    }


# %%
# We can see that we now get the values per datapoint (as before) and the values without previous aggregation.
# From a scientific perspective, we can see that these values are quite different.
# Again, which version to choose for scoring will depend on the use case.
complicated_agg, complicated_single = Scorer(score)(pipe, example_data)
complicated_agg

# %%
# The raw matches array is still available in the `single` results.
complicated_single["per_sample"]

# %%
# However, we can customize this behaviour for our aggregator by creating an instance of the aggregator in which we set
# `return_raw_scores` class variable to False for our specific usecase.
single_value_precision_recall_f1_agg_no_raw = SingleValuePrecisionRecallF1(return_raw_scores=False)


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
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_sample": single_value_precision_recall_f1_agg_no_raw(matches),
    }


# %%
# Now we can see that the raw matches array is not returned anymore.
# In case of a single scorer, the single return value would just be `None`, instead of a dict with the respective key
# missing.
complicated_agg_now_raw, complicated_single_no_raw = Scorer(score)(pipe, example_data)
complicated_single_no_raw.keys()


# %%
# Generalizing the custom aggregator
# ----------------------------------
# In the previous examples, that can calculate values after concatenating all values.
# However, it only works for the precision, recall and f1-score.
# We can generalize this, by extracting the calculation of the precision, recall and f1-score into a parameter of
# the aggregator.
# This way, we can use the same aggregator for different scores.
#
# Note, that we don't provide such generalized aggregators in tpcp on purpose, as they really depend on the specific
# usecase, your data, and the type of scores you want to calculate.
# Hence, we recommend to use these examples as a starting point to implement your own custom aggregators.
from typing import Callable, Union


class SingleValueAggregator(Aggregator[np.ndarray]):
    def __init__(
        self, func: Callable[[Sequence[np.ndarray]], Union[float, dict[str, float]]], *, return_raw_scores: bool = True
    ):
        self.func = func
        super().__init__(return_raw_scores=return_raw_scores)

    def aggregate(self, /, values: Sequence[np.ndarray], **_) -> dict[str, float]:
        return self.func(np.vstack(values))


# %%
# With this our aggregator from before becomes just a special case of the new aggregator.
def calculate_precision_recall_f1(matches: Sequence[np.ndarray]) -> dict[str, float]:
    precision, recall, f1_score = precision_recall_f1_score(np.vstack(matches))
    return {"precision": precision, "recall": recall, "f1_score": f1_score}


single_value_precision_recall_f1_agg_from_gen = SingleValueAggregator(calculate_precision_recall_f1)


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
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_sample": single_value_precision_recall_f1_agg_from_gen(matches),
    }


complicated_agg, complicated_single = Scorer(score)(pipe, example_data)
complicated_agg

# %%
# We can even move the initialization of the aggregator into the score function, or pass the Aggregator itself as a
# parameter.
# This allows us to make the score function itself generalizable.
#
# This works, because we check if the aggregators all have the same config, but we don't enforce them to all be the
# same object.
#
# This allows for quite powerful and flexible scoring functions that we could then use with `partial` to create
# different versions of the score function.
#
# While we are at it, we also make the `tolerance_s` a parameter of the score function.


def score(
    pipeline: MyPipeline, datapoint: ECGExampleData, *, tolerance_s: float, per_sample_agg: Aggregator[np.ndarray]
):
    # We use the `safe_run` wrapper instead of just run. This is always a good idea.
    # We don't need to clone the pipeline here, as GridSearch will already clone the pipeline internally and `run`
    # will clone it again.
    pipeline = pipeline.safe_run(datapoint)
    matches = match_events_with_reference(
        pipeline.r_peak_positions_.to_numpy(),
        datapoint.r_peak_positions_.to_numpy(),
        tolerance=tolerance_s * datapoint.sampling_rate_hz,
    )
    precision, recall, f1_score = precision_recall_f1_score(matches)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_sample": per_sample_agg(matches),
    }


# %%
# With that we can reconstruct the `return_raw_scores=False` behaviour from before using a partial or a lambda
from functools import partial

complicated_agg_now_raw, complicated_single_no_raw = Scorer(
    partial(
        score,
        per_sample_agg=SingleValueAggregator(calculate_precision_recall_f1, return_raw_scores=False),
        tolerance_s=0.02,
    ),
    # Note: You could also run this with multiple jobs, but this creates issues with the way we test the examples.
    n_jobs=1,
)(pipe, example_data)
complicated_agg_now_raw

# %%
complicated_single_no_raw.keys()


# %%
# And finally remove the cache to not affect other examples.
from tpcp.caching import remove_any_cache

remove_any_cache(MyPipeline)
