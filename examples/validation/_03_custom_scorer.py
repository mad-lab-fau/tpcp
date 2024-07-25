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

In the following, we will demonstrate solutions for two typical usecases:

1. Instead of averaging the scores you want to use another metric (e.g. median) or you want to weight the scores
   based on the datatype.
2. You want to calculate a score, that can not be first aggregated on a datapoint level.
   This can happen, if each datapoint has multiple events.
   If you score (e.g. F1 score) on each datapoint first, you will get a different result, compared to calculating the F1
   score across all events of a dataset, independent of the datapoint they belong to.
   (Note, which of the two cases you want will depend on your usecase and the data distributions per datapoint)

"""
from collections.abc import Sequence
from pathlib import Path

# %%
# Setup
# -----
# We will simply reuse the pipline from the general QRS detection example.
# For all of our custom scorer, we will use this pipeline and apply it to all datapoints of the ECG example dataset.
import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import (
    QRSDetector,
    match_events_with_reference,
    precision_recall_f1_score,
)
from examples.datasets.datasets_final_ecg import ECGExampleData
from tpcp import Parameter, Pipeline, cf

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)


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


pipe = MyPipeline()

# %%
# Custom Median Scorer
# --------------------
# To create a custom score aggregation, we first need a score function.
# We will use a similar score function as we used in the QRS detection example.
# It returns the precision, recall and f1 score of the QRS detection for each datapoint.


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
# By default, these values will be aggregated by averaging over all datapoints.
# We can see that by running an instance of the scorer on the example dataset.
from tpcp.validate import Scorer

baseline_results_agg, baseline_results_single = Scorer(score)(pipe, example_data)
baseline_results_agg

# %%
baseline_results_single

# %%
# The scorer provides the results per datapoint and the aggregated values.
# We can see that the aggregation was performed using the average
import numpy as np

assert baseline_results_agg["f1_score"] == np.mean(baseline_results_single["f1_score"])


# %%
# We can change this behaviour by implementing a custom Aggregator.
# This is a simple class inheriting from :class:`tpcp.validate.Aggregator`, implementing a `aggregate` class - method.
# This method gets the score values and the datapoints that generated them as keyword only arguments.
# (Note, if you need just the values and not the datapoints, you can use the `**_` syntax to catch all unused parameters.)
#
# Below we have implemented a custom aggregator that calculates the median of the per-datapoint scores.
# In addition, it prints a log message when it is called, so we can better understand how it works.
from tpcp.validate import Aggregator
from tpcp.validate._scorer import FloatAggregator

median_agg = FloatAggregator(np.median)


# %%
# We can apply this Aggregator in two ways:
#
# 1. By using it as `default_aggregator` in the Scorer constructor.
#    In this case, the aggregator will be used for all scores.
# 2. By wrapping specific return values of the score method.
#
# Let's start with the first way.
median_results_agg, median_results_single = Scorer(score, default_aggregator=median_agg)(pipe, example_data)
median_results_agg
# %%
# We can see via the log-printing that the aggregator was called 3 times (once per score).
assert median_results_agg["f1_score"] == np.median(median_results_single["f1_score"])
assert median_results_agg["precision"] == np.median(median_results_single["precision"])

# %%
# In the second case, we can select which scores we want to aggregate in a different way.
# All scores without a specific aggregator will be aggregated by the default aggregator.
#
# Below, only the F1-score will be aggregated by the median aggregator.


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
    return {"precision": precision, "recall": recall, "f1_score": median_agg(f1_score)}


partial_median_results_agg, partial_median_results_single = Scorer(score)(pipe, example_data)
partial_median_results_agg

# %%
assert partial_median_results_agg["f1_score"] == np.median(partial_median_results_single["f1_score"])
assert partial_median_results_agg["precision"] == np.mean(partial_median_results_single["precision"])

# %%
# .. warning:: Note, that you score function must return the same aggregator for a score across all datapoints.
#              If not, we will raise an error!


# %%
# Multi-Return Aggregator
# -----------------------
# Sometimes an aggregator needs to return multiple values.
# We can easily do that, by returning a dict from the `aggregate` method.
#
# As example, we will calculate the mean and standard deviation of the returned scores in one aggregation.
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


def mean_and_std(vals: Sequence[float]):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


MeanAndStdAggregator = FloatAggregator(mean_and_std)


multi_agg_agg, multi_agg_single = Scorer(score, default_aggregator=MeanAndStdAggregator)(pipe, example_data)

# %%
# When multiple values are returned, the names are concatenated with the names of the scores using `__`.
multi_agg_agg

# %%
# Complicated Aggregation
# -----------------------
# In cases where we do not want to or can not aggregate the scores on a per-datapoint basis, we can return arbitrary
# data from the score function and pass it to a complex aggregator.
# There are no restrictions on the data you can pass from the scorer.
# Only the aggregator needs to be able to handle the values and then return a float or a dict with float values.
#
# In this example, we will use a custom aggregator to calculate the precision, recall and f1-score without
# aggregating on a datapoint level first.
# For that we return the raw `matches` from the score function and wrap them into an aggregator that concatenates all
# of them, before throwing them into the `precision_recall_f1_score` function.


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
# `return_raw_scores` class variable to
# False for the our specific
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
from typing import Callable, Union


class SingleValueAggregator(Aggregator[np.ndarray]):
    def __init__(
        self, func: Callable[[Sequence[np.ndarray]], Union[float, dict[str, float]]], *, return_raw_scores: bool = True
    ):
        self.func = func
        super().__init__(return_raw_scores=return_raw_scores)

    def aggregate(self, /, values: Sequence[np.ndarray], **_) -> dict[str, float]:
        print("SingleValueAggregator Aggregator called")
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
# We can even move the initialization of the aggregator into the score function.
# This allows us to make the score function itself generalizable.
#
# This works, because we check if the aggregators all have the same config, but we don't enforce them to all be the
# same object.
#
# This allows for quite powerful and flexible scoring functions that we could then use with `partial` to create
# different versions of the score function.
from functools import partial


def score(pipeline: MyPipeline, datapoint: ECGExampleData, *, agg_func: Callable, return_raw_scores: bool = True):
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
        "per_sample": SingleValueAggregator(agg_func, return_raw_scores=return_raw_scores)(matches),
    }


# %%
# With that we can reconstruct the `return_raw_scores=False` behaviour from before.
complicated_agg_now_raw, complicated_single_no_raw = Scorer(
    partial(score, agg_func=calculate_precision_recall_f1, return_raw_scores=False)
)(pipe, example_data)
complicated_agg_now_raw

# %%
complicated_single_no_raw.keys()

# %%
# Weighted Aggregation
# --------------------
# So far all aggregators only used the values for aggregation.
# However, sometimes we want to treat values differently depending on where they came from.
# For these "complicated" weighting cases, we can use the `datapoint` parameter that is passed to the `aggregate`
# method.
#
# In the following example, we want to calculate the Macro Average over all participant groups (see dataset below).
# This means, we want to average the parameters first in each group and then average the results.
example_data


# %%
# For this, we use everything we learned before and create a general `GroupWeightedAggregator`.
# The aggregator will apply an arbitrary function to the values, but group the values by a specific column in the
# datapoint index first and then take the average over the results.
class MacroAgg(Aggregator[float]):
    def __init__(
        self, func: Union[Callable[[Sequence[float]], float], str], groupby: str, *, return_raw_scores: bool = True
    ):
        self.func = func
        self.groupby = groupby
        super().__init__(return_raw_scores=return_raw_scores)

    def aggregate(self, /, values: Sequence[float], datapoints: Sequence[ECGExampleData], **_) -> dict[str, float]:
        patient_groups = [d.group_label for d in datapoints]
        data_index = pd.MultiIndex.from_tuples(patient_groups, names=patient_groups[0]._fields)

        data = pd.DataFrame({"value": values}, index=data_index)
        per_group = data.groupby(self.groupby).agg(self.func)["value"]
        return {**per_group.to_dict(), "group_mean": per_group.mean()}


macro_mean = MacroAgg("mean", "patient_group")


# %%
# In our score function, we wrap the f1-score with the new aggregator (we could of cause also wrap the others,
# or use the `default_aggregator` parameter).
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
    return {"precision": precision, "recall": recall, "f1_score": macro_mean(f1_score)}


group_weighted_agg, group_weighted_single = Scorer(score)(pipe, example_data)
group_weighted_agg


# %%
# No-Aggregation Aggregator
# -------------------------
# Sometimes you might want to return data from a score function that should not be aggregated.
# This could be arbitrary metadata or scores will value that can not be averaged.
# In this case you can simply use the :class:`~tpcp.validate.NoAgg` aggregator.
# This will return only the single values and no aggregated items.
#
# In the example below, we will only aggregate the precision and recall, but not the f1-score.
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
    return {"precision": precision, "recall": recall, "f1_score": no_agg(f1_score)}


# %%
# We can see that the f1-score is not contained in the aggregated results.
no_agg_agg, no_agg_single = Scorer(score)(pipe, example_data)
no_agg_agg

# %%
# But we can still access the value in the single results.
no_agg_single["f1_score"]
