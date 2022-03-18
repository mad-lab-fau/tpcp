r"""
.. _grid_search:

Grid Search optimal Algorithm Parameter
=======================================

In case no better way exists to optimize a parameter of a algorithm or pipeline an exhaustive Gridsearch might be a
good idea.
`tpcp` provides a Gridsearch that is algorithm agnostic (as long as you can wrap your algorithm into a pipeline).

As example, we are going to Gridsearch some parameters of the `QRSDetector` we implemented in
:ref:`custom_algorithms_qrs_detection`.

"""
# %%
# To perform a GridSearch (or any other form of parameter optimization in Gaitmap), we first need to have a
# **Dataset**, a **Pipeline** and a **score** function.
#
# 1. The Dataset
# --------------
# Datsets wrap multiple recordings into an easy-to-use interface that can be passed around between the higher
# level `tpcp` functions.
# Learn more about this :ref:`here <custom_datasets>`.
# If you are lucky, you do not need to create the dataset on your own, but someone has already created a dataset
# for the data you want to use.
#
# Here, we're just going to reuse the ECGExample dataset we created in :ref:`custom_dataset_ecg`.
#
# For our GridSearch, we need an instance of this dataset.
from pathlib import Path

from examples.datasets.datasets_final_ecg import ECGExampleData

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path(".").resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)

# %%
# 1. The Pipeline
# ---------------
# The pipeline simply defines what algorithms we want to run on our data and defines, which parameters of the pipeline
# you still want to be able to modify (e.g. to optimize in the GridSearch).
#
# The pipeline usually needs 3 things:
#
# 1. It needs to be subclass of :class:`~tpcp.Pipeline`.
# 2. It needs to have a `run` method that runs all the algorithmic steps and stores the results as class attributes.
#    The `run` method should expect only a single data point (in our case a single recording of one sensor) as input.
# 3. A `init` that defines all parameters that should be adjustable. Note, that the names in the function signature of
#    the `init` method, **must** match the corresponding attribute names (e.g. `max_cost` -> `self.max_cost`).
#    If you want to adjust multiple parameters that all belong to the same algorithm, it might also be convenient to
#    just pass the algorithm as a parameter. However, keep potential issues with mutable defaults in mind (:ref:`more
#    info <mutable_defaults>`).
#
# Here we simply extract the data and sampling rate from the datapoint and then run the algorithm.
# We store the final results we are interested in on the pipeline object.
#
# For the final GridSearch, we need an instance of the pipeline object.
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


pipe = MyPipeline()


# %%
# 3. The scorer
# -------------
# In the context of a gridsearch, we want to calculate the performance of our algorithm and rank the different
# parameter candidates accordingly.
# This is what our score function is for.
# It gets a pipeline object (**without** results!) and a data point (i.e. a single recording) as input and should
# return a some sort of performance metric.
# A higher value is always considered better.
# If you want to calculate multiple performance measures, you can also return a dictionary of such values.
# In any case, the performance for a specific parameter combination in the GridSearch will be calculated as the mean
# over all datapoints.
# (Note, if you want to change this, you can create custom subclasses of :class:`~tpcp.validate.Scorer`).
#
# A typical score function will first call `safe_run` (which calls `run` internally) on the pipeline and then
# compare the output with some reference.
# This reference should be supplied as part of the dataset.
#
# Instead of using a function as scorer (shown here), you can also implement a method called `score` on your pipeline.
# Then just pass `None` (which is the default) for the `scoring` parameter in the GridSearch (and other optimizers).
# However, a function is usually more flexible.
#
# In this case we compare the identified R-peaks with the reference and identify which R-peaks were correctly
# found within a certain margin around the reference points
# Based on these matches, we calculate the precision, the recall, and the f1-score using some helper functions.
from examples.algorithms.algorithms_qrs_detection_final import match_events_with_reference


def score(pipeline: MyPipeline, datapoint: ECGExampleData):
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
    precision = n_tp / len(pipeline.r_peak_positions_)
    recall = n_tp / len(datapoint.r_peak_positions_)
    f1_score = (2 * n_tp) / (len(pipeline.r_peak_positions_) + len(datapoint.r_peak_positions_))
    return {"precision": precision, "recall": recall, "f1_score": f1_score}


# %%
# The Parameters
# --------------
# The last step before running the GridSearch, is to select the parameters we want to test for each dataset.
# For this, we can directly use sklearn's `ParameterGrid`.
#
# In this example, we will just test three values for the `high_pass_filter_cutoff_hz`.
# As this is a nested paramater, we use the `__` syntax to set it.
from sklearn.model_selection import ParameterGrid

parameters = ParameterGrid({"algorithm__high_pass_filter_cutoff_hz": [0.25, 0.5, 1]})

# %%
# Running the GridSearch
# ----------------------
# Now we have all the pieces to run the GridSearch.
# After initializing, we can use `optimize` to run the GridSearch.
#
# .. note:: If the score function returns a dictionary of scores, `rank_scorer` must be set to the name of the score,
#           that should be used to decide on the best parameter set.
from tpcp.optimize import GridSearch

gs = GridSearch(pipe, parameters, scoring=score, return_optimized="f1_score")
gs = gs.optimize(example_data)

# %%
# The main results are stored in `gs_results_`.
# It shows the mean performance per parameter combination, the rank for each parameter combination and the
# performance for each individual data point (in our case a single recording of one participant).
results = gs.gs_results_
results

#%%
pd.DataFrame(results)

# %%
# Further, the `optimized_pipeline_` parameter holds an instance of the pipeline initialized with the best parameter
# combination.
print("Best Para Combi:", gs.best_params_)
print("Paras of optimized Pipeline:", gs.optimized_pipeline_.get_params())

# %%
# To run the optimized pipeline, we can directly use the `run`/`safe_run` method on the GridSearch object.
# This makes it possible to use the `GridSearch` as a replacement for your pipeline object with minimal code changes.
#
# If you tried to call `run`/`safe_run` (or `score` for that matter), before the optimization, an error is raised.
r_peaks = gs.safe_run(example_data[0]).r_peak_positions_
r_peaks
