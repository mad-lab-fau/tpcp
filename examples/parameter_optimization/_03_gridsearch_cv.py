r"""
.. _gridsearch_cv:

GridSearchCV
============

When trying to optimize parameters for algorithms that have trainable components, it is required to perform
the parameter search on a validation set (that is separate from the test set used for the final validation).
Even better, is to use a cross validation for this step.
In tpcp this can be done by using :class:`~tpcp.optimize.GridSearchCV`.

This example explains how to use this method.
To learn more about the concept, review the :ref:`evaluation guide <algorithm_evaluation>` and the `sklearn guide on
tuning hyperparameters <https://scikit-learn.org/stable/modules/grid_search.html#grid-search>`_.

"""
import random

import pandas as pd

random.seed(1)  # We set the random seed for repeatable results

# %%
# Dataset
# -------
# As always, we need a dataset, a pipeline, and a scoring method for a parameter search.
# Here, we're just going to reuse the ECGExample dataset we created in :ref:`custom_dataset_ecg`.
from pathlib import Path

from examples.datasets.datasets_final_ecg import ECGExampleData

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path(".").resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)

from examples.algorithms.algorithms_qrs_detection_final import OptimizableQrsDetector

# %%
# The Pipeline
# ------------
# When using `GridSearchCV` our pipeline must be "optimizable".
# Otherwise, we have no need for the CV part and could just use a simple gridsearch.
# Here we are going to create an optimizable pipeline that wraps the optimizable version of the QRS detector we
# developed in :ref:`custom_algorithms_qrs_detection`.
#
# An optimzable pipeline usaully needs the following things:
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
#    If you want to adjust multiple parameters that all belong to the same algorithm, it might also be convenient to
#    just pass the algorithm as a parameter. However, keep potential issues with mutable defaults in mind (:ref:`more
#    info <mutable_defaults>`). As `OptimizableQrsDetector` is a tpcp-algorithm class, we can do that in our case.
# 5. At least one of the input parameters must be marked as `OptimizableParameter` in the class-level typehints.
#    If parameters are nested tpcp objects you can use the `__` syntax to mark nested values as optimizable.
#    Note, that you always need to mark the parameters you want to optimize in the current pipeline.
#    Annotations in nested objects are not considered.
#    The more precise you are with these annotations, the more help the runtime checks in tpcp can provide.
# 6. (Optionally) Mark parameters as `PureParameter` using the type annotations. This can be used by GridSearchCv to
#    apply some performance optimizations. However, be careful with that! In our case, there are not `PureParameters`,
#    As all (nested) input parameters change the output of the `self_optimize` method.
#    Todo: Full dedicated example for `PureParameter`
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


pipe = MyPipeline()


# %%
# The Scorer
# ----------
# The scorer is identical to the scoring function used in the other examples.
# The F1-score is still the most important parameter for our comparison.
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
# Data Splitting
# --------------
# Like with a normal cross validation, we need to decide on the number of folds and type of splits.
# In `tpcp` we support all cross validation iterators provided in :ref:`sklearn
# <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`.
#
# To keep the runtime low for this example, we are going to use a 2-fold CV.
from sklearn.model_selection import KFold

cv = KFold(n_splits=2)

# %%
# The Parameters
# --------------
# The pipeline above exposes a couple of (nested) parameters.
# `min_r_peak_height_over_baseline` is the parameter we want to optimize.
# All other parameters are effectively hyper-parameters as they change the outcome of the optimization.
# We could differentiate further and say that only `r_peak_match_tolerance_s` is a true hyper parameter, as it only
# effects the outcome of the optimization, but the `run` method is independent from it.
# `max_heart_rate_bpm` and `high_pass_filter_cutoff_hz` effect both the optimization and `run`.
#
# We could run the gridsearch over any combination of parameters.
# However, to keep things simple, we will only test a couple of values for `high_pass_filter_cutoff_hz`.
from sklearn.model_selection import ParameterGrid

parameters = ParameterGrid({"algorithm__high_pass_filter_cutoff_hz": [0.25, 0.5, 1]})

# %%
# GridSearchCV
# ------------
# Setting up the GridSearchCV object is similar to the normal GridSearch, we just need to add the additional `cv`
# parameter.
# Then we can simply run the search using the `optimize` method.
from tpcp.optimize import GridSearchCV

gs = GridSearchCV(pipeline=MyPipeline(), parameter_grid=parameters, scoring=score, cv=cv, return_optimized="f1_score")
gs = gs.optimize(example_data)

# %%
# Results
# -------
# The output is also comparable to the output of the GridSearch.
# The main results are stored in the `cv_results_` parameter.
# But instead of just a single performance value per parameter, we get one value per fold and the mean and std over
# all folds.
results = gs.cv_results_
results_df = pd.DataFrame(results)

results_df

# %%
# The mean score is the primary parameter used to select the best parameter combi (if `return_optimized` is True).
# All other values performance values are just there to provide further inside.

results_df[["mean_test_precision", "mean_test_recall", "mean_test_f1_score"]]

# %%
# For even more insight, you can inspect the scores per datapoint:

results_df.filter(like="test_single")

# %%
# If `return_optimized` was set to True (or the name of a score), a final optimization is performed using the best
# set of parameters and **all** the available data.
# The resulting pipeline will be stored in `optimizable_pipeline_`.
print("Best Para Combi:", gs.best_params_)
print("Paras of optimized Pipeline:", gs.optimized_pipeline_.get_params())

# %%
# To run the optimized pipeline, we can directly use the `run`/`safe_run` method on the GridSearch object.
# This makes it possible to use the `GridSearch` as a replacement for your pipeline object with minimal code changes.
#
# If you tried to call `run`/`safe_run` (or `score` for that matter), before the optimization, an error is
# raised.
r_peaks = gs.safe_run(example_data[0]).r_peak_positions_
r_peaks
