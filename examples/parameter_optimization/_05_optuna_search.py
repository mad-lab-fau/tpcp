r"""
.. _build_in_optuna_optimizer:

Build-in Optuna Optimizers
==========================
The :ref:`custom optuna example <custom_optuna_optimizer>` shows how to implement a specific optuna optimizer with
full control over all aspects.
This is still the recommended way to do things, as you often will have specific requirements for your objective
function.

However, there are still a number of problems that can be solved by a relative generic GridSearch or GridSearchCV.
Therefore, we provide Optuna equivalents for these usecases to make use of the advanced samplers optuna provides.

.. note:: We still recommend to read through the :ref:`custom optuna example <custom_optuna_optimizer>` before using
          the specific implementations demonstrated here.
"""

# %%
# OptunaSearch - GridSearch on Steroids
# +++++++++++++++++++++++++++++++++++++
# The `OptunaSearch` class can be used in all cases where you would use :class:`~tpcp.optimize.GridSearch`.
# The following is equivalent to the GridSearch example (:ref:`grid_search`).

from pathlib import Path

import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector
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
# Optuna Study
# ------------
# To use optuna we need to create an optuna study, or rather a function that returns one, that can be used by
# `OptunaSearch` to create it.
# We will set this up identical to the :ref:`custom optuna example <custom_optuna_optimizer>`.
#
# .. note:: We use a in-memory study here, if you want to use multiprocessing or ensure that your search can be
#           continued, use a different study backend.
from optuna import Trial, samplers


def get_study_params(seed):
    # We use a simple RandomSampler, but every optuna sampler will work
    sampler = samplers.RandomSampler(seed=seed)
    return {"direction": "maximize", "sampler": sampler}


# %%
# Search Space
# ------------
# In contrast to `GridSearch` where we define a fix parameter grid, in optuna we define a search space.
# Which value sin this search space will actually be evaluated depends on the chosen sampler.
# This also needs to be a function that takes the current trial object as input.
def create_search_space(trial: Trial):
    trial.suggest_float("algorithm__min_r_peak_height_over_baseline", 0.1, 2, step=0.1)
    trial.suggest_float("algorithm__high_pass_filter_cutoff_hz", 0.1, 2, step=0.1)


# %%
# Score
# -----
# We use the same scoring function as in the `GridSearch` example:
from examples.algorithms.algorithms_qrs_detection_final import match_events_with_reference, precision_recall_f1_score


def score(pipeline: MyPipeline, datapoint: ECGExampleData):
    # We use the `safe_run` wrapper instead of just run. This is always a good idea.
    # We don't need to clone the pipeline here, as OptunaSearch will already clone the pipeline internally.
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
# Running the search
# ------------------
# Now we can run the search.
# Note, that because our scoring function returns a dictionary, we need to specify the key we want to optimize by
# passing it to `score_name`.
# In this case, we want to maximize the f1 score.
from tpcp.optimize.optuna import OptunaSearch

opti = OptunaSearch(
    pipe, get_study_params, create_search_space, scoring=score, n_trials=10, score_name="f1_score", random_seed=42
)
opti = opti.optimize(example_data)

# %%
# Inspecting the results
# ----------------------
# The results are very similar to the output of `GridSearch`.
# Besides the main results, we provide the results for each single datapoint and the respective grouplabel for the
# datapoints.

results = pd.DataFrame(opti.search_results_)
results

# %%
# We can also get the best para combi and an instance of the pipeline initialized with the best parameter
# combination.
print("Best Para Combi:", opti.best_params_)
print("Best score:", opti.best_score_)
print("Paras of optimized Pipeline:", opti.optimized_pipeline_.get_params())
