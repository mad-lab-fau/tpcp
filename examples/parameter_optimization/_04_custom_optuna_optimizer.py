r"""
.. _custom_optuna_optimizer:

Custom Optuna Optimizer
=======================

The most popular method of (Hyper-)Parameter optimization is GridSearch (or GridSearchCV for optimizable pipelines).
These methods perform an exhaustive search of the parameter space by simply testing every option.
Considering, that training and testing an algorithm can be very costly, exhaustive gridsearch takes a long time and is
sometimes not feasible at all due to the required computational load.

For these cases various alternatives exist like :class:`~sklearn.model_selection.RandomizedSearchCV`,
:class:`~sklearn.model_selection.HalvingGridSearchCV` or advanced blackbox optimizer like
:class:`~optuna.samplers.TPESampler`.
`tpcp` does not implement all of these methods explicitly, as it would simply be too much work.
However, we try to make it relatively simply to bring such methods into the `tpcp` ecosystem by providing an interface
for `Optuna <https://optuna.readthedocs.io/en/stable/index.html>`_.
Optuna is a state-of-the-art hyperparameter optimization framework, that allows to implement any of the methods
mentioned above (and more) and allows to easily create custom samplers and pruners.

However, Optuna uses a (very elegant) functional interface that does not play well with the sklearn inspired
interface of `tpcp`.
Therefore, we provide the :class:`~tpcp.optimize.optuna.CustomOptunaOptimize` class which you can subclass to create
your own Optuna based optimizer.

.. note:: There is no need to create a custom subclass, if you only want to run the hyperparameter optimization and
          not nest the optimization into other `tpcp` method like :func:`~tpcp.validate.cross_validate`.
          For these cases you can simply use Optuna with its default interface and just call the respective `tpcp`
          methods in the objective function.

In this example, we are going to create an optimized GridSearch using custom pruning, that terminates trials early,
if we already realise at the first couple of participants, that the parameter combination will not work well.

Keep in mind, that this example should merely demonstrate the possibility to integrate Optuna with tpcp.
You are very much encouraged to read through the Optuna documentation and create your own project specific optimizers.
"""

# %%
# The Prerequisites
# -----------------
# First we need a dataset and a pipeline we want to optimize.
# For this example we are using the `QRSDetector` pipeline (the non-trainable version) and the `ECGExampleData` dataset.
# Checkout the other examples to learn more about them.
# We will simply copy the code over create an instance of both objects to be used later.
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple

import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector
from examples.datasets.datasets_final_ecg import ECGExampleData
from tpcp import Dataset, Parameter, Pipeline, cf

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path(".").resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"

# The dataset
example_data = ECGExampleData(data_path)


class MyPipeline(Pipeline):
    algorithm: Parameter[QRSDetector]

    r_peak_positions_: pd.Series

    def __init__(self, algorithm: QRSDetector = cf(QRSDetector())):
        self.algorithm = algorithm

    def run(self, datapoint: ECGExampleData):
        # Note: We need to clone the algorithm instance, to make sure we don't leak any data between runs.
        algo = self.algorithm.clone()
        algo.detect(datapoint.data, datapoint.sampling_rate_hz)

        self.r_peak_positions_ = algo.r_peak_positions_
        return self


# The pipeline
pipe = MyPipeline()

# %%
# What we want to do
# ------------------
# In the `Gridsearch Example <grid_search>`__, we already performed a gridsearch using the
# tpcp-:class:`~tpcp.optimize.GridSearch` class.
# Here we want to do something similar, but improve the GridSearch in two key aspects:
#
# 1. Instead of doing an exhaustive GridSearch, we use one of Optunas advanced samplers
# 2. When we encounter a parameter combination that doesn't work, we want to stop testing as early as possible to not
#    waist any time on bad para combis.
#
# We will start by implementing the first aspect than then make some modifications to enable the second.
#
# The first thing we need for any gridsearch is a score function that tells us how good our parameter combination works.
# In the `Gridsearch Example <grid_search>`__ we used a scorer that returned *accuracy*, *precision* and *f1-score*.
# We will use basically the same function here, but only return the f1-score, as this is the parameter we want to
# optimize.
# We could still calculate and return multiple other scores, but this would complicate the implementation of our
# Optimizer and hence, is kept as exercise for the reader ;) .

from examples.algorithms.algorithms_qrs_detection_final import match_events_with_reference


def f1_score(pipeline: MyPipeline, datapoint: ECGExampleData) -> float:
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
    f1_score = (2 * n_tp) / (len(pipeline.r_peak_positions_) + len(datapoint.r_peak_positions_))
    return f1_score


from optuna import Study, Trial

# %%
# The custom optimizer
# --------------------
# Optimizers in `tpcp` are nothing magically.
# There are simply algorithm that take a pipeline as input parameter and have an action method called `optimize` that
# takes in a dataset and then optimizes some parameters of the passed pipeline using this data.
#
# The :class:`~tpcp.optimize.optuna.CustomOptunaOptimize` class already implements most of that for us and simply
# requires us to implement our objective function for the optimization like we would need for Optuna anyway.
#
# Here, we define the objective function within the `create_objective` method of our custom optimizer and return it.
# The objective we define here is slightly different from the pure Optuna objective function, as it also takes a
# pipeline and a dataset as input in addition to the trial-object.
#
# The content of our objective function is very similar to our score function, but we do not expect just a single
# datapoint, but an entire dataset, and we need to handle getting and applying our parameters within the objective
# function.
#
# Because we define the function nested within another method, we have access to all class parameters.
# hence, if we want to add certain configuration to our objective, we can add parameters to the Optimizer itself and
# then access it in the objective function.
#
# For the Optimizer we want to build we primarily need two custom pieces of configuration:
#
# 1. The score function we want to use. We want to make that configurable and not hard-code "f1-score" into our
#    optimizer.
# 2. The search space for parameter search. In Optuna the search space is defined by calls to methods on the `trial`
#    object. Therefore, we take in a callable that gets the trial object passed and returns the selected parameters.
#    You will see how this works later on.
#
# With these two pieces of configuration in place our objective needs to simply do four things:
# First, we need to call the search space function to get the parameters.
# Then apply these parameters to our pipeline.
# For each of the datapoints within our test dataset, we need to calculate how good the pipeline with the new
# parameters works and then return the aggregated score.
#
# To avoid writing our own for-loop (for now) for the third step, we use :class:`~tpcp.validate.Scorer` with our
# custom score function.
# :class:`~tpcp.validate.Score` handles looping and aggregating results over multiple datapoints.
#
# With that our implementation looks as follows:
from tpcp.optimize.optuna import CustomOptunaOptimize
from tpcp.validate import Scorer


class OptunaSearch(CustomOptunaOptimize):
    create_search_space: Parameter[Callable[[Trial], Dict[str, Any]]]
    score_function: Parameter[Callable[[Pipeline, Dataset], float]]

    def __init__(
        self,
        pipeline: Pipeline,
        study: Optional[Study],
        create_search_space: Callable[[Trial], None],
        score_function: Callable[[Pipeline, Dataset], float],
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        return_optimized: bool = True,
    ) -> None:
        self.create_search_space = create_search_space
        self.score_function = score_function
        super().__init__(pipeline, study, n_trials=n_trials, timeout=timeout, return_optimized=return_optimized)

    def create_objective(self) -> Callable[[Trial, Pipeline, Dataset], Union[float, Sequence[float]]]:
        # Here we define our objective function

        def objective(trial: Trial, pipeline: Pipeline, dataset: Dataset) -> float:
            # First we need to select parameters for the current trial
            self.create_search_space(trial)
            # Then we apply these parameters to the pipeline
            pipeline = pipeline.set_params(**trial.params)

            # We wrap the score function with a scorer to avoid writing our own for-loop to aggregate the results.
            scorer = Scorer(self.score_function)

            # Calculate the results per datapoint
            # Note, that we could expose the `error_score` parameter on an optimizer level.
            # But let's keep it simple for now.
            average_score, single_scores = scorer(pipeline, dataset, error_score="raise")

            # As a bonus, we use the custom params option of optuna to store the individual scores per datapoint and the
            # respective datalabels
            trial.set_user_attr("single_scores", single_scores)
            trial.set_user_attr("data_labels", dataset.groups)
            return average_score

        return objective


# %%
# Running the optimization
# ------------------------
# To run the optimization, we need to create a new Optuna study, a custom sampler and the function that defines our
# search space:
from optuna import create_study, samplers

# We use a simple RandomSample, but every optuna sampler will work
sampler = samplers.RandomSampler(seed=42)
# We use a simple in memory study with the direction "maximize", as we want to optimize the highest f1-score
study = create_study(direction="maximize", sampler=sampler)

# %%
# The search space function requires a little more explanation:
# In optuna, we can use the `suggest_...` methods on a trial to get a new value within a given range.
# This uses our sampler in the background to suggest a new value that makes sense based on the trials that are already
# completed.
# The selected parameters are stored on the trial object so that we can access after the function was called.
#
# We use the names of the parameters we want to modify on our pipeline (using the `__` for nested values).
# This makes applying the parameters to the pipeline later on easy.


def create_search_space(trial: Trial):
    trial.suggest_float("algorithm__min_r_peak_height_over_baseline", 0.1, 2, step=0.1)
    trial.suggest_float("algorithm__high_pass_filter_cutoff_hz", 0.1, 2, step=0.1)


# %%
# Finally we are ready to run the pipeline.
# We create a new instance and set the stop criteria (in this case 10 random trials).
# Then we can use the familiar Optimize interface to run everything.
opti = OptunaSearch(
    pipe,
    study,
    create_search_space=create_search_space,
    score_function=f1_score,
    n_trials=10,
)

opti = opti.optimize(example_data)
print(f"The best performance was achieved with the parameters {opti.best_params_} and a f1-score of {opti.best_score_}")

# %%
# We can use `opti.search_results_` to get a full overview over all results.
# These parameters and parameter names are slightly modified compared to the normal Optuna output, to make it similar
# to the output of `GridSearch`.
pd.DataFrame(opti.search_results_)

# %%
# If you need even more inside, you can access the study object directly.
opti.study_

# %%
# And like with all Optimizers, we can access the `optimized_pipeline_` and call `run` directly on the optimizer.

opti.optimized_pipeline_
# %%
out_pipe = opti.run(example_data[0])
out_pipe.r_peak_positions_

# %%
# With this we created a simple random search optimizer (or Gridsearch, or whatever sampler we want to use) using
# Optuna.
# By using :class:`~tpcp.optimize.optuna.CustomOptunaOptimize` we get compatibility with the `tpcp` optimizer interface.
# This means, we could throw this optimizer into `cross_validate` and things would just work.
#
# A step further: Custom Pruning
# ------------------------------
# Simply to demonstrate the power of having full access to all Optuna features, we will implement a custom pruner
# that stops testing a trial, when one datapoint scores below a certain threshold.
# The idea is that when we iterate through he datapoints and process them one by one, and find one datapoint were
# the performance is really bad, we already know that this will not be our best choice/a choice we want to use.
# Hence, there is no need to compute scores for the remaining datapoints.
#
# In Optuna we can implement this using a custom Pruner.
# The pruner will be called everytime we report a new result and will tell us, if we should stop evaluating the trial.
# the *callback* feature of the :class:`~tpcp.validate.Scorer` class.
#
# .. note:: This a unusual usage of pruning. Usually pruning is used to stop after a certain number of training
#           epochs of an ML classifier and not to stop half-way through evaluating your dataset.
#           But it works and is practical.
#
# To create our custom pruner we can need a new class, sub-classing :class:`~optuna.pruners.BasePruner`.
# Then we implement a prune method that simply checks, if the current intermediate value is below a certain threshold.
# If yes we return True, telling optuna that the trial can be pruned.

from optuna.pruners import BasePruner
from optuna.study.study import Study
from optuna.trial import FrozenTrial


class MinDatapointPerformancePruner(BasePruner):
    def __init__(self, min_performance: float):
        self.min_performance = min_performance

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        step = trial.last_step

        if step is not None:
            score = trial.intermediate_values[step]
            if score < self.min_performance:
                return True
        return False


# %%
# Then we need to modify our optimizer to work with the pruner.
# We need to report each calculated value from each datapoint to Optuna as soon as it is calculated and not wait
# until we ran through the entire dataset.
# We can do that by passing a callback function to the `Scorer`.
# This callback will be called after each datapoint is evaluated and allows us to access the most recent score.
#
# We define this callback within the objective function to have access to the trial object of the outer scope.
# Using the trial object, we can report the most recent score to Optuna using `trial.report`.
# This will call the pruner and allows us to check afterwards, if the trial should be pruned.
# We then write some debug information and end the trial by raising a :class:`~optuna.TrialPruned` exception.

from optuna import TrialPruned


class OptunaSearchEarlyStopping(CustomOptunaOptimize):
    create_search_space: Parameter[Callable[[Trial], Dict[str, Any]]]
    score_function: Parameter[Callable[[Pipeline, Dataset], float]]

    def __init__(
        self,
        pipeline: Pipeline,
        study: Optional[Study],
        create_search_space: Callable[[Trial], None],
        score_function: Callable[[Pipeline, Dataset], float],
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        return_optimized: bool = True,
    ) -> None:
        self.create_search_space = create_search_space
        self.score_function = score_function
        super().__init__(pipeline, study, n_trials=n_trials, timeout=timeout, return_optimized=return_optimized)

    def create_objective(self) -> Callable[[Trial, Pipeline, Dataset], Union[float, Sequence[float]]]:
        def objective(trial: Trial, pipeline: Pipeline, dataset: Dataset) -> float:
            # First we need to select parameters for the current trial
            self.create_search_space(trial)
            # Then we apply these parameters to the pipeline
            pipeline = pipeline.set_params(**trial.params)

            def single_score_callback(*, step: int, dataset: Dataset, scores: Tuple[float], **_):
                # We need to report the new score value.
                # This will call the pruner internally and then tell us, if we should stop
                trial.report(float(scores[step]), step)
                if trial.should_prune():
                    # Apparently our last value was bad and we should abort.
                    # However, before we do so, we will save the scores so far as debug information
                    trial.set_user_attr("single_scores", scores)
                    trial.set_user_attr("data_labels", dataset[: step + 1].groups)
                    # and finally we abort the trial
                    raise TrialPruned(
                        f"Pruned at datapoint {step} ({dataset[step].groups[0]}) with value "
                        f"{scores[step]}."
                    )

            # We wrap the score function with a scorer to avoid writing our own for-loop to aggregate the results.
            # We pass our callback and `trial` which is passed as a generic kwarg to scorer and hence can be accessed
            # from within our callback.
            scorer = Scorer(self.score_function, single_score_callback=single_score_callback)

            # Calculate the results per datapoint
            # Note, that we could expose the `error_score` parameter on an optimizer level.
            # But let's keep it simple for now.
            average_score, single_scores = scorer(pipeline, dataset, error_score="raise")

            # As a bonus, we use the custom params option of optuna to store the individual scores per datapoint and the
            # respective datalabels
            trial.set_user_attr("single_scores", single_scores)
            trial.set_user_attr("data_labels", dataset.groups)
            return average_score

        return objective


# %%
# Running the new Optimizer stays the same (we even reuse the search space).
# We only need to add an instance of our pruner to the study.
sampler = samplers.RandomSampler(seed=42)
study = create_study(direction="maximize", sampler=sampler, pruner=MinDatapointPerformancePruner(0.3))

opti_early_stop = OptunaSearchEarlyStopping(
    pipe,
    study,
    create_search_space=create_search_space,
    score_function=f1_score,
    n_trials=10,
)

opti_early_stop.optimize(example_data)

# %%
# And then we can inspect the output.
# Compared to our previous run, we can see that many trials report NaN as score and "PRUNED" in the `state` column.
# For each of these values we saved some time.
# For the other trials, we get the same results as earlier.
pd.DataFrame(opti_early_stop.search_results_)

# %%
# Next steps
# ----------
# Building a custom optimizer is a little more involved than just using `GridSearch`, however it allows great
# flexibility with relatively small overhead compared to a pure implementation in Optuna.
#
# In this example we created a objective function that only makes sense for pipelines that don't have an internal
# optimization.
# However, instead of just a simple search, you could also create a CV based search by using
# :func:`~tpcp.validate.cross_validate` within your objective to split the passed data into multiple train test sets
# and optimize hyperparameters similar to :class:`~tpcp.optimize.GridSearchCV`.
