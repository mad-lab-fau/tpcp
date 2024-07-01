"""
Advanced cross-validation
-------------------------
In many real world datasets, a normal k-fold cross-validation might not be ideal, as it assumes that each data point is
fully independent of each other.
This is often not the case, as our dataset might contain multiple data points from the same participant.
Furthermore, we might have multiple "stratification" variables that we want to keep balanced across the folds.
For example, different clinical conditions or different measurement devices.

This two concepts of "grouping" and "stratification" are sometimes complicated to understand and certain (even though
common) cases are not supported by the standard sklearn cross-validation splitters, without "abusing" the API.
For this reason, we create dedicated support for this in tpcp to tackle these cases with a little more confidence.
"""
# %%
# Let's start by re-creating the simple example from the normal cross-validation example.
#
# Dataset
# +++++++
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
# ++++++++
import pandas as pd

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector
from tpcp import Parameter, Pipeline, cf


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


# %%
# The Scorer
# ++++++++++
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
# Stratifcation
# +++++++++++++
# With this setup done, we can have a closer look at the dataset.
example_data

# %%
# The index has two columns, one indicating the participant group and one indicating the participant id.
# In this simple example, all groups appear the same amount of times and the index is ordered in a way that
# each fold will likely get a balanced amount of participants from each group.
#
# To show the impact of grouping and stratification, we take a subset of the data, that removes some participants from
# "group_1" to create an imbalance.
data_imbalanced = example_data.get_subset(index=example_data.index.query("participant not in ['114', '121']"))

# %%
# Running a simple cross-validation with 2 folds, will have all group-1 participants in the test data of the first fold:
#
# Note, that we skip optimization of the pipeline, to keep the example simple and fast.
from sklearn.model_selection import KFold

from tpcp.optimize import DummyOptimize
from tpcp.validate import cross_validate

cv = KFold(n_splits=2)

pipe = MyPipeline()
optimizable_pipe = DummyOptimize(pipe)

results = cross_validate(optimizable_pipe, data_imbalanced, scoring=score, cv=cv)
result_df = pd.DataFrame(results)

# %%
# We can see that the test data of the first fold contains only participants from group 1.
result_df["test_data_labels"].explode()

# %%
# This works fine when the groups are just "additional information", and are unlikely to affect the data within.
# For example, if the groups just reflect in which hospital the data was collected.
# However, when the group reflect information that is likely to affect the data (e.g. a relevant medical indication),
# we need to make sure that the actual group probabilities are remain the same in all folds.
# This can be done through stratification.
#
# .. note:: It is important to understand that "stratification" is not "balancing" the groups.
#           Group balancing should never be done during data splitting, as it will change the data distribution in your
#           test set, which will no longer reflect the real-world distribution.
#
# To stratify by the "patient group" we can use the `TpcpSplitter` class.
# We will provide it with a base splitter that enables stratification (in this case a `StratifiedKFold` splitter) and
# the column(s) to stratify by.
from sklearn.model_selection import StratifiedKFold

from tpcp.validate import DatasetSplitter

cv = DatasetSplitter(base_splitter=StratifiedKFold(n_splits=2), stratify="patient_group")

results = cross_validate(optimizable_pipe, data_imbalanced, scoring=score, cv=cv)
result_df_stratified = pd.DataFrame(results)
result_df_stratified["test_data_labels"].explode()

# %%
# Now we can see that the groups are balanced in each fold and both folds get one of the remaining group 1 participants.
#
# Grouping
# ++++++++
# Where stratification ensures that the distribution of a specific column is the same in all folds, grouping ensures
# that all data of one group is always either in the train or the test set, but never split across it.
# This is useful, when we have data points that are somehow correlated and the existence of data points from the same
# group in both the train and the test set of the same fold could hence be considered a "leak".
#
# A typical example for this is when we have multiple data points from the same participant.
# In our case here, we will use the "patient_group" as grouping variable for demonstration purposes, as we don't have multiple
# data points per participant.
#
# Note, that we use the "non-subsampled" example data here.
from sklearn.model_selection import GroupKFold

cv = DatasetSplitter(base_splitter=GroupKFold(n_splits=2), groupby="patient_group")

results = cross_validate(optimizable_pipe, example_data, scoring=score, cv=cv)
result_df_grouped = pd.DataFrame(results)
result_df_grouped["test_data_labels"].explode()

# %%
# We can see that this forces the creation of unequal sice splits to ensure that the groups are kept together.
# This is important to keep in mind when using grouping, as it can lead to unequally sized test sets.
#
# Combining Grouping and Stratification
# +++++++++++++++++++++++++++++++++++++
# Of course, we can also combine grouping and stratification.
# A typical example would be to stratify by clinical condition and group by participant.
# This is also easily possible with the `TpcpSplitter` class by providing both arguments.
#
# For the dataset that we have here, this does of course not make much sense, so we are not going to show an example
# here.
