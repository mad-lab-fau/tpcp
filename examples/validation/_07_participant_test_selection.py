r"""
.. _participant_test_selection:

Advanced train-test splits
==========================

In previous examples we already showed how to control which datapoints appear
in training and evaluation and how to select data before it enters optimization.
These tools can already solve many problems, but sometimes we need an additional
selection step after the main splitter runs.
A typical example is when we have multiple recordings per participant and only
want to train or evaluate on a subset of these recordings.
Conceptually, we want to perform a grouped K-fold split on the participants and
then select recordings within the training and test sets of each fold.

In the concrete example below we have three recordings per participant, but we
are only interested in evaluating on the continuous long walking test, which
represents our deployment use case.
All recordings from the training participants can still add value to training,
including their continuous walking tests. The other recordings from held-out
participants are unused in that fold.

This example shows how to implement this using a custom selector function on
:class:`~tpcp.validate.DatasetSplitter`.
"""

# %%
# Dummy dataset
# -------------
# We create a dummy dataset with three recordings per participant and six
# participants. Each recording is a separate datapoint.

import pandas as pd
from tpcp import Dataset
from tpcp.validate import DatasetSplitter


class ParticipantTestDataset(Dataset):
    """Test recordings from six participants."""

    def create_index(self) -> pd.DataFrame:
        """List each participant's test recordings."""
        return pd.DataFrame(
            [
                (participant, test)
                for participant in range(1, 7)
                for test in (
                    "Short walking test",
                    "Continuous long walking test",
                    "Stair climbing test",
                )
            ],
            columns=["participant", "test"],
        )


dataset = ParticipantTestDataset()
dataset

# %%
# The group splitter
# ------------------
# The most important part is that we want to split the dataset by participant,
# so that no participant has recordings in both training and evaluation.
# This is usually done with a grouped K-fold splitter.
# :class:`~tpcp.validate.DatasetSplitter` adapts sklearn splitters to work with
# tpcp datasets and supports grouping directly by dataset columns.
# With ``base_splitter=3`` and ``groupby="participant"``, it uses sklearn's
# :class:`~sklearn.model_selection.GroupKFold` with three folds.
#
# All recordings from a participant stay on the same side of each fold, and two
# participants are held out for evaluation in each fold. The dataset still has
# one datapoint per recording.
grouped_splitter = DatasetSplitter(base_splitter=3, groupby="participant")

grouped_folds = list(grouped_splitter.split(dataset))
len(grouped_folds)

# %%
grouped_folds[0]

# %%
# Custom test selection
# ---------------------
# Now we want to expand this base splitter to keep only the continuous walking
# tests in the evaluation set. We can do this by providing a custom
# ``test_selector`` function to :class:`~tpcp.validate.DatasetSplitter`.
# We leave ``train_selector`` unset to keep all twelve recordings from the four
# training participants, including their continuous walking tests.
#
# Each selector receives only its own side of a specific fold: ``test_selector``
# receives the held-out dataset, while ``train_selector``, if provided, receives
# the training dataset. Each must return a dataset whose group labels are a
# subset of the labels assigned to that side.
#
# Here we select the continuous walking tests with ``get_subset``. The other
# tests of the held-out participants are unused; they do not move into training.
cv = DatasetSplitter(
    base_splitter=3,
    groupby="participant",
    test_selector=lambda ds: ds.get_subset(test="Continuous long walking test"),
)
folds = list(cv.split(dataset))

# %%
# Inspect the first fold
# ----------------------
# Native tpcp splitters yield group labels. Use those labels to select the
# corresponding Dataset subsets. All three test types appear in training.

train_labels, test_labels = folds[0]
train = dataset.get_subset(group_labels=train_labels)
test = dataset.get_subset(group_labels=test_labels)
train

# %%
# Only two continuous walking recordings appear in evaluation. No recording from
# either held-out participant appears in training, including their other tests.

test

# %%
# Use this ``cv`` object directly with :func:`~tpcp.validate.cross_validate` or
# :class:`~tpcp.optimize.GridSearchCV`. The splitter defines which recordings
# belong to the experiment; pipelines and optimizers need no test-type logic.
# Keep the dataset at recording level. Calling
# ``dataset.groupby("participant")`` would merge a participant's tests into
# one datapoint. Group labels select whole datapoints, so that representation
# cannot select just one test within it.
