r"""
.. _participant_test_selection:

Evaluate one test type on held-out participants
===============================================

Each participant completes several tests. We want to train on all tests from
training participants, then evaluate only the ``Continuous long walking test``
from participants held out for testing. Other tests of held-out participants
are unused in that fold.

:class:`~tpcp.validate.DatasetSplitter` first assigns participants to folds,
then applies a selector to the test side. The training selector stays unset,
so training keeps every assigned recording, including the continuous walking
tests. See :ref:`per_side_subset_selection` for the selection contract.
"""

# %%
# One datapoint per test recording
# --------------------------------
# This index-only dataset is enough to inspect the protocol. Each participant
# has three test recordings, and each row is a separate Dataset datapoint.

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
dataset.index

# %%
# Assign participants, then select the evaluation test
# ----------------------------------------------------
# ``groupby="participant"`` keeps every recording of a participant on the
# same side of a fold. It does not merge recordings into a single datapoint.
# The selector receives only recordings of that fold's held-out participants.
# Leaving ``train_selector`` at its default ``None`` retains all recordings
# of the training participants.


def select_continuous_walking(ds: Dataset) -> Dataset:
    """Keep continuous walking tests from the supplied fold side."""
    return ds.get_subset(test="Continuous long walking test")


cv = DatasetSplitter(
    base_splitter=3,
    groupby="participant",
    test_selector=select_continuous_walking,
)
folds = list(cv.split(dataset))

# %%
# Inspect the first fold
# -----------------------
# Native tpcp splitters yield group labels. Use those labels to select the
# corresponding Dataset subsets. All three test types appear in training.

train_labels, test_labels = folds[0]
train = dataset.get_subset(group_labels=train_labels)
test = dataset.get_subset(group_labels=test_labels)
train.index

# %%
# Only continuous walking recordings appear in evaluation. No recording from
# either held-out participant appears in training, including their other tests.

test.index

# %%
# Summarize every fold
# ---------------------
# Each fold trains on twelve recordings from four participants and evaluates
# two recordings from two different participants. The participant lists show
# the separation; every participant is held out once across the three folds.

rows = []
for fold, (train_labels, test_labels) in enumerate(folds, start=1):
    train = dataset.get_subset(group_labels=train_labels)
    test = dataset.get_subset(group_labels=test_labels)
    rows.append(
        {
            "fold": fold,
            "training participants": train.index["participant"]
            .unique()
            .tolist(),
            "test participants": test.index["participant"].unique().tolist(),
            "training recordings": len(train),
            "training continuous walking tests": len(
                select_continuous_walking(train)
            ),
            "evaluation recordings": len(test),
        }
    )

pd.DataFrame(rows).set_index("fold")

# %%
# Use this ``cv`` object directly with :func:`~tpcp.validate.cross_validate` or
# :class:`~tpcp.optimize.GridSearchCV`. The splitter defines which recordings
# belong to the experiment; pipelines and optimizers need no test-type logic.
# Keep the dataset at recording level. Calling
# ``dataset.groupby("participant")`` would merge a participant's tests into
# one datapoint. Group labels select whole datapoints, so that representation
# cannot select just one test within it.
