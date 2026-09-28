r"""
Adding a fixed subset to every training fold
===========================================

Some datapoints are useful for training but should never appear in testing. For
example, artificial edge cases can supplement the ordinary recordings used for
cross-validation. Here we split the normal datapoints with
:class:`~sklearn.model_selection.KFold` and add a fixed random subset of the
train-only datapoints to every training fold using
:class:`~tpcp.validate.CombinedSplitter` and :class:`~tpcp.validate.NoSplit`.
"""

# %%
# Normal and train-only datapoints
# --------------------------------
# This small dataset has twelve normal datapoints and six train-only datapoints.
# Each index row represents one datapoint. We only need the index to demonstrate
# splitting; no signal data or pipeline is required.

import pandas as pd
from sklearn.model_selection import KFold
from tpcp import Dataset
from tpcp.validate import CombinedSplitter, NoSplit


class ExampleDataset(Dataset):
    """Dataset with normal recordings and additional train-only recordings."""

    def create_index(self) -> pd.DataFrame:
        """Create a deterministic index with one row per datapoint."""
        return pd.DataFrame(
            [("normal", f"normal_{i:02d}") for i in range(12)]
            + [("train_only", f"extra_{i:02d}") for i in range(6)],
            columns=["kind", "recording"],
        )


dataset = ExampleDataset()
dataset.index

# %%
# Choose a fixed random subset for training
# -----------------------------------------
# The selector below receives only the train-only part of the dataset. It
# samples three of its six datapoints without replacement. The integer seed
# makes repeated calls select the same recordings for the same input.
# ``NoSplit`` evaluates this selector once when splitting and repeats the
# resulting assignment in every fold.


def select_training_extras(ds: Dataset) -> Dataset:
    """Select three reproducible training extras from the supplied dataset."""
    return ds.get_subset(index=ds.index.sample(n=3, random_state=42))


selected_extras = select_training_extras(dataset.get_subset(kind="train_only"))
selected_extras.index

# %%
# Combine the two splitting rules
# -------------------------------
# Each pair supplies a selector and a splitter. ``KFold`` sees only normal
# datapoints, so the train-only recordings cannot enter its test folds.
# ``NoSplit`` contributes the selected extras to training and an empty list to
# testing because its ``test`` selector is omitted. The remaining three
# train-only datapoints are unused.
#
# Both splitters must report the same number of folds. A raw sklearn splitter
# such as ``KFold`` can be passed directly; ``CombinedSplitter`` adapts its
# positional outputs to dataset group labels.

n_splits = 3
cv = CombinedSplitter(
    parts=[
        (
            lambda ds: ds.get_subset(kind="normal"),
            KFold(n_splits=n_splits, shuffle=True, random_state=0),
        ),
        (
            lambda ds: ds.get_subset(kind="train_only"),
            NoSplit(n_splits=n_splits, train=select_training_extras),
        ),
    ]
)

# %%
# Inspect the folds
# -----------------
# Native splitters yield group labels. Select the corresponding datasets with
# ``get_subset(group_labels=...)``. Each fold contains eight normal training
# datapoints plus the same three extras, and four normal test datapoints. Every
# normal datapoint appears in testing exactly once across the three folds.

folds = list(cv.split(dataset))
rows = []
for fold, (train_labels, test_labels) in enumerate(folds, start=1):
    train = dataset.get_subset(group_labels=train_labels)
    test = dataset.get_subset(group_labels=test_labels)
    rows.append(
        {
            "fold": fold,
            "normal training": train.get_subset(kind="normal")
            .index["recording"]
            .to_list(),
            "train-only extras": train.get_subset(kind="train_only")
            .index["recording"]
            .to_list(),
            "testing": test.index["recording"].to_list(),
        }
    )

fold_summary = pd.DataFrame(rows).set_index("fold")
fold_summary

# %%
# The ``train-only extras`` column stays identical while the normal train/test
# assignments change. Pass this same ``cv`` object to ``cross_validate`` or
# ``GridSearchCV`` when using a pipeline with this dataset.
