r"""
.. _complex_clinical_evaluation:

A complete hospital-to-home evaluation
======================================

This example implements the study outlined in the README using generated
sensor streams and toy algorithms.

We compare hospital-to-home and home-to-home generalization to **unseen
participants**. We show both a single participant train/test split and
three-fold outer cross-validation for final evaluation.
Within each outer training set, a two-fold participant-wise grid search selects
hyperparameters using only the training source. The winning model is then refit
on all outer-training recordings from that source and tested on home recordings
of the held-out participants.

The supervised two-state Gaussian HMM and the independent threshold detector
below are teaching stand-ins, not implementations of the paper's algorithm or a
state-of-the-art method. Replace them with real detectors while keeping the same
dataset, pipeline, scoring function, and evaluation loop.
"""  # noqa: D400

# %%
# Recordings, metadata, and annotations
# -------------------------------------
# One index row describes a recording, not a sensor sample. Each participant has
# one hospital visit and three home days. Properties load the corresponding
# time series, global metadata, and daily annotations on demand. Real datasets
# would read files here. The signal has three channels and variable lengths;
# the index remains a small table describing these non-tabular data.

from typing import Self

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ParameterGrid,
    train_test_split,
)
from tpcp import (
    Algorithm,
    Dataset,
    OptimizableParameter,
    OptimizablePipeline,
    Parameter,
)
from tpcp.optimize import GridSearchCV
from tpcp.validate import DatasetSplitter, Scorer, validate


class ClinicalDataset(Dataset):
    """Deterministic synthetic recordings for twelve participants."""

    def create_index(self) -> pd.DataFrame:
        """List recordings without loading sensor samples."""
        return pd.DataFrame(
            [
                (p, source, day)
                for p in range(12)
                for source, day in [
                    ("hospital", 0),
                    ("home", 1),
                    ("home", 2),
                    ("home", 3),
                ]
            ],
            columns=["participant", "source", "day"],
        )

    @property
    def metadata(self) -> dict:
        """Global metadata shared by all recordings of a participant."""
        self.assert_is_single(["participant"], "metadata")
        participant = int(self.index["participant"].iloc[0])
        return {
            "age": 45 + participant,
            "diagnosis": "case" if participant % 2 else "control",
        }

    @property
    def daily_annotations(self) -> dict:
        """Recording annotations, separate from participant metadata."""
        self.assert_is_single(None, "daily_annotations")
        return {
            "reported_activity": "clinic test"
            if self.group_label.source == "hospital"
            else "daily living"
        }

    def _generate_recording(self) -> tuple[np.ndarray, np.ndarray]:
        self.assert_is_single(None, "recording")
        participant, source, day = self.group_label
        rng = np.random.default_rng(100 * participant + day)
        labels = np.zeros(160 + 20 * day, dtype=int)
        labels[30:55] = 1
        labels[100:130] = 1
        # Home data have a domain shift and more noise. Each participant has a
        # small personal offset shared by all their recordings.
        offset = participant * 0.025 + (0.25 if source == "home" else 0)
        noise = 0.8 if source == "home" else 0.5
        signal = offset + labels[:, None] * np.array([1.6, 1.2, 1.8])
        signal = signal + rng.normal(0, noise, size=(len(labels), 3))
        return signal, labels

    @property
    def data(self) -> pd.DataFrame:
        """Continuous three-channel data, sampled at 20 Hz."""
        signal, _ = self._generate_recording()
        return pd.DataFrame(
            signal, columns=["acc_x", "acc_y", "acc_z"]
        ).rename_axis("sample")

    @property
    def reference_states(self) -> np.ndarray:
        """Per-sample reference annotation: background=0, event=1."""
        return self._generate_recording()[1]


dataset = ClinicalDataset()
dataset.index.head()

# %%
dataset[0].metadata

# %%
dataset[0].daily_annotations

# %%
# Two interchangeable dummy algorithms
# ------------------------------------
# The HMM learns a mean sensor vector for each state from annotated training
# sequences. Viterbi decoding penalizes state changes, favoring continuous
# events. It assumes unit emission variance and equal initial state
# probabilities.
# ``transition_cost`` is the log ratio of staying versus switching
# probabilities.
# Sequences remain separate so there are no artificial transitions between days.
#
# Learned values must be constructor parameters so tpcp cloning preserves them.
# The threshold baseline uses the same training interface but predicts each
# sample
# independently. Its tunable offset moves the decision boundary. Both algorithms
# receive sensor data only at inference time, never reference annotations.


class DummyStateDetector(Algorithm):
    """Common supervised training interface for the two toy detectors."""

    _action_methods = "detect"
    state_means: OptimizableParameter[np.ndarray | None]
    states_: np.ndarray

    def __init__(self, state_means: np.ndarray | None = None):
        self.state_means = state_means

    def self_optimize(self, dataset: ClinicalDataset) -> Self:
        """Estimate state means using only the supplied training recordings."""
        signals = np.concatenate([d.data.to_numpy() for d in dataset])
        labels = np.concatenate([d.reference_states for d in dataset])
        self.state_means = np.stack(
            [signals[labels == state].mean(axis=0) for state in (0, 1)]
        )
        return self


class DummyHmmDetector(DummyStateDetector):
    """Small supervised Gaussian HMM with fixed transition probabilities."""

    transition_cost: Parameter[float]

    def __init__(
        self,
        transition_cost: float = 2.0,
        state_means: np.ndarray | None = None,
    ):
        self.transition_cost = transition_cost
        self.state_means = state_means

    def detect(self, data: pd.DataFrame) -> Self:
        """Decode the state sequence using Viterbi recursion."""
        emission_cost = 0.5 * (
            (data.to_numpy()[:, None, :] - self.state_means) ** 2
        ).sum(axis=2)
        costs = emission_cost[0]
        previous = np.zeros((len(data), 2), dtype=int)
        transition = np.array(
            [[0, self.transition_cost], [self.transition_cost, 0]]
        )
        for sample in range(1, len(data)):
            candidates = costs[:, None] + transition
            previous[sample] = candidates.argmin(axis=0)
            costs = candidates.min(axis=0) + emission_cost[sample]
        states = np.zeros(len(data), dtype=int)
        states[-1] = costs.argmin()
        for sample in range(len(data) - 1, 0, -1):
            states[sample - 1] = previous[sample, states[sample]]
        self.states_ = states
        return self


class DummyThresholdDetector(DummyStateDetector):
    """Independent sample classifier standing in for a comparison algorithm."""

    def __init__(
        self, offset: float = 0.0, state_means: np.ndarray | None = None
    ):
        self.offset = offset
        self.state_means = state_means

    def detect(self, data: pd.DataFrame) -> Self:
        """Threshold the mean across sensor channels."""
        threshold = self.state_means.mean() + self.offset
        self.states_ = (data.mean(axis=1).to_numpy() > threshold).astype(int)
        return self


# %%
# One pipeline and scorer for both detectors
# ------------------------------------------
# Training changes only the nested ``state_means`` parameter. Hyperparameters
# remain fixed during each fit. Clone before training or inference so learned
# values and action results do not leak between runs.
#
# We score the event state per sample, then average recording scores. These are
# segmentation scores, not event matching scores. For real event detection,
# replace this scorer with event matching and a tolerance appropriate to the
# task.
# Every participant has three home days, so the final mean weights participants
# equally as well. Unequal recording counts would require explicit aggregation.


class EventPipeline(OptimizablePipeline[ClinicalDataset]):
    """Adapt either detector to tpcp training and evaluation."""

    algorithm: Parameter[DummyStateDetector]
    algorithm__state_means: OptimizableParameter[np.ndarray | None]
    states_: np.ndarray

    def __init__(self, algorithm: DummyStateDetector):
        self.algorithm = algorithm

    def self_optimize(self, dataset: ClinicalDataset, **_kwargs) -> Self:
        """Fit the detector on the supplied training recordings."""
        self.algorithm = self.algorithm.clone().self_optimize(dataset)
        return self

    def run(self, datapoint: ClinicalDataset) -> Self:
        """Predict one recording without reading reference annotations."""
        self.states_ = self.algorithm.clone().detect(datapoint.data).states_
        return self


def score(
    pipeline: EventPipeline, datapoint: ClinicalDataset
) -> dict[str, float]:
    """Compare predicted and annotated states for a single recording."""
    prediction = pipeline.safe_run(datapoint).states_
    reference = datapoint.reference_states
    return {
        "f1": f1_score(reference, prediction, zero_division=0),
        "precision": precision_score(reference, prediction, zero_division=0),
        "recall": recall_score(reference, prediction, zero_division=0),
    }


# %%
# Two outer evaluation strategies
# -------------------------------
# Make outer splits on a participant-grouped view, then select
# recordings
# from those participants in the original dataset. All algorithms and training
# sources reuse these exact folds. Hospital recordings of a held-out participant
# are excluded too, even though we only score their home recordings.
#
# The inner search uses GroupKFold on participant IDs. Splitting individual days
# at random would allow recordings from one participant into both training and
# validation. For hospital-to-home we tune on hospital validation recordings;
# for home-to-home we tune on home validation recordings. Tuning on held-out
# home recordings would contaminate the final estimate.

participants = dataset.groupby("participant")
# The simple split holds out four of the twelve participants once. The CV
# strategy holds out four participants per fold and tests everyone once.
holdout_train, holdout_test = train_test_split(
    np.arange(len(participants)), test_size=4, random_state=42
)
outer_splits = {
    "train_test": [(holdout_train, holdout_test)],
    "cross_validation": list(
        KFold(n_splits=3, shuffle=True, random_state=42).split(participants)
    ),
}
algorithms = {
    "dummy_hmm": (
        DummyHmmDetector(),
        {"algorithm__transition_cost": [0.5, 2.0, 5.0]},
    ),
    "dummy_baseline": (
        DummyThresholdDetector(),
        {"algorithm__offset": [-0.2, 0.0, 0.2]},
    ),
}
fold_rows = []
recording_rows = []
# Keep searches and datasets available for inspecting parameters and folds.
evaluations = []
for evaluation_name, folds in outer_splits.items():
    for fold, (train_positions, test_positions) in enumerate(folds):
        train_ids = (
            participants[train_positions].index["participant"].unique().tolist()
        )
        test_ids = (
            participants[test_positions].index["participant"].unique().tolist()
        )
        test = dataset.get_subset(participant=test_ids, source="home")
        for source in ("hospital", "home"):
            train = dataset.get_subset(participant=train_ids, source=source)
            for name, (algorithm, grid) in algorithms.items():
                search = GridSearchCV(
                    pipeline=EventPipeline(algorithm.clone()),
                    parameter_grid=ParameterGrid(grid),
                    scoring=Scorer(score, progress_bar=False),
                    cv=DatasetSplitter(
                        GroupKFold(n_splits=2), groupby="participant"
                    ),
                    return_optimized="f1",
                    progress_bar=False,
                ).optimize(train)
                # Refit the selected configuration on all of `train`.
                # validate only runs inference; it never fits on `test`.
                result = validate(
                    search.optimized_pipeline_,
                    test,
                    scoring=score,
                    progress_bar=False,
                )
                fold_rows.append(
                    {
                        "evaluation": evaluation_name,
                        "fold": fold,
                        "train_source": source,
                        "algorithm": name,
                        "f1": result["agg__f1"][0],
                        "best_params": search.best_params_,
                    }
                )
                for label, f1 in zip(
                    result["data_labels"][0],
                    result["single__f1"][0],
                    strict=True,
                ):
                    recording_rows.append(
                        {
                            "evaluation": evaluation_name,
                            "fold": fold,
                            "train_source": source,
                            "algorithm": name,
                            "participant": label.participant,
                            "day": label.day,
                            "f1": f1,
                        }
                    )
                evaluations.append(
                    {
                        "evaluation": evaluation_name,
                        "train": train,
                        "test": test,
                        "search": search,
                    }
                )

# %%
# Inspect the results
# -------------------
# There are sixteen fitted searches: four for the single split and twelve for
# outer cross-validation, covering two training sources and two detectors.
# Each home recording is tested once per cross-validation comparison. The single
# split evaluates only the four held-out participants. These two strategies are
# alternative evaluation designs; do not pool their results.
# ``fold_results`` shows the selected hyperparameters and held-out F1 per fold;
# ``recording_results`` retains individual recording scores for further
# analysis.
# The table groups by evaluation strategy. The single split has no across-fold
# standard deviation, so its std is NaN. For cross-validation, the standard
# deviation across three folds is descriptive, not a confidence interval.

fold_results = pd.DataFrame(fold_rows)
recording_results = pd.DataFrame(recording_rows)
summary = fold_results.groupby(["evaluation", "train_source", "algorithm"])[
    "f1"
].agg(["mean", "std"])
fold_results

# %%
recording_results

# %%
summary

# %%
# These scores describe generated data only. For a real study, replace the data
# properties and dummy detectors, specify the intended deployment population,
# and choose appropriate event metrics before interpreting the comparison.
# To train a deployment model afterwards, rerun the inner search on all
# participants from the chosen source. Its training score must not replace
# the held-out performance reported here.
