r"""
.. _complex_clinical_evaluation:

A complete hospital-to-home evaluation
======================================

Do we need labeled home data to build an algorithm that performs well at home?
This example compares hospital-only development with development using labeled
home recordings. Both approaches are evaluated on the same home recordings
from unseen participants. Hospital-only development uses no home labels for
training or hyperparameter selection.

We build the evaluation from the inside out: run two dummy algorithms, fit the
learnable algorithm through a trainer, search hyperparameters, evaluate a single
participant-wise train/test split, and finally use outer cross-validation.
Dummy sensor recordings keep the example focused on the evaluation structure.
Their results cannot answer the clinical research question.
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
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from tpcp import (
    Algorithm,
    Dataset,
    OptimizableParameter,
    OptimizablePipeline,
    Parameter,
    Pipeline,
)
from tpcp.optimize import GridSearch, GridSearchCV, Optimize
from tpcp.validate import (
    BaseDatasetSplitter,
    DatasetSplitter,
    Scorer,
    cross_validate,
    validate,
)


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
# The first algorithm learns a threshold from annotated training recordings
# and has a tunable smoothing window. The second has no training step, but its
# fixed threshold can still be selected by hyperparameter search.
# Learned values are constructor parameters so tpcp cloning preserves them.
# At inference time, both algorithms receive only sensor data.


class DummySmoothedDetector(Algorithm):
    """Dummy detector with a tunable smoothing window."""

    _action_methods = "detect"
    state_means: OptimizableParameter[np.ndarray | None]
    states_: np.ndarray
    window_size: Parameter[int]

    def __init__(
        self, window_size: int = 5, state_means: np.ndarray | None = None
    ):
        self.window_size = window_size
        self.state_means = state_means

    def detect(self, data: pd.DataFrame) -> Self:
        """Threshold a smoothed signal."""
        signal = (
            data.mean(axis=1)
            .rolling(self.window_size, center=True, min_periods=1)
            .mean()
        )
        self.states_ = (signal.to_numpy() > self.state_means.mean()).astype(int)
        return self


class DummyThresholdDetector(Algorithm):
    """Non-learnable detector with a tunable threshold."""

    _action_methods = "detect"
    states_: np.ndarray

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def detect(self, data: pd.DataFrame) -> Self:
        """Threshold the mean across sensor channels."""
        self.states_ = (data.mean(axis=1).to_numpy() > self.threshold).astype(
            int
        )
        return self


# %%
# One pipeline and scorer for both detectors
# ------------------------------------------
# Both detectors use the same prediction pipeline. A trainer subclass adds
# a training step that changes only ``state_means``. Hyperparameters remain
# fixed during each fit. Clone before training and inference to isolate runs.
#
# We score the event state per sample, then average recording scores. These are
# segmentation scores, not event matching scores. For real event detection,
# replace this scorer with event matching and a tolerance appropriate to the
# task.
# Every participant contributes three home days, so they have equal weight.


class EventPipeline(Pipeline[ClinicalDataset]):
    """Run either detector through the same prediction interface."""

    algorithm: Parameter[DummySmoothedDetector | DummyThresholdDetector]
    states_: np.ndarray

    def __init__(
        self, algorithm: DummySmoothedDetector | DummyThresholdDetector
    ):
        self.algorithm = algorithm

    def run(self, datapoint: ClinicalDataset) -> Self:
        """Predict one recording without reading reference annotations."""
        self.states_ = self.algorithm.clone().detect(datapoint.data).states_
        return self


class DummyDetectorTrainer(EventPipeline, OptimizablePipeline[ClinicalDataset]):
    """Fit a detector and reuse the normal prediction pipeline."""

    algorithm: Parameter[DummySmoothedDetector]
    algorithm__state_means: OptimizableParameter[np.ndarray | None]

    def self_optimize(self, dataset: ClinicalDataset, **_kwargs) -> Self:
        """Fit a cloned detector using the supplied training recordings."""
        signals = np.concatenate([d.data.to_numpy() for d in dataset])
        labels = np.concatenate([d.reference_states for d in dataset])
        state_means = np.stack(
            [signals[labels == state].mean(axis=0) for state in (0, 1)]
        )
        self.algorithm = self.algorithm.clone().set_params(
            state_means=state_means
        )
        return self


# %%
# Train and run on a few recordings
# ---------------------------------
# Start with the innermost operation. The trainer owns fitting, while the
# detector owns prediction and its learned parameters. After fitting, the
# detector can run independently of the trainer. The baseline runs directly.
# These recordings demonstrate the interfaces; this is not a performance
# estimate. The outer evaluations below start with fresh, untrained objects.

demo_train = dataset.get_subset(participant=[0, 1, 2, 3], source="hospital")
datapoint = demo_train[0]
baseline = DummyThresholdDetector().detect(datapoint.data)
baseline.states_[:20]

# %%
trainer = Optimize(DummyDetectorTrainer(DummySmoothedDetector())).optimize(
    demo_train
)
learned_detector = trainer.optimized_pipeline_.algorithm
learned_detector.state_means

# %%
learned_detector.detect(datapoint.data).states_[:20]

# %%
# Score predictions
# -----------------
# Both algorithms use the same prediction pipeline and scorer. Training and
# hyperparameter search do not belong in the scoring function.


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
# Search hyperparameters on training data
# ---------------------------------------
# Next, wrap the inner training operation in hyperparameter search. The
# learnable detector uses GridSearchCV with participant-wise inner folds.
# GridSearchCV fits each candidate on inner training participants, scores it on
# inner validation participants, and refits the winner on all supplied data.
# The baseline has no fit step, so GridSearch selects its threshold directly
# on the supplied recordings. Both return a pipeline with the same run API.

# All configuration lives here. The learnable algorithm uses a trainer and
# participant-wise inner CV. The non-learnable algorithm needs no fitting, but
# its threshold is still selected using labeled training data. Both optimizers
# expose the same interface to the outer evaluation.
searches = {
    "learnable": GridSearchCV(
        pipeline=DummyDetectorTrainer(DummySmoothedDetector()),
        parameter_grid=ParameterGrid({"algorithm__window_size": [1, 5, 11]}),
        scoring=Scorer(score, progress_bar=False),
        cv=DatasetSplitter(2, groupby="participant"),
        return_optimized="f1",
        progress_bar=False,
    ),
    "baseline": GridSearch(
        pipeline=EventPipeline(DummyThresholdDetector()),
        parameter_grid=ParameterGrid({"algorithm__threshold": [0.7, 0.9, 1.1]}),
        scoring=Scorer(score, progress_bar=False),
        return_optimized="f1",
        progress_bar=False,
    ),
}

demo_searches = {
    name: search.clone().optimize(demo_train)
    for name, search in searches.items()
}
pd.DataFrame(
    {name: search.best_params_ for name, search in demo_searches.items()}
)

# %%
# The selected pipeline can now predict a recording. These are training
# recordings, so this score is only an interface demonstration.

validate(
    demo_searches["learnable"].optimized_pipeline_,
    demo_train[:1],
    scoring=score,
    progress_bar=False,
)

# %%
# Select participants and recording sources in the splitter
# ---------------------------------------------------------
# Every inner and outer split groups by participant. Source selection is a
# separate step: training uses either hospital or home recordings, while
# testing always uses home recordings.
#
# DatasetSplitter handles the participant assignments. The small adapter below
# only filters each side by source afterwards. It implements the native
# BaseDatasetSplitter interface so cross_validate can handle the entire outer
# evaluation. The trainer and algorithms never select their own training data.


class RecordingSourceSplitter(BaseDatasetSplitter):
    """Filter the two sides of a participant-wise split by recording source."""

    def __init__(
        self,
        splitter: DatasetSplitter,
        train_source: str,
        test_source: str = "home",
    ):
        self.splitter = splitter
        self.train_source = train_source
        self.test_source = test_source

    def get_n_splits(self, dataset: ClinicalDataset) -> int:
        """Delegate the fold count to the participant splitter."""
        return self.splitter.get_n_splits(dataset)

    def split(self, dataset: ClinicalDataset, n_splits: int | None = None):
        """Select sources after assigning participants to folds."""
        for train, test in self.splitter.split(dataset, n_splits=n_splits):
            yield (
                dataset.get_subset(group_labels=train)
                .get_subset(source=self.train_source)
                .group_labels,
                dataset.get_subset(group_labels=test)
                .get_subset(source=self.test_source)
                .group_labels,
            )


# %%
# Evaluate one outer train/test split
# -----------------------------------
# A single grouped shuffle split holds out four participants. All experimental
# configurations reuse the same assignments, so their home test recordings are
# identical. Only the source of development labels changes between the two
# research conditions. Even the baseline's hyperparameter search must use only
# hospital labels in the hospital-only condition.

participant_holdout = DatasetSplitter(
    GroupShuffleSplit(n_splits=1, test_size=4, random_state=42),
    groupby="participant",
)
experiments = {
    "hospital_to_home_learnable": (
        searches["learnable"],
        RecordingSourceSplitter(
            participant_holdout.clone(), train_source="hospital"
        ),
    ),
    "home_to_home_learnable": (
        searches["learnable"],
        RecordingSourceSplitter(
            participant_holdout.clone(), train_source="home"
        ),
    ),
    "hospital_to_home_baseline": (
        searches["baseline"],
        RecordingSourceSplitter(
            participant_holdout.clone(), train_source="hospital"
        ),
    ),
    "home_to_home_baseline": (
        searches["baseline"],
        RecordingSourceSplitter(
            participant_holdout.clone(), train_source="home"
        ),
    ),
}

# Each entry contains its optimizer and splitter. Adding another method does
# not require changing any evaluation code or checking algorithm types.
outer_scorer = Scorer(score, progress_bar=False)
holdout_results = {
    name: cross_validate(
        search,
        dataset,
        scoring=outer_scorer,
        cv=splitter,
        return_optimizer=True,
        progress_bar=False,
    )
    for name, (search, splitter) in experiments.items()
}
pd.concat(
    {name: pd.DataFrame(result) for name, result in holdout_results.items()}
)[
    [
        "test__agg__f1",
        "test__agg__precision",
        "test__agg__recall",
        "test__data_labels",
    ]
]

# %%
# Extend the outer evaluation to cross-validation
# -----------------------------------------------
# Change only the participant splitter to three-fold grouped CV. Source
# selectors and optimizers stay the same. cross_validate clones each optimizer
# and runs the full search within each outer training fold. Every participant's
# home days are tested exactly once per experimental configuration.

cv_experiments = {
    name: (
        search,
        splitter.clone().set_params(
            splitter=DatasetSplitter(3, groupby="participant")
        ),
    )
    for name, (search, splitter) in experiments.items()
}
cv_results = {
    name: cross_validate(
        search,
        dataset,
        scoring=outer_scorer,
        cv=splitter,
        return_optimizer=True,
        progress_bar=False,
    )
    for name, (search, splitter) in cv_experiments.items()
}
cv_result_table = pd.concat(
    {name: pd.DataFrame(result) for name, result in cv_results.items()}
)
cv_result_table[
    [
        "test__agg__f1",
        "test__agg__precision",
        "test__agg__recall",
        "test__data_labels",
    ]
]

# %%
# tpcp retains recording scores, data labels, and fitted optimizers. These
# scores average recordings; every participant has three home days, so all
# participants have equal weight. The standard deviation across three folds
# below is descriptive, not a confidence interval. Keep the single-split and
# cross-validation results separate.

cv_result_table.groupby(level=0)["test__agg__f1"].agg(["mean", "std"])

# %%
# With real data, the paired held-out home results would let us assess whether
# home labels improve performance over hospital-only development. The dummy
# data cannot establish that conclusion. To fit a deployment detector, repeat
# its configured search on all development participants from the chosen source.
# Report held-out performance rather than its development score.
