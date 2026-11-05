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
learnable algorithm through a generic pipeline, search parameters, evaluate one
participant-wise train/test split, and finally use outer cross-validation.
Dummy sensor recordings keep the example focused on the evaluation structure.
Their results cannot answer the clinical research question.

The tpcp approach is to give each layer one job. A :class:`~tpcp.Dataset`
provides recordings and labels. An :class:`~tpcp.Algorithm` processes a signal,
and a :class:`~tpcp.Pipeline` connects it to the dataset. Optimizers wrap
fitting and parameter selection. :func:`~tpcp.validate.cross_validate`
evaluates the whole procedure. These layers compose, so changing an algorithm
does not require rewriting the evaluation.

For background, see the :ref:`dataset and pipeline guide
<datasets_algorithms_pipelines>` and the :ref:`evaluation guide
<algorithm_evaluation>`.
"""  # noqa: D400

# %%
# Recordings, metadata, and annotations
# -------------------------------------
# Start with a :class:`~tpcp.Dataset`. Its index lists the available recordings;
# the recordings themselves can be arbitrary Python objects. Here each row
# identifies a participant, a source, and a day. Sensor arrays do not need to be
# flattened into a table to use tpcp's selection and evaluation tools.
#
# Properties provide data, participant metadata, and annotations on demand. In a
# real dataset they would load files. The generator below supplies three sensor
# channels, two annotated event periods, and noisier home recordings. See
# :ref:`custom_dataset_basics` for indexing, subsets, and grouping, and
# :ref:`custom_dataset_ecg` for a dataset backed by real recordings.

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
)
from tpcp.optimize import GridSearch, GridSearchCV, Optimize
from tpcp.validate import (
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
# Selecting one row returns another dataset object with the same properties.
# Access the sensor samples without changing the representation of the index.
dataset[0].data.head()

# %%
# Participant metadata and recording annotations remain separate properties.
dataset[0].metadata

# %%
dataset[0].daily_annotations

# %%
# Two interchangeable dummy algorithms
# ------------------------------------
# An :class:`~tpcp.Algorithm` keeps configuration in constructor parameters and
# writes results to attributes ending in ``_``. ``_action_methods`` declares the
# method that produces those results. tpcp supplies ``clone()``,
# ``get_params()``, and ``set_params()`` for composing and configuring objects.
# See :ref:`custom_algorithms_qrs_detection` for a complete algorithm example.
#
# We use two small detectors. One smooths the signal and uses a threshold fitted
# from labeled recordings. Its window size is a hyperparameter. The baseline
# uses a fixed threshold, which can still be selected by hyperparameter search.
# Neither detector needs to know where recordings or labels are stored.


class DummySmoothedDetector(Algorithm):
    """Dummy detector with a tunable smoothing window."""

    _action_methods = "detect"
    threshold: OptimizableParameter[float | None]
    states_: np.ndarray
    window_size: Parameter[int]

    def __init__(self, window_size: int = 5, threshold: float | None = None):
        self.window_size = window_size
        self.threshold = threshold

    def smooth(self, data: pd.DataFrame) -> pd.Series:
        """Apply the same preprocessing during fitting and prediction."""
        return (
            data.mean(axis=1)
            .rolling(self.window_size, center=True, min_periods=1)
            .mean()
        )

    def self_optimize(
        self, data: list[pd.DataFrame], labels: list[np.ndarray]
    ) -> Self:
        """Learn a threshold directly from labeled signals."""
        signals = np.concatenate(
            [self.smooth(recording).to_numpy() for recording in data]
        )
        reference = np.concatenate(labels)
        background_mean = signals[reference == 0].mean()
        event_mean = signals[reference == 1].mean()
        self.threshold = float((background_mean + event_mean) / 2)
        return self

    def detect(self, data: pd.DataFrame) -> Self:
        """Threshold a smoothed signal."""
        signal = self.smooth(data)
        self.states_ = (signal.to_numpy() > self.threshold).astype(int)
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
# Connect a detector to a dataset
# -------------------------------
# An :class:`~tpcp.OptimizablePipeline` connects dataset properties to algorithm
# inputs. ``run`` forwards sensor data to ``detect``. ``self_optimize`` extracts
# signals and labels, then forwards them to the algorithm's ``self_optimize``.
# The algorithm owns fitting; the pipeline contains no algorithm-specific logic.
# Reference labels are absent from the inference path.
#
# ``OptimizableParameter`` marks the complete algorithm as optimizable. The
# algorithm owns its learned parameters; the pipeline does not need to name
# them or change its annotations for a different detector. See
# :ref:`optimize_pipelines` and the :ref:`optimization guide <optimization>`
# for these conventions.
#
# Cloning isolates each fit and prediction while preserving fitted constructor
# parameters. See :ref:`general_concepts` for parameters, results, and cloning.


class EventPipeline(OptimizablePipeline[ClinicalDataset]):
    """Adapt prediction and optional algorithm fitting to a clinical dataset."""

    algorithm: OptimizableParameter[Algorithm]
    states_: np.ndarray

    def __init__(self, algorithm: Algorithm):
        self.algorithm = algorithm

    def run(self, datapoint: ClinicalDataset) -> Self:
        """Predict one recording without reading reference annotations."""
        self.states_ = self.algorithm.clone().detect(datapoint.data).states_
        return self

    def self_optimize(self, dataset: ClinicalDataset, **_kwargs) -> Self:
        """Forward the supplied signals and labels to the algorithm."""
        self.algorithm = self.algorithm.clone().self_optimize(
            data=[d.data for d in dataset],
            labels=[d.reference_states for d in dataset],
        )
        return self


# %%
# Train and run on a few recordings
# ---------------------------------
# Start with one hospital recording. The baseline can run immediately.
# :class:`~tpcp.optimize.Optimize` calls the pipeline's ``self_optimize`` on the
# supplied training dataset and exposes the result as ``optimized_pipeline_``.
# Extracting its ``algorithm`` gives a detector that can run on its own. These
# recordings demonstrate the interfaces; this is not a performance estimate. The
# outer evaluations below start with fresh, untrained objects.

demo_train = dataset.get_subset(participant=[0, 1, 2, 3], source="hospital")
datapoint = demo_train[0]
baseline = DummyThresholdDetector().detect(datapoint.data)
baseline.states_[:20]

# %%
fitted = Optimize(EventPipeline(DummySmoothedDetector())).optimize(demo_train)
learned_detector = fitted.optimized_pipeline_.algorithm
learned_detector.threshold

# %%
learned_detector.detect(datapoint.data).states_[:20]

# %%
# Score predictions
# -----------------
# A scoring function receives a pipeline and one recording. It runs prediction,
# then compares the result with the reference labels. Returning a dictionary
# lets tpcp retain several metrics while optimizing one of them.
#
# :class:`~tpcp.validate.Scorer` applies this function across a dataset and
# aggregates the results. We use per-sample F1, precision, and recall; a real
# event-detection study might use event matching instead. For scoring and custom
# aggregation, see :ref:`validation_example` and :ref:`custom_scorer`.


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
# In tpcp, both fitting and parameter search are optimization: each produces a
# configured pipeline. :class:`~tpcp.optimize.GridSearchCV` fits the pipeline on
# inner training participants, scores candidates on inner validation
# participants, and refits the winner on all supplied training recordings.
# :class:`~tpcp.validate.DatasetSplitter` keeps participants together when
# ``groupby="participant"`` is set. See :ref:`gridsearch_cv` for this inner
# loop.
#
# The baseline has no fitting step. :class:`~tpcp.optimize.GridSearch` can
# select its threshold directly on the development recordings; see
# :ref:`grid_search`. Its use of labels still counts as development and must
# stay outside testing. It bypasses ``self_optimize`` entirely. For evaluation
# without parameter search, :class:`~tpcp.optimize.DummyOptimize` serves the
# same no-fitting role. Calling ``self_optimize`` on a pipeline whose algorithm
# lacks that method would fail naturally; the pipeline does not special-case it.
#
# Each entry below carries its pipeline, search space, and scorer.
# The ``algorithm__...`` search keys address parameters on the nested detector.
# ``return_optimized="f1"`` ranks candidates by F1 and makes the selected
# pipeline available as ``optimized_pipeline_``. Evaluation code can treat both
# optimizers identically, without checking algorithm types.
searches = {
    "learnable": GridSearchCV(
        pipeline=EventPipeline(DummySmoothedDetector()),
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
# ``best_params_`` reports the selected configuration above.
# :func:`~tpcp.validate.validate` below scores the selected pipeline without
# fitting it. These are still development recordings, so this score is only an
# interface demonstration, not an estimate of generalization.

validate(
    demo_searches["learnable"].optimized_pipeline_,
    demo_train[:1],
    scoring=score,
    progress_bar=False,
)

# %%
# Select participants and recording sources in the splitter
# ---------------------------------------------------------
# :class:`~tpcp.validate.DatasetSplitter` first assigns recordings to folds,
# keeping each participant's recordings together. Its ``train_selector`` and
# ``test_selector`` then select recordings from their respective assigned sides.
# Selecting a source cannot introduce a held-out participant into training.
#
# These selectors define the experiment. Neither pipeline nor algorithm needs
# to know whether we are developing on hospital or home recordings. For an
# example that retains all training tests and selects only one evaluation test,
# see :ref:`participant_test_selection`.


def hospital_recordings(data: ClinicalDataset) -> ClinicalDataset:
    """Select hospital recordings from an already assigned fold side."""
    return data.get_subset(source="hospital")


def home_recordings(data: ClinicalDataset) -> ClinicalDataset:
    """Select home recordings from an already assigned fold side."""
    return data.get_subset(source="home")


# %%
# Evaluate one outer train/test split
# -----------------------------------
# :func:`~tpcp.validate.cross_validate` wraps the *whole optimizer*, not just
# the trained detector. It selects the outer training data, runs the search
# there, and scores the resulting pipeline on the outer test data. A splitter
# with one split gives a conventional train/test evaluation through that same
# interface. See :ref:`cross_validation` for the evaluation API.
#
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
        participant_holdout.clone().set_params(
            train_selector=hospital_recordings, test_selector=home_recordings
        ),
    ),
    "home_to_home_learnable": (
        searches["learnable"],
        participant_holdout.clone().set_params(
            train_selector=home_recordings, test_selector=home_recordings
        ),
    ),
    "hospital_to_home_baseline": (
        searches["baseline"],
        participant_holdout.clone().set_params(
            train_selector=hospital_recordings, test_selector=home_recordings
        ),
    ),
    "home_to_home_baseline": (
        searches["baseline"],
        participant_holdout.clone().set_params(
            train_selector=home_recordings, test_selector=home_recordings
        ),
    ),
}

# Each entry contains its optimizer and splitter. Adding another method does not
# require changing any evaluation code or checking algorithm types.
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
# Each row below is one outer split. ``test__agg__...`` contains the mean
# recording score, and ``test__data_labels`` identifies the tested recordings.
score_columns = [
    "test__agg__f1",
    "test__agg__precision",
    "test__agg__recall",
    "test__data_labels",
]
holdout_table = pd.concat(
    {name: pd.DataFrame(result) for name, result in holdout_results.items()}
)
holdout_table[score_columns]

# %%
# Extend the outer evaluation to cross-validation
# -----------------------------------------------
# Change only the participant splitter to three-fold grouped CV. tpcp's
# ``clone().set_params(...)`` creates a new configuration while preserving the
# single-split experiment. Updating ``base_splitter`` changes the fold strategy;
# participant grouping and source selectors stay in place.
# :func:`~tpcp.validate.cross_validate` clones each optimizer and runs the full
# search within each outer training fold. Every participant's home days are
# tested exactly once per experimental configuration.

cv_experiments = {
    name: (
        search,
        splitter.clone().set_params(base_splitter=3),
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
cv_result_table[score_columns]

# %%
# The result also retains ``test__single__f1`` for individual recording scores.
# With ``return_optimizer=True``, ``optimizer`` holds each fitted outer-fold
# search, including its ``best_params_`` and inner search results. These scores
# average recordings; every participant has three home days, so all participants
# have equal weight. The standard deviation across three folds below is
# descriptive, not a confidence interval. Keep the single-split and
# cross-validation results separate.

cv_result_table.groupby(level=0)["test__agg__f1"].agg(["mean", "std"])

# %%
# Inspect an individual fold's selected configuration without fitting again.
cv_results["hospital_to_home_learnable"]["optimizer"][0].best_params_

# %%
# With real data, the paired held-out home results would let us assess whether
# home labels improve performance over hospital-only development. The dummy data
# cannot establish that conclusion. To fit a deployment detector, repeat its
# configured search on all development participants from the chosen source.
# Report held-out performance rather than its development score.
