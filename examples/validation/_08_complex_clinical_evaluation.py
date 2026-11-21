r"""
.. _complex_clinical_evaluation:

Comparing algorithms on complex clinical data
=============================================

The main reason tpcp exists is that research on real-world data often does not
fit the simple train/test split on tabular data assumed by sklearn.
Real-world data is often hierarchical, with multiple recordings per participant
and multiple participants per study. We might have time series, images,
tabular data, and participant metadata, all stored in different places. And we
can rarely expect complete data. Missing recordings and labels are common.

We also often need to compare very different algorithms, such as deep learning
models and rule-based methods. We want to use the same data, metrics, and
evaluation code for all of them to avoid errors.

Finally, our research questions might require unusual evaluation approaches.
What happens if we only train on a specific subset of the data? What if we train
on a participant's full history rather than their latest recordings?

In this example, we implement an algorithm comparison with these kinds of
requirements. It should give you a starting point for your own evaluations.

Here is the scenario we want to investigate. We have performed a movement study
using wearable sensors and want to detect gait events, such as heel strikes,
from the sensor data. We plan to deploy the algorithm in a home environment.
We have labeled home and hospital recordings from some participants, plus
hospital recordings from further participants who have no home data.

We want to understand whether hospital labels are sufficient to develop an
algorithm that performs well at home, or whether we also need actual labeled
home recordings.
"""  # noqa: D400

# %%
# The datasets
# ------------
# tpcp lets us give multiple datasets a shared interface.
# Their adapters handle the differences in storage. Dataset A contains hospital
# and home recordings from participants who have both. Dataset B contains only
# hospital recordings from a separate set of participants who have no home data.
#
# The two datasets come from different studies and use different storage
# formats. We write a separate adapter for each, based on a common
# :class:`~tpcp.Dataset` interface. The evaluation pipeline can then use either
# dataset without knowing how it loads its recordings.
# See :ref:`datasets_algorithms_pipelines` for more on these interfaces.
import numpy as np
import pandas as pd
from tpcp import Dataset


class BaseMovementStudyDataset(Dataset):
    """Base class for movement study datasets."""

    @property
    def data(self) -> pd.DataFrame:
        """Return the sensor data as a pandas DataFrame."""
        raise NotImplementedError

    @property
    def reference_events_(self) -> np.ndarray:
        """Return reference event sample indices as a NumPy array."""
        raise NotImplementedError

    @property
    def metadata(self) -> dict:
        """Return the participant metadata as a dictionary."""
        raise NotImplementedError


# %%
# Now we can implement the two adapters. For the loading logic, we pretend that
# A stores acceleration in g, while B stores it in m/s². The adapter for A
# converts its data to the same unit as B. A real study might need more involved
# preprocessing, but the approach is the same.
#
# We put this preprocessing in the loader, so we don't need a second copy of the
# data on disk that could get out of sync with the original recordings.
# Here, we generate sine waves with varying heights and some noise instead of
# reading files. The reference events are peaks in the signal before adding
# noise.
#
# We use the participant and recording IDs to seed the random number generator,
# so the data is reproducible.
from scipy.signal import find_peaks


def _make_dummy_recording(
    participant: int, recording_id: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a 100 Hz signal and reference events before adding noise."""
    recording_codes = {"short_test": 0, "long_test": 1, "day1": 2, "day2": 3}
    lengths = {
        "short_test": 1000,
        "long_test": 2000,
        "day1": 3000,
        "day2": 3000,
    }
    rng = np.random.default_rng(
        42 + 100 * participant + recording_codes[recording_id]
    )
    time = np.arange(lengths[recording_id]) / 100
    frequency = rng.uniform(0.9, 1.2)
    cycle_times = np.arange(int(np.ceil(time[-1] * frequency)) + 2) / frequency
    amplitude = np.interp(
        time, cycle_times, rng.uniform(0.6, 1.4, len(cycle_times))
    )
    phase = 2 * np.pi * frequency * time + rng.uniform(-0.2, 0.2)
    clean = amplitude[:, None] * np.sin(
        phase[:, None] + np.array([0.0, 0.1, -0.1])
    )
    clean *= np.array([1.0, 0.8, 1.2])
    events = find_peaks(clean.mean(axis=1))[0]
    noisy = clean + rng.normal(scale=0.08, size=clean.shape)
    return pd.DataFrame(noisy, columns=["acc_x", "acc_y", "acc_z"]), events


class DatasetA(BaseMovementStudyDataset):
    """Dataset A with hospital and home recordings."""

    def create_index(self) -> pd.DataFrame:
        """Create the index for dataset A."""
        return pd.DataFrame(
            [
                (participant, condition, recording_id)
                for participant in range(1, 7)
                for condition, recording_id in [
                    ("hospital", "short_test"),
                    ("hospital", "long_test"),
                    ("home", "day1"),
                    ("home", "day2"),
                ]
            ],
            columns=["participant", "condition", "recording_id"],
        )

    @staticmethod
    def _convert_acc_to_m_s2(data: pd.DataFrame) -> pd.DataFrame:
        """Convert the accelerometer data from g to m/s^2."""
        return data * 9.81

    @property
    def data(self) -> pd.DataFrame:
        """Return the sensor data as a pandas DataFrame."""
        # Imagine that dataset A stores these recordings in g.
        data_m_s2, _ = _make_dummy_recording(
            int(self.group_label.participant), self.group_label.recording_id
        )
        stored_data_g = data_m_s2 / 9.81
        return self._convert_acc_to_m_s2(stored_data_g)

    @property
    def reference_events_(self) -> np.ndarray:
        """Return event positions from the signal before adding noise."""
        return _make_dummy_recording(
            int(self.group_label.participant), self.group_label.recording_id
        )[1]

    @property
    def metadata(self) -> dict:
        """Return the participant metadata as a dictionary."""
        self.assert_is_single(["participant"], "metadata")
        participant = int(self.index["participant"].iloc[0])
        return {
            "age": 30 + participant,
            "gender": "male" if participant % 2 == 0 else "female",
            "diagnosis": "case" if participant % 2 == 0 else "control",
        }


DatasetA()

# %%
# Here is one of the recordings. The red points mark the reference events.
import matplotlib.pyplot as plt

recording = DatasetA()[0]
signal = recording.data.mean(axis=1)
time = np.arange(len(signal)) / 100  # The dummy recordings use 100 Hz.
reference_events = recording.reference_events_

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(time, signal, linewidth=1, label="Mean sensor signal")
ax.scatter(
    time[reference_events],
    signal.iloc[reference_events],
    color="tab:red",
    label="Reference events",
    zorder=3,
)
ax.set(xlabel="Time [s]", ylabel="Acceleration [m/s²]")
ax.legend(loc="lower right", ncol=2)
fig.tight_layout()


# %%
# Dataset B works much like A, but only has hospital recordings. Its data is
# already in m/s², so the loader does not need a unit conversion.


class DatasetB(BaseMovementStudyDataset):
    """Dataset B with hospital recordings only."""

    def create_index(self) -> pd.DataFrame:
        """Create the index for dataset B."""
        return pd.DataFrame(
            [
                (participant, condition, recording_id)
                for participant in range(7, 13)
                for condition, recording_id in [
                    ("hospital", "short_test"),
                    ("hospital", "long_test"),
                ]
            ],
            columns=["participant", "condition", "recording_id"],
        )

    @property
    def data(self) -> pd.DataFrame:
        """Return the sensor data as a pandas DataFrame."""
        # Dataset B already stores its recordings in m/s^2.
        return _make_dummy_recording(
            int(self.group_label.participant), self.group_label.recording_id
        )[0]

    @property
    def reference_events_(self) -> np.ndarray:
        """Return event positions from the signal before adding noise."""
        return _make_dummy_recording(
            int(self.group_label.participant), self.group_label.recording_id
        )[1]

    @property
    def metadata(self) -> dict:
        """Return the participant metadata as a dictionary."""
        self.assert_is_single(["participant"], "metadata")
        participant = int(self.index["participant"].iloc[0])
        return {
            "age": 30 + participant,
            "gender": "male" if participant % 2 == 0 else "female",
            "diagnosis": "case" if participant % 2 == 0 else "control",
        }


DatasetB()

# %%
# For the evaluation, we also want to use both datasets together. This wrapper
# combines their indexes and adds an origin column. When we request a recording,
# it delegates loading to the corresponding source dataset.
# The ``datasets`` field is a composite parameter: a list of named tpcp objects.
# This lets us inspect and configure the source datasets through the wrapper.
#
# You can adapt this recipe to other datasets as well. We recommend this
# approach for combining multiple datasets that share the same base class.


class CombinedStudyDataset(BaseMovementStudyDataset):
    """Delegate each recording to its source dataset."""

    _composite_params = ("datasets",)

    datasets: list[tuple[str, BaseMovementStudyDataset]]

    def __init__(self, datasets, *, groupby_cols=None, subset_index=None):
        self.datasets = datasets
        super().__init__(
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    def create_index(self):
        return pd.concat(
            [
                dataset.index.assign(origin=name)
                for name, dataset in self.datasets
            ],
            ignore_index=True,
        )

    @property
    def source_datapoint(self) -> BaseMovementStudyDataset:
        self.assert_is_single(None, "source_datapoint")

        source = dict(self.datasets)[self.group_label.origin]
        source_index = self.index.loc[:, source.index.columns]
        return source.get_subset(index=source_index)

    @property
    def data(self):
        return self.source_datapoint.data

    @property
    def metadata(self):
        return self.source_datapoint.metadata

    @property
    def reference_events_(self):
        return self.source_datapoint.reference_events_


CombinedStudyDataset(datasets=[("A", DatasetA()), ("B", DatasetB())])

# %%
# The algorithms
# --------------
# Next, we create two dummy algorithms: one learnable and one rule-based.
# You could replace these with a deep learning model and a more sophisticated
# rule-based method. Don't read too much into these particular implementations;
# the point is to demonstrate the evaluation workflow.
#
# As with the datasets, we use a common base class to make the algorithms'
# interface explicit. This lets us use them interchangeably in the pipeline.

from typing import Self

from tpcp import Algorithm, OptimizableParameter, PureParameter


class BaseEventDetector(Algorithm):
    """Common event-detection interface with optional fitting."""

    _action_methods = "detect"
    events_: np.ndarray

    def detect(self, data: pd.DataFrame) -> Self:
        """Detect events and store their sample indices in ``events_``."""
        raise NotImplementedError

    def self_optimize(
        self, data: list[pd.DataFrame], labels: list[np.ndarray]
    ) -> Self:
        """Fit the detector when supported by its implementation."""
        raise NotImplementedError("This detector does not support fitting.")


# %%
# The first algorithm averages the sensor channels and uses find_peaks to find
# peaks above a threshold. It has parameters we can optimize via GridSearch,
# but nothing to fit.
class PeakDetector(BaseEventDetector):
    """Simple peak detector with a tunable height threshold."""

    threshold: float
    min_distance_samples: int

    def __init__(self, threshold: float = 0.9, min_distance_samples: int = 50):
        self.threshold = threshold
        self.min_distance_samples = min_distance_samples

    def detect(self, data: pd.DataFrame) -> Self:
        """Find sufficiently high peaks in the mean sensor signal."""
        self.events_ = find_peaks(
            data.mean(axis=1).to_numpy(),
            height=self.threshold,
            distance=self.min_distance_samples,
        )[0]
        return self


# %%
# The second algorithm learns what the signal around an event looks like.
# We cut out a little window around each reference event and average these
# windows into a template.
# For new data, we look for places that match this template and pick peaks in
# the similarity signal.
# We normalize the windows so that changing the height of a step does not
# change its shape.
# Still a dummy algorithm. You could put a neural network here instead.
#
# The parameters now have different roles. The window radius affects training,
# while the correlation threshold and minimum peak distance only affect
# inference. We mark the inference-only parameters as
# :class:`~tpcp.PureParameter` and the learned template as
# :class:`~tpcp.OptimizableParameter`. The search can use these annotations to
# avoid repeating training when only inference parameters change.
# See :ref:`optimization` for more on parameter roles.
from numpy.lib.stride_tricks import sliding_window_view


class TemplateMatchingDetector(BaseEventDetector):
    """Learn an event template and find similar windows in new recordings."""

    min_correlation: PureParameter[float]
    min_distance_samples: PureParameter[int]
    template_radius: int
    template: OptimizableParameter[np.ndarray | None]

    def __init__(
        self,
        template_radius: int = 20,
        min_correlation: float = 0.7,
        min_distance_samples: int = 50,
        template: np.ndarray | None = None,
    ):
        self.template_radius = template_radius
        self.min_correlation = min_correlation
        self.min_distance_samples = min_distance_samples
        self.template = template

    @staticmethod
    def _normalize_windows(windows: np.ndarray) -> np.ndarray:
        centered = windows - windows.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return np.divide(
            centered, norms, out=np.zeros_like(centered), where=norms > 0
        )

    def self_optimize(
        self, data: list[pd.DataFrame], labels: list[np.ndarray]
    ) -> Self:
        """Average normalized windows around the reference event indices."""
        windows = []
        radius = self.template_radius
        for recording, events in zip(data, labels, strict=True):
            signal = recording.mean(axis=1).to_numpy()
            for event in events:
                # Only use complete windows; never join separate recordings.
                if radius <= event < len(signal) - radius:
                    windows.append(signal[event - radius : event + radius + 1])
        normalized = self._normalize_windows(np.stack(windows))
        self.template = self._normalize_windows(
            normalized.mean(axis=0, keepdims=True)
        )[0]
        return self

    def detect(self, data: pd.DataFrame) -> Self:
        """Match the learned template against windows in a new recording."""
        assert self.template is not None, "Fit the detector before detection."
        signal = data.mean(axis=1).to_numpy()
        windows = sliding_window_view(signal, len(self.template))
        similarity = self._normalize_windows(windows) @ self.template
        peaks = find_peaks(
            similarity,
            height=self.min_correlation,
            distance=self.min_distance_samples,
        )[0]
        # Correlation windows start before the event; shift to their center.
        self.events_ = peaks + len(self.template) // 2
        return self


# %%
# Connect a detector to a dataset
# -------------------------------
# The algorithms take DataFrames as input, while each dataset datapoint gives
# us access to a recording, reference events, and metadata. A pipeline connects
# these interfaces. ``run`` extracts the signal for prediction, and
# ``self_optimize`` extracts the signals and labels for fitting.
#
# We mark the whole algorithm as optimizable, so the pipeline does not need to
# know which parameters its algorithm learns. Both methods just delegate to the
# algorithm. This gives evaluation and parameter search a common interface.
from typing import Generic, TypeVar

from tpcp import OptimizablePipeline

A = TypeVar("A", bound=BaseEventDetector)


class EventPipeline(OptimizablePipeline[BaseMovementStudyDataset], Generic[A]):
    """Adapt prediction and optional algorithm fitting to a clinical dataset."""

    algorithm: OptimizableParameter[A]
    events_: np.ndarray

    def __init__(self, algorithm: A):
        self.algorithm = algorithm

    def run(self, datapoint: BaseMovementStudyDataset) -> Self:
        """Predict one recording without reading reference annotations."""
        self.events_ = self.algorithm.clone().detect(datapoint.data).events_
        return self

    def self_optimize(
        self, dataset: BaseMovementStudyDataset, **_kwargs
    ) -> Self:
        """Forward the supplied signals and labels to the algorithm."""
        self.algorithm = self.algorithm.clone().self_optimize(
            data=[d.data for d in dataset],
            labels=[d.reference_events_ for d in dataset],
        )
        return self


# %%
# Try the pipelines
# -----------------
# Let's check that this works before adding evaluation machinery. First, run
# the rule-based detector on one recording. Then fit the learnable detector on
# a few other recordings and run it on the same datapoint.

dataset_b = DatasetB()
datapoint = dataset_b[0]

EventPipeline(PeakDetector()).run(datapoint).events_[:20]
# %%
train_datapoints = dataset_b[1:3]

pipeline = EventPipeline(TemplateMatchingDetector())
pipeline.self_optimize(train_datapoints)

pipeline.algorithm.template

# %%
pipeline.run(datapoint).events_[:20]


# %%
# Score predictions
# -----------------
# We compare event positions rather than treating sample indices as class
# labels. Each detection can match one reference event within a fixed tolerance.
# Unmatched detections are false positives; unmatched references are false
# negatives. From these counts, we calculate precision, recall, and F1.
#
# This small helper matches events in chronological order. Once a reference
# event has a match, we cannot use it again.
# This is a very basic implementation. Do not use it in a real evaluation.
def event_detection_scores(
    events: np.ndarray, reference: np.ndarray, tolerance_samples: int
) -> dict[str, float]:
    """Calculate event precision, recall, and F1 with one-to-one matching."""
    events = np.sort(events)
    reference = np.sort(reference)
    reference_index = 0
    true_positives = 0
    for event in events:
        while (
            reference_index < len(reference)
            and reference[reference_index] < event - tolerance_samples
        ):
            reference_index += 1
        if (
            reference_index < len(reference)
            and reference[reference_index] <= event + tolerance_samples
        ):
            true_positives += 1
            reference_index += 1

    precision = true_positives / len(events) if len(events) else 0.0
    recall = true_positives / len(reference) if len(reference) else 0.0
    total = len(events) + len(reference)
    f1 = 2 * true_positives / total if total else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# %%
# The scorer connects this calculation to our pipeline and dataset interfaces.


def score(
    pipeline: EventPipeline, datapoint: BaseMovementStudyDataset
) -> dict[str, float]:
    """Compare detected and reference events for one recording."""
    prediction = pipeline.safe_run(datapoint).events_
    reference = datapoint.reference_events_
    return event_detection_scores(
        prediction,
        reference,
        tolerance_samples=5,  # 50 ms for our 100 Hz dummy recordings.
    )


# %%
# Search hyperparameters on training data
# ---------------------------------------
# Now we can add parameter search to both algorithms. For the learnable one,
# :class:`~tpcp.optimize.GridSearchCV` fits and evaluates parameter combinations
# on separate inner folds. For the rule-based one,
# :class:`~tpcp.optimize.GridSearch` searches inference parameters without a
# fitting step.
#
# We keep this search inside an outer train/test split, so the final evaluation
# uses unseen participants. Let's try this on dataset A first.
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from tpcp.optimize import GridSearch, GridSearchCV
from tpcp.validate import DatasetSplitter, Scorer, train_test_split, validate

# %%
# We hold out half the participants and keep all their recordings together.
# DatasetSplitter returns group labels, which we use to select the two datasets.
dataset_a = DatasetA()
holdout_splitter = DatasetSplitter(
    GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42),
    groupby="participant",
)
train_labels, test_labels = train_test_split(holdout_splitter, dataset_a)
train_data = dataset_a.get_subset(group_labels=train_labels)
test_data = dataset_a.get_subset(group_labels=test_labels)

# %%
# ``template_radius`` changes the training windows and learned template.
# ``min_correlation`` changes peak selection during inference.
# With ``pure_parameters=True``, GridSearchCV reads the
# algorithm's ``PureParameter`` annotations through the pipeline's
# ``OptimizableParameter`` field. It can reuse training across inference-only
# parameter changes. See :ref:`pure_parameters` for more on this distinction.
# The peak detector's search only changes its peak-height threshold.
scorer = Scorer(score, progress_bar=False)
searches = {
    "learnable": GridSearchCV(
        pipeline=EventPipeline(TemplateMatchingDetector()),
        parameter_grid=ParameterGrid(
            {
                "algorithm__template_radius": [15, 25],
                "algorithm__min_correlation": [0.65, 0.85],
            }
        ),
        pure_parameters=True,
        scoring=scorer,
        cv=DatasetSplitter(2, groupby="participant"),
        return_optimized="f1",
        progress_bar=False,
    ),
    "baseline": GridSearch(
        pipeline=EventPipeline(PeakDetector()),
        parameter_grid=ParameterGrid({"algorithm__threshold": [0.5, 0.9, 1.1]}),
        scoring=scorer,
        return_optimized="f1",
        progress_bar=False,
    ),
}

# %%
# Both searches have the same interface. They only see the outer training data.
# GridSearchCV selects parameters using its inner participant-wise folds and
# then refits the winner on all training recordings.
optimized_searches = {
    name: search.clone().optimize(train_data)
    for name, search in searches.items()
}
pd.DataFrame(
    {name: search.best_params_ for name, search in optimized_searches.items()}
)

# %%
# Finally, evaluate the selected pipelines on the held-out participants.
# validate performs prediction and scoring, without fitting or selecting
# parameters on the test data.
test_results = {
    name: validate(search.optimized_pipeline_, test_data, scoring=scorer)
    for name, search in optimized_searches.items()
}
pd.concat(
    {name: pd.DataFrame(result) for name, result in test_results.items()}
).filter(like="agg__")

# %%
# Assemble the final dataset and splitters
# ----------------------------------------
# Now we can set up the full evaluation on the combined dataset. We compare two
# training conditions: hospital recordings only, and hospital plus home
# recordings. Both must be evaluated on home recordings from unseen
# participants.
#
# :class:`~tpcp.validate.SubsetSplitter` applies a splitter to just dataset A.
# Within A, :class:`~tpcp.validate.DatasetSplitter` assigns whole participants
# to folds. Then ``test_selector`` keeps only home recordings for evaluation.
# We discard the held-out participants' hospital recordings from that fold.
#
# Dataset B contributes only to training. A train-only
# :class:`~tpcp.validate.NoSplit` adds all of B to each fold's training set.
# With ``n_splits=None``, it takes the fold count from A's splitter.
# :class:`~tpcp.validate.CombinedSplitter` combines the two assignments.

from tpcp.validate import CombinedSplitter, NoSplit, SubsetSplitter

combined_splitter = CombinedSplitter(
    parts=[
        (
            "A",
            SubsetSplitter(
                selector=lambda ds: ds.get_subset(origin="A"),
                splitter=DatasetSplitter(
                    3,
                    groupby="participant",
                    test_selector=lambda ds: ds.get_subset(condition="home"),
                ),
            ),
        ),
        (
            "B",
            NoSplit(
                n_splits=None,
                train=lambda ds: ds.get_subset(origin="B"),
            ),
        ),
    ]
)

# %%
# We can then derive a version that only keeps hospital recordings for training.
# We change ``train_selector`` in part A. ``test_selector`` stays the same:
# both conditions must be evaluated on the same unseen participants at home.
# Dataset B has only hospital recordings, so its part stays unchanged.
hospital_only_splitter = combined_splitter.clone().set_params(
    parts__A__splitter__train_selector=lambda ds: ds.get_subset(
        condition="hospital"
    ),
)

dataset = CombinedStudyDataset([("A", DatasetA()), ("B", DatasetB())])
dataset

# %%
# Configure the experiments
# -------------------------
# Each inner search receives the outer training data after source selection.
# For hospital-only training, this already contains only hospital recordings.
# We group by participant and stratify by origin to balance datasets A and B.
# For mixed training, we stratify by condition to balance hospital and home
# recordings. With both groupby and stratify, DatasetSplitter uses sklearn's
# StratifiedGroupKFold. It keeps participants together while trying to balance
# the categories across folds.
#
# B can contribute to inner validation for parameter selection. The outer test
# sets still contain only home recordings from unseen participants in A.
inner_hospital_splitter = DatasetSplitter(
    2,
    groupby="participant",
    stratify="origin",
)
inner_mixed_splitter = DatasetSplitter(
    2,
    groupby="participant",
    stratify="condition",
)

# %%
# Each experiment carries its search and its outer splitter.
# For the learnable algorithm, we configure the inner CV on the search itself.
# GridSearch has no fitting step and selects its parameters on the supplied
# training data. Both searches still use the same optimization interface.
experiments = {
    "hospital_only_learnable": (
        searches["learnable"].clone().set_params(cv=inner_hospital_splitter),
        hospital_only_splitter,
    ),
    "hospital_and_home_learnable": (
        searches["learnable"].clone().set_params(cv=inner_mixed_splitter),
        combined_splitter,
    ),
    "hospital_only_baseline": (
        searches["baseline"].clone(),
        hospital_only_splitter,
    ),
    "hospital_and_home_baseline": (
        searches["baseline"].clone(),
        combined_splitter,
    ),
}

# %%
# Evaluate one outer train/test split
# -----------------------------------
# First, let's run this with a single train/test split.
# We replace the grouped K-fold split in part A with a grouped shuffle split
# that holds out two of its six participants with ``n_splits=1``.
# This is equivalent to a single random, grouped train/test split.
# B still contributes all its recordings to training, and the source selectors
# stay in place.
# Notice that we only change a nested parameter of the splitter. We don't have
# to rewrite the splitting logic for each experiment.
holdout_experiments = {
    name: (
        search,
        splitter.clone().set_params(
            parts__A__splitter__base_splitter=GroupShuffleSplit(
                n_splits=1, test_size=2, random_state=42
            ),
        ),
    )
    for name, (search, splitter) in experiments.items()
}

# %%
# :func:`~tpcp.validate.cross_validate` takes the whole optimizer.
# For each outer fold, it clones the optimizer, runs its parameter search on
# the training data, and scores the selected pipeline on the test data.
# A splitter with one fold gives us a conventional train/test evaluation
# through this same interface. See :ref:`cross_validation` for more details.
# No evaluation code needs to know which algorithm it is running.
from tpcp.validate import cross_validate

holdout_results = {
    name: cross_validate(
        search,
        dataset,
        scoring=scorer,
        cv=splitter,
        return_optimizer=True,
        progress_bar=False,
    )
    for name, (search, splitter) in holdout_experiments.items()
}

# %%
# Each row is one experiment's outer fold. The test labels show that all four
# experiments use the same home recordings for evaluation.
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
# Now we use the original three-fold splitters from our experiment definitions.
# Everything else stays the same. Each outer training fold gets its own
# parameter search and, for the learnable algorithm, its own inner CV.
# Every participant from A is held out once, with only their home recordings
# used for evaluation. B is always available for training.
cv_results = {
    name: cross_validate(
        search,
        dataset,
        scoring=scorer,
        cv=splitter,
        return_optimizer=True,
        progress_bar=False,
    )
    for name, (search, splitter) in experiments.items()
}
cv_result_table = pd.concat(
    {name: pd.DataFrame(result) for name, result in cv_results.items()}
)
cv_result_table[score_columns]

# %%
# The scores average recordings. All participants in A have two home days,
# so they contribute equally. We can summarize the three outer folds below.
# Their standard deviation describes variation across folds; it is not a
# confidence interval. Keep the single-split and CV results separate.
cv_result_table.groupby(level=0)["test__agg__f1"].agg(["mean", "std"])

# %%
# With ``return_optimizer=True``, each result retains the fitted outer-fold
# search. We can inspect its selected parameters and, for GridSearchCV, the
# inner search results without fitting again.
cv_results["hospital_only_learnable"]["optimizer"][0].best_params_

# %%
# This gives us the comparison we wanted: hospital labels only versus hospital
# and home labels, evaluated on identical unseen participants at home.
# Both the learnable algorithm and the baseline use the same evaluation code.
# The dummy signals help us check the workflow; they cannot tell us whether
# home labels are needed for a real study.
# To fit a deployment algorithm, repeat the selected search on all available
# development data from the chosen condition. Report the held-out results as
# the performance estimate, rather than the score used to select parameters.
