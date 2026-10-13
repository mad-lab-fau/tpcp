# ruff: noqa: D205, D400
"""
Parallel Optuna Trials on Multiple GPUs
=======================================

This recipe selects a GPU inside a custom Optuna objective. The optimizer has a
``gpu_ids`` parameter containing the available GPU indices. TPCP supplies a
logical ``worker_id`` for each trial chunk, so the objective selects
``gpu_ids[worker_id]`` and builds its model inside a TensorFlow device scope.
Joblib remains responsible for scheduling the chunks in its process pool.

This parallelizes independent trials, rather than distributing one model fit.
The example runs a tiny NumPy regression on CPUs by default, without TensorFlow.
Supply ``GPU_IDS`` below to use the TensorFlow version. Install a GPU-enabled
TensorFlow build following the
`TensorFlow installation guide <https://www.tensorflow.org/install/pip>`_.
Indices refer to GPUs visible to TensorFlow, including restrictions imposed by
``CUDA_VISIBLE_DEVICES`` or your cluster scheduler.

Choosing the worker count
--------------------------
Copy this helper into your script to use the requested CPU worker count or one
job per entry in your GPU list. Use ``None`` for CPU execution and a nonempty
sequence of distinct GPU indices for GPU execution.
"""

import os
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from joblib import parallel_config
from optuna.samplers import TPESampler
from tpcp import (
    Dataset,
    OptimizableParameter,
    OptimizablePipeline,
    make_optimize_safe,
)
from tpcp.optimize import Optimize
from tpcp.optimize.optuna import CustomOptunaOptimize


def select_n_jobs(cpu_n_jobs: int, *, gpu_ids: Sequence[int] | None) -> int:
    """Use the supplied CPU worker count or the length of the GPU list."""
    return cpu_n_jobs if gpu_ids is None else len(gpu_ids)


# %%
# A tiny dataset and trainable pipeline
# --------------------------------------
# Replace these synthetic regression classes with your dataset and pipeline.
# The TensorFlow model is built during training, inside the objective's device
# scope. A model constructed earlier already has its variables placed on a
# device; wrapping its training call does not move those variables.


class RegressionDataset(Dataset):
    """Deterministic samples for a small linear regression."""

    def create_index(self):
        return pd.DataFrame({"sample": range(32)})

    @property
    def features(self) -> np.ndarray:
        """Input values and a bias column."""
        x = self.index["sample"].to_numpy(dtype=np.float32) / 31
        return np.column_stack([x, np.ones_like(x)])

    @property
    def target(self) -> np.ndarray:
        """Reference values."""
        return 2 * self.features[:, 0] - 0.5


class RegressionPipeline(OptimizablePipeline[RegressionDataset]):
    """Fit with NumPy or TensorFlow, inheriting the caller's device scope."""

    weights: OptimizableParameter[np.ndarray | None]

    def __init__(
        self,
        learning_rate: float = 0.1,
        *,
        use_gpu: bool = False,
        weights: np.ndarray | None = None,
    ):
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.weights = weights

    @make_optimize_safe
    def self_optimize(self, dataset: RegressionDataset):
        """Run a few gradient steps, keeping training deliberately tiny."""
        x, y = dataset.features, dataset.target
        if self.use_gpu:
            import tensorflow as tf

            x, y = tf.constant(x), tf.constant(y)
            weights = tf.Variable(tf.zeros(2))
            for _ in range(20):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(
                        tf.square(tf.linalg.matvec(x, weights) - y)
                    )
                weights.assign_sub(
                    self.learning_rate * tape.gradient(loss, weights)
                )
            self.weights = weights.numpy()
        else:
            weights = np.zeros(2, dtype=np.float32)
            for _ in range(20):
                weights -= self.learning_rate * (
                    2 * x.T @ (x @ weights - y) / len(y)
                )
            self.weights = weights
        return self

    def run(self, datapoint: RegressionDataset):
        """Predict using the fitted weights."""
        self.predictions_ = datapoint.features @ self.weights
        return self


# %%
# Custom search with a configured GPU list
# -----------------------------------------
# ``worker_id`` is the zero-based trial-chunk index. It is stable for every
# trial in that chunk, independent of the process that executes the job.
# TPCP inspects the objective signature once and passes this argument only
# when the objective declares it. Existing three-argument objectives still work.
#
# The dataclass interface adds ``gpu_ids`` while retaining the base optimizer's
# parameters; see :ref:`dataclasses` for more about this interface.
# Use the helper above for ``n_jobs`` so each chunk has one GPU-list entry.
# The objective trains on one subset and ranks candidates on a held-out subset.
# Use your usual inner cross-validation instead for a real experiment.
#
# ``tf.device`` scopes cover model construction and training. The final fit
# uses the first configured GPU through the same scope helper. CPU execution
# uses ``nullcontext`` and does not import TensorFlow.
# Memory growth is enabled before the first TensorFlow operation so processes
# can share visible GPUs without reserving their entire memory at startup.
# Run this recipe in a fresh script where TensorFlow has not been initialized
# with a different memory configuration.
# See `TensorFlow device scopes <https://www.tensorflow.org/api_docs/python/tf/device>`_
# and `GPU memory growth <https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth>`_.


@dataclass(repr=False)
class RegressionSearch(
    CustomOptunaOptimize.as_dataclass()[RegressionPipeline, RegressionDataset]
):
    """Select a learning rate, using the configured GPUs when supplied."""

    gpu_ids: Sequence[int] | None = None

    def _device_scope(self, worker_id: int):
        if self.gpu_ids is None:
            return nullcontext()

        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        return tf.device(f"/GPU:{self.gpu_ids[worker_id]}")

    def create_objective(self):
        def objective(trial, pipeline, dataset, *, worker_id):
            trial.set_user_attr("worker_id", worker_id)
            trial.set_user_attr("pid", os.getpid())
            trial.set_user_attr(
                "gpu_id",
                None if self.gpu_ids is None else self.gpu_ids[worker_id],
            )
            with self._device_scope(worker_id):
                pipeline.set_params(
                    learning_rate=trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    )
                )
                training = dataset.get_subset(index=dataset.index.iloc[::2])
                validation = dataset.get_subset(index=dataset.index.iloc[1::2])
                fitted = (
                    Optimize(pipeline).optimize(training).optimized_pipeline_
                )
                return float(
                    np.mean(
                        (
                            fitted.run(validation).predictions_
                            - validation.target
                        )
                        ** 2
                    )
                )

        return objective

    def return_optimized_pipeline(self, pipeline, dataset, study):
        with self._device_scope(0):
            return super().return_optimized_pipeline(pipeline, dataset, study)


# %%
# Shared local storage and execution
# ------------------------------------
# A fresh UUID gives each invocation its own database filename and study name.
# ``load_if_exists=False`` prevents accidentally reopening a study during its
# initial creation. TPCP then loads this same study in every job; there is no
# database per GPU or trial.
#
# SQLite is suitable for this small local example. For many workers or multiple
# nodes, use a server-backed Optuna database. Keep SQLite on a local disk.
# The database remains available for inspection or archiving after execution.
#
# The optimizer performs the final fit automatically on all samples. Joblib's
# normal pool reuse remains enabled; GPU choice is made inside each execution.


def get_study_params(seed: int, *, database: Path, study_name: str):
    """Share one local study and seed each job's sampler."""
    return {
        "storage": f"sqlite:///{database.resolve().as_posix()}",
        "study_name": study_name,
        "load_if_exists": False,
        "direction": "minimize",
        "sampler": TPESampler(seed=seed),
    }


def run_search(
    *, cpu_n_jobs: int, gpu_ids: Sequence[int] | None, study_directory: Path
):
    """Run the small search and let the optimizer refit on all samples."""
    run_id = uuid4().hex
    study_directory.mkdir(parents=True, exist_ok=True)
    search = RegressionSearch(
        pipeline=RegressionPipeline(use_gpu=gpu_ids is not None),
        get_study_params=partial(
            get_study_params,
            database=study_directory / f"study-{run_id}.sqlite3",
            study_name=f"regression-{run_id}",
        ),
        gpu_ids=gpu_ids,
        n_trials=4,
        n_jobs=select_n_jobs(cpu_n_jobs, gpu_ids=gpu_ids),
        random_seed=42,
        return_optimized=True,
    )
    with parallel_config(backend="loky"):
        search.optimize(RegressionDataset())
    return search, search.optimized_pipeline_


# %%
# Supply the GPU indices you want to use, or keep ``None`` for CPU execution.
# More GPUs allow more trials to run concurrently; they do not speed up one fit.
CPU_N_JOBS = 2
GPU_IDS = None
STUDY_DIRECTORY = Path("optuna_studies")

if __name__ == "__main__":
    search, fitted_pipeline = run_search(
        cpu_n_jobs=CPU_N_JOBS, gpu_ids=GPU_IDS, study_directory=STUDY_DIRECTORY
    )
    print(search.study_.study_name)
    print(
        search.study_.trials_dataframe(
            attrs=("number", "value", "params", "user_attrs")
        )
    )
    print("Final fitted weights:", fitted_pipeline.weights)
