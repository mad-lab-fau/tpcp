# ruff: noqa: D400
"""
.. _multi_gpu_optuna:

Parallel Optuna Trials on Multiple GPUs
=======================================

Use one trial chunk per configured GPU. TPCP passes a ``worker_id`` to a custom
objective that declares it, and the objective selects ``gpu_ids[worker_id]``.
The core pattern is:

.. code-block:: python

    n_jobs = len(gpu_ids)


    def objective(trial, pipeline, dataset, *, worker_id):
        gpu_id = gpu_ids[worker_id]
        with tf.device(f"/GPU:{gpu_id}"):
            # Construct and train the model here, then return its score.
            ...

For example, with ``gpu_ids=[2, 5]`` and ``n_jobs=2``, chunk 0 evaluates its
trials on GPU 2 and chunk 1 on GPU 5. Each chunk evaluates its trials
sequentially; the chunks can run concurrently. This parallelizes independent
trials. It does not distribute a single model fit across GPUs.

A ``worker_id`` identifies a chunk, not an operating-system process. It stays
the same for every trial in that chunk. Joblib schedules the chunks and may
reuse a process for a different chunk in a later search. Serial execution uses
``worker_id=0``. Existing objectives without this argument still work.

Runnable CPU and GPU example
----------------------------
The complete example below runs a tiny NumPy regression on CPUs by default,
without TensorFlow. Set ``GPU_IDS = [0, 1]`` to try two GPUs, or supply your own
nonempty sequence of distinct GPU indices. Install a GPU-enabled TensorFlow
build following the
`TensorFlow installation guide <https://www.tensorflow.org/install/pip>`_.
Indices refer to GPUs visible to TensorFlow, including restrictions imposed by
``CUDA_VISIBLE_DEVICES`` or your cluster scheduler.

Run the example from the repository root:

.. code-block:: bash

    python -m examples.recipies._08_multi_gpu_optuna

The imported :ref:`regression dataset and pipeline
<sphx_glr_auto_examples_algorithms_algorithms_linear_regression.py>` provide
small demo implementations. Replace them with your own dataset and pipeline.
The pipeline constructs its TensorFlow variables during training, inside the
objective's device scope. Wrapping the training of an already constructed
model in a device scope does not move its existing variables.
"""

import os
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
from joblib import parallel_config
from optuna.samplers import TPESampler
from tpcp.optimize import Optimize
from tpcp.optimize.optuna import CustomOptunaOptimize

from examples.algorithms.algorithms_linear_regression import (
    RegressionDataset,
    RegressionPipeline,
)

# %%
# Select a device for trials and the final fit
# --------------------------------------------
# The objective sets a learning rate, trains on one subset, and measures mean
# squared error on a held-out subset. Use your usual inner cross-validation
# for a real experiment.
#
# ``gpu_ids`` is an optimizer parameter. The dataclass interface retains the
# base optimizer's parameters; see :ref:`dataclasses` for details.
# The same device-scope helper covers trial training and the final fit. CPU
# execution uses ``nullcontext`` and does not import TensorFlow.


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

        # Enable memory growth before the first TensorFlow operation so each
        # process allocates GPU memory as needed. Use a fresh script where
        # TensorFlow has not been initialized with a different configuration.
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
                predictions = fitted.run(validation).predictions_
                mse = np.mean((predictions - validation.target) ** 2)
                return float(mse)

        return objective

    def return_optimized_pipeline(self, pipeline, dataset, study):
        # TPCP refits in the parent process. Select the first configured GPU.
        with self._device_scope(0):
            return super().return_optimized_pipeline(pipeline, dataset, study)


# %%
# Configure workers and shared storage
# ------------------------------------
# Use the requested CPU worker count or one job per configured GPU.
# ``gpu_ids=None`` selects CPU execution.
# See `TensorFlow device scopes <https://www.tensorflow.org/api_docs/python/tf/device>`_
# and `GPU memory growth <https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth>`_
# for the TensorFlow configuration used above.
#
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


def select_n_jobs(cpu_n_jobs: int, *, gpu_ids: Sequence[int] | None) -> int:
    """Use the supplied CPU worker count or the length of the GPU list."""
    return cpu_n_jobs if gpu_ids is None else len(gpu_ids)


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
CPU_N_JOBS = 2
GPU_IDS = None
STUDY_DIRECTORY = Path("optuna_studies")

if __name__ == "__main__":
    search, fitted_pipeline = run_search(
        cpu_n_jobs=CPU_N_JOBS, gpu_ids=GPU_IDS, study_directory=STUDY_DIRECTORY
    )
    print(search.study_.study_name)
    trial_table = search.study_.trials_dataframe(
        attrs=("number", "value", "user_attrs")
    ).rename(
        columns={
            "number": "trial",
            "value": "mse",
            "user_attrs_worker_id": "chunk",
            "user_attrs_pid": "pid",
            "user_attrs_gpu_id": "requested_gpu",
        }
    )
    print(
        trial_table[
            ["trial", "mse", "chunk", "pid", "requested_gpu"]
        ].to_string(index=False)
    )
    print("Final fitted weights:", fitted_pipeline.weights)

# %%
# Read the trial table
# --------------------
# Each row shows a trial's score, chunk ID, process ID, and requested GPU.
# Trials in the same chunk have the same ``chunk`` and ``requested_gpu``.
# The default CPU run shows chunk scheduling and automatic final refitting;
# ``requested_gpu`` is ``None`` because it makes no GPU request.
#
# With GPU execution enabled, ``requested_gpu`` records the device selected by
# the objective. It does not verify where TensorFlow placed each operation.
# To confirm actual placement on your hardware, enable TensorFlow's
# `device placement logging <https://www.tensorflow.org/guide/gpu#logging_device_placement>`_
# before the first TensorFlow operation inside each worker.
