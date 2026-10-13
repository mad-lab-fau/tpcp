# ruff: noqa: D205, D400
"""
Small Trainable Regression Pipeline
===================================

Synthetic dataset and pipeline used by the :ref:`multi_gpu_optuna` recipe.
Keeping the classes in an importable module allows joblib workers to serialize
and hash them by their module names, including when the recipe runs in the
documentation gallery. The caller selects the TensorFlow device scope before
training; fitted weights are stored as a NumPy array.
"""

import numpy as np
import pandas as pd
from tpcp import (
    Dataset,
    OptimizableParameter,
    OptimizablePipeline,
    make_optimize_safe,
)


class RegressionDataset(Dataset):
    """Deterministic samples for a small linear regression."""

    def create_index(self):
        """Index the synthetic regression samples."""
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
