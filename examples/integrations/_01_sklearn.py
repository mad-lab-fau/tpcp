"""
SciKit-Learn integration example
================================

Tpcp at its core is heavily inspired by SciKit-Learn and borrows many concepts from it.
However, there are some differences, which are explained in the
:ref:`differences to scikit-learn <differences_to_sklearn>`.
This means we need some level of integration with SciKit-Learn to make it easy to use tpcp in combination with it.

.. note:: In many cases it is not necessary to use SciKit-Learn in combination with tpcp.
   Just use scikit-learn on its own.
   Only if you have a case, where most of what you want to do is classical algorithmic stuff, but you need some ML
   (aka scikit-learn) in the middle, you might want to use tpcp to integrate everything.

In this example, we show a recipe, on how to create algorithms and pipelines, that integrate a sci-kit learn model.
We further show, how we can use tpcp based hyperparameter optimization to optimize model parameters and "regular"
algorithm parameters at the same time.

Basic integration
-----------------

For the most basic integration, we will design an algorithm that simply wraps a scikit-learn model.
We will pass the model as parameter to the algorithm and then implement a "action" method, that calls the predict method
of the model.
In the second step, we will implement a "self_optimize" method, that calls the fit method of the model.

We will create an algorithm, that is able to classify pre-ventricular contractions (PVCs) in ECG data.
It will get the raw data of one full recording and the positions of the R-peaks as input and outputs the indices of the
R-peaks, that are classified as PVCs.

Everything will be implemented on top of the dataset that was introduced in the
:ref:`dataset example <custom_dataset_ecg>`.

Understanding the data
~~~~~~~~~~~~~~~~~~~~~~
Let's have a look at the data that our action method will get as input.
This will be the raw ECG data and the positions of the R-peaks.
"""
from pathlib import Path

import numpy as np

from examples.datasets.datasets_final_ecg import ECGExampleData

# Loading the data
try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"

dataset = ECGExampleData(data_path=Path(data_path))
example_dp = dataset[0]

ecg_data = example_dp.data
example_r_peaks = example_dp.r_peak_positions_

ecg_data
# %%
example_r_peaks

# %%
# These two data structures are going to be our inputs.
# And we want to predict the subsets of the R-peaks, that are PVCs.
# They can be accessed via the `pvc_positions_` attribute of the data point.
example_dp.pvc_positions_

# %%
# Let's build a naive feature based classifier to do that (it will not be good, but should serve its purpose as an
# example).
# To extract the features, we will look at the 50 samples before and after each R-peak and calculate simple features
# like mean and standard deviation.
# We will do that first outside any algorithm class to understand the concept.

def extract_features(ecg_data, r_peak_positions):


    r_peak_regions = np.array([ecg_data.iloc[r_peak - 50 : r_peak + 50] for r_peak in r_peak_positions])

    features = np.array([
        np.mean(r_peak_regions, axis=1),
        np.std(r_peak_regions, axis=1)
    ])

    return features.T

features = extract_features(ecg_data, example_r_peaks["r_peak_position"])
features

# %%
# This represents the features for all r_peaks in one recording.
# We can now train a simple classifier on this data.
# We will use a support vector machine (SVM) for this.
from sklearn.svm import SVC

labels = np.full(False, len(example_r_peaks))
labels[example_dp.pvc_positions_] = True
model = SVC()
model.fit(features, labels)

# %%
# We test the model on our training data and see if it learned anything.
predictions = model.predict(features)
predictions
print(predictions)
# from enum import StrEnum
# from typing import Union
#
# import pandas as pd
# from sklearn.base import ClassifierMixin
# from sklearn.utils.validation import check_is_fitted
#
# from tpcp import Algorithm
#
#
# class _PretrainedModel(StrEnum):
#     """Enum for pretrained models."""
#     model_1 = "model_1"
#     model_2 = "model_2"
#     @staticmethod
#     def load_model(model_name)-> ClassifierMixin:
#         ...
#
#
# class CustomSklearn(Algorithm):
#     results_: pd.DataFrame
#
#     PRETRAINED_MODELS = _PretrainedModel
#
#     def __init__(self, model: Union[_PretrainedModel, ClassifierMixin] = _PretrainedModel.model_1):
#         self.model = model
#
#     @staticmethod
#     def _check_init_model(model) -> ClassifierMixin:
#         if isinstance(model, _PretrainedModel):
#             # Load model from somewhere
#             return _PretrainedModel.load_model(model.value)
#         if isinstance(model, ClassifierMixin):
#             return model
#         raise TypeError(f"Unknown model type {type(model)}")
#
#
#     def predict(self, data, sampling_rate_hz=None):
#         model = self._check_init_model(self.model)
#
#         if not check_is_fitted(model):
#             raise RuntimeError("Model is not fitted. Call self_optimize before calling predict.")
#
#         self.results_ = model.predict(data)
#
#         return self
#
#     def self_optimize(self, data):
#         model = self._check_init_model(self.model)
#         model = model.fit(data)
#         self.model = model
#         return self
#
#
# CustomSklearn.PRETRAINED_MODELS.model_1