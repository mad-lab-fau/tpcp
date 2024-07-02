r"""
.. _tensorflow:

Tensorflow/Keras
================

.. note:: This example requires the `tensorflow` package to be installed.

Theoretically, tpcp is framework agnostic and can be used with any framework.
However, due to the way some frameworks handle their objects, some special handling internally is required.
Hence, this example does not only serve as example on how to use tensorflow with tpcp, but also as a test case for these
special cases.

When using tpcp with any machine learning framework, you either want to use a pretrained model with a normal pipeline
or a train your own model as part of an Optimizable Pipeline.
Here we show the second case, as it is more complex, and you are likely able to figure out the first case yourself.

This means, we are planning to perform the following steps:

1. Create a pipeline that creates and trains a model.
2. Allow the modification of model hyperparameters.
3. Run a simple cross-validation to demonstrate the functionality.

This example reimplements the basic MNIST example from the
[tensorflow documentation](https://www.tensorflow.org/tutorials/keras/classification).

Some Notes
----------

In this example we show how to implement a Pipeline that uses tensorflow.
You could implement an Algorithm in a similar way.
This would actually be easier, as no specific handling of the input data would be required.
For a pipeline, we need to create a custom Dataset class, as this is the expected input for a pipeline.

"""


# %%
# The Dataset
# -----------
# We are using the normal fashion MNIST dataset for this example
# It consists of 60.000 images of 28x28 pixels, each with a label.
# We will ignore the typical train-test split, as we want to do our own cross-validation.
#
# In addition, we will simulate an additional "index level".
# In this (and most typical deep learning datasets), each datapoint is one vector for which we can make one prediction.
# In tpcp, we usually deal with datasets, where you might have multiple pieces of information for each datapoint.
# For example, one datapoint could be a patient, for which we have an entire time series of measurements.
# We will simulate this here, by creating the index of our dataset as 1000 groups each containing 60 images.
#
# Other than that, the dataset is pretty standard.
# Besides the `create_index` method, we only need to implement the `input_as_array` and `labels_as_array` methods that
# allow us to easily access the data once we selected a single group.
from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf

from tpcp import Dataset

tf.keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()


@lru_cache(maxsize=1)
def get_fashion_mnist_data():
    # Note: We throw train and test sets together, as we don't care about the official split here.
    #       We will create our own split later.
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    return np.array(list(train_images) + list(test_images)), list(train_labels) + list(test_labels)


class FashionMNIST(Dataset):
    def input_as_array(self) -> np.ndarray:
        self.assert_is_single(None, "input_as_array")
        group_id = int(self.group_label.group_id)
        images, _ = get_fashion_mnist_data()
        return images[group_id * 60 : (group_id + 1) * 60].reshape((60, 28, 28)) / 255

    def labels_as_array(self) -> np.ndarray:
        self.assert_is_single(None, "labels_as_array")
        group_id = int(self.group_label.group_id)
        _, labels = get_fashion_mnist_data()
        return np.array(labels[group_id * 60 : (group_id + 1) * 60])

    def create_index(self) -> pd.DataFrame:
        # There are 60.000 images in total.
        # We simulate 1000 groups of 60 images each.
        return pd.DataFrame({"group_id": list(range(1000))})


# %%
# We can see our Dataset works as expected:
dataset = FashionMNIST()
dataset[0].input_as_array().shape

# %%
dataset[0].labels_as_array().shape

# %%
# The Pipeline
# ------------
# We will create a pipeline that uses a simple neural network to classify the images.
# In tpcp, all "things" that should be optimized need to be parameters.
# This means our model itself needs to be a parameter of the pipeline.
# However, as we don't have the model yet, as its creation depends on other hyperparameters, we add it as an optional
# parameter initialized with `None`.
# Further, we prefix the parameter name with an underscore, to signify, that this is not a parameter that should be
# modified manually by the user.
# This is just convention, and it is up to you to decide how you want to name your parameters.
#
# We further introduce a hyperparameter `n_dense_layer_nodes` to show how we can influence the model creation.
#
# The optimize method
# +++++++++++++++++++
# To make our pipeline optimizable, it needs to inherit from `OptimizablePipeline`.
# Further we need to mark at least one of the parameters as `OptiPara` using the type annotation.
# We do this for our `_model` parameter.
#
# Finally, we need to implement the `self_optimize` method.
# This method will get the entire training dataset as input and should update the `_model` parameter with the trained
# model.
# Hence, we first extract the relevant data (remember, each datapoint is 60 images), by concatinating all images over
# all groups in the dataset.
# Then we create the Keras model based on the hyperparameters.
# Finally, we train the model and update the `_model` parameter.
#
# Here we chose to wrap the method with `make_optimize_safe`.
# This decorator will perform some runtime checks to ensure that the method is implemented correctly.
#
# The run method
# ++++++++++++++
# The run method expects that the `_model` parameter is already set (i.e. the pipeline was already optimized).
# It gets a single datapoint as input (remember, a datapoint is a single group of 60 images).
# We then extract the data from the datapoint and let the model make a prediction.
# We store the prediction on our output attribute `predictions_`.
# The trailing underscore is a convention to signify, that this is an "result" attribute.
import warnings
from typing import Optional

from typing_extensions import Self

from tpcp import OptimizablePipeline, OptiPara, make_action_safe, make_optimize_safe


class KerasPipeline(OptimizablePipeline):
    n_dense_layer_nodes: int
    n_train_epochs: int
    _model: OptiPara[Optional[tf.keras.Sequential]]

    predictions_: np.ndarray

    def __init__(self, n_dense_layer_nodes=128, n_train_epochs=5, _model: Optional[tf.keras.Sequential] = None):
        self.n_dense_layer_nodes = n_dense_layer_nodes
        self.n_train_epochs = n_train_epochs
        self._model = _model

    @property
    def predicted_labels_(self):
        return np.argmax(self.predictions_, axis=1)

    @make_optimize_safe
    def self_optimize(self, dataset, **_) -> Self:
        data = np.vstack([d.input_as_array() for d in dataset])
        labels = np.hstack([d.labels_as_array() for d in dataset])

        print(data.shape)
        if self._model is not None:
            warnings.warn("Overwriting existing model!")

        self._model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.n_dense_layer_nodes, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        self._model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self._model.fit(data, labels, epochs=self.n_train_epochs)

        return self

    @make_action_safe
    def run(self, datapoint) -> Self:
        if self._model is None:
            raise RuntimeError("Model not trained yet!")
        data = datapoint.input_as_array()

        self.predictions_ = self._model.predict(data)
        return self


# %%
# Testing the pipeline
# --------------------
# We can now test our pipeline.
# We will run the optimization using a couple of datapoints (to keep everything fast) and then use `run` to get the
# predictions for a single unseen datapoint.
pipeline = KerasPipeline().self_optimize(FashionMNIST()[:10])
p1 = pipeline.run(FashionMNIST()[11])
print(p1.predicted_labels_)
print(FashionMNIST()[11].labels_as_array())

# %%
# We can see that even with just 5 epochs, the model already performs quite well.
# To quantify we can calculate the accuracy for this datapoint:
from sklearn.metrics import accuracy_score

accuracy_score(p1.predicted_labels_, FashionMNIST()[11].labels_as_array())

# %%
# Cross Validation
# ----------------
# If we want to run a cross validation, we need to formalize the scoring into a function.
# We will calculate two types of accuracy:
# First, the accuracy per group and second, the accuracy over all images across all groups.
# For more information about how this works, check the :ref:`Custom Scorer <custom_scorer>` example.
from collections.abc import Sequence

from tpcp.validate import Aggregator


class SingleValueAccuracy(Aggregator[np.ndarray]):
    RETURN_RAW_SCORES = False

    @classmethod
    def aggregate(cls, /, values: Sequence[tuple[np.ndarray, np.ndarray]], **_) -> dict[str, float]:
        return {"accuracy": accuracy_score(np.hstack([v[0] for v in values]), np.hstack([v[1] for v in values]))}


def scoring(pipeline, datapoint):
    result: np.ndarray = pipeline.safe_run(datapoint).predicted_labels_
    reference = datapoint.labels_as_array()

    return {
        "accuracy": accuracy_score(result, reference),
        "per_sample": SingleValueAccuracy((result, reference)),
    }


# %%
# Now we can run a cross validation.
# We will only run it on a subset of the data, to keep the runtime manageable.
#
# .. note:: You might see warnings about retracing of the model.
#           This is because we clone the pipeline before each call to the run method.
#           This is a good idea to ensure that all pipelines are independent of each other, however, might result in
#           some performance overhead.
from tpcp.optimize import Optimize
from tpcp.validate import cross_validate

pipeline = KerasPipeline(n_train_epochs=10)
cv_results = cross_validate(Optimize(pipeline), FashionMNIST()[:100], scoring=scoring, cv=3)

# %%
# We can now look at the results per group:
cv_results["test__single__accuracy"]

# %%
# And the overall accuracy as the average over all samples of all groups within a fold:
cv_results["test__per_sample__accuracy"]
