import numpy.testing
import pytest

from tpcp import clone
from tpcp._hash import custom_hash

tensorflow = pytest.importorskip("tensorflow")

import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def create_model(input_shape=(3,)):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(4, activation="relu", input_shape=input_shape, kernel_initializer=GlorotUniform(seed=42)))
    model.add(Dense(3, activation="relu", input_shape=input_shape, kernel_initializer=GlorotUniform(seed=42)))
    return model


def create_fitted_model():
    model = create_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.fit(tf.constant([[0, 1, 2]]), tf.constant([[0, 1, 2]]))
    return model


@pytest.fixture(params=(create_model(), tf.constant([0, 1, 2])))
def tensorflow_objects(request):
    return request.param


def test_model_equivalent():
    model1 = create_model()
    model2 = create_model()
    cloned_model = clone(model1)

    numpy.testing.assert_equal(model1.get_weights(), model2.get_weights())
    assert model1.get_config() == model2.get_config()

    numpy.testing.assert_equal(model1.get_weights(), cloned_model.get_weights())
    assert model1.get_config() == cloned_model.get_config()


def test_trained_model_equivalent():
    model1 = create_fitted_model()
    model2 = create_fitted_model()
    cloned_model = clone(model1)

    numpy.testing.assert_equal(model1.get_weights(), model2.get_weights())
    assert model1.get_config() == model2.get_config()

    numpy.testing.assert_equal(model1.get_weights(), cloned_model.get_weights())
    assert model1.get_config() == cloned_model.get_config()

    assert (model1.predict(tf.constant([[0, 1, 2]])) == model2.predict(tf.constant([[0, 1, 2]]))).all()
    assert (model1.predict(tf.constant([[0, 1, 2]])) == cloned_model.predict(tf.constant([[0, 1, 2]]))).all()


def test_hash_model():
    model = create_model()
    first = custom_hash(model)
    second = custom_hash(create_model())
    cloned = custom_hash(clone(model))

    different_model = create_model(input_shape=(4,))

    assert first == second
    # assert first == cloned

    assert custom_hash(different_model) != first


def test_hash_trained_model():
    model = create_fitted_model()
    first = custom_hash(model)
    # second = custom_hash(create_model().compile(optimizer="adam", loss="categorical_crossentropy"))
    cloned = custom_hash(clone(model))

    # assert first == second
    assert first == cloned


def test_hash_tensor():
    data = [0, 1, 2]
    tensor = tf.constant(data)

    first = custom_hash(tensor)
    cloned = custom_hash(clone(tensor))
    second = custom_hash(tf.constant(data))

    assert first == second
    assert first == cloned


@pytest.mark.parametrize("c", (list, tuple))
def test_container_tensor(tensorflow_objects, c):
    tmp = c([tensorflow_objects])
    assert custom_hash(tmp) == custom_hash(clone(tmp))


def test_dict_tensor(tensorflow_objects):
    tmp = {"a": tensorflow_objects}
    assert custom_hash(tmp) == custom_hash(clone(tmp))
