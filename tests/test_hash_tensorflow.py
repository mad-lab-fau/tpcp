import numpy.testing
import pytest

from tpcp import clone
from tpcp._hash import custom_hash

tensorflow = pytest.importorskip("tensorflow")

import tensorflow as tf


def create_model():
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(
        4, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
    )(inputs)
    outputs = tf.keras.layers.Dense(
        3, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
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

    assert first == second
    assert first == cloned


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
