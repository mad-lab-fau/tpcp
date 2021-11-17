"""This tests the BaseAlgorithm and fundamental functionality."""
from inspect import Parameter, signature
from typing import Any, Dict, Tuple

import joblib
import pytest

from tpcp import BaseAlgorithm, default, mdf
from tpcp._utils._general import _DEFAULT_PARA_NAME
from tpcp.exceptions import MutableDefaultsError


def _init_getter():
    def _fake_init(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    return _fake_init


def create_test_class(action_method_name, params=None, private_params=None, action_method=None, **_) -> BaseAlgorithm:
    params = params or {}
    private_params = private_params or {}

    class_dict = {"_action_method": action_method_name, "__init__": _init_getter()}
    user_set_params = {**params, **private_params}
    if action_method:
        class_dict = {**class_dict, action_method_name: action_method}

    # We create the class once, then create a proper signature and then create the class again to trigger the
    # signature related checks of the metaclass
    test_class = type("TestClass", (BaseAlgorithm,), class_dict)

    # Set the signature to conform to the expected conventions
    sig = signature(test_class.__init__)
    sig = sig.replace(parameters=(Parameter(k, Parameter.KEYWORD_ONLY, default=v) for k, v in params.items()))
    test_class.__init__.__signature__ = sig
    class_dict = {**class_dict, "__init__": test_class.__init__}
    # Recreate the class with the correct init
    test_class = type("TestClass", (BaseAlgorithm,), class_dict)

    test_instance = test_class(**user_set_params)

    return test_instance


@pytest.fixture(
    params=[
        dict(
            action_method_name="test",
            attributes={"attr1_": "test1"},
            params={},
            other_params={},
            private_params={},
            action_method=None,
        ),
        dict(
            action_method_name="test",
            attributes={"attr1_": "test1", "attr2_": "test2"},
            params={"para1": "test1", "para2": "test2"},
            other_params={"other_para1": "other_test1", "other_para2": "other_test2"},
            private_params={"_private": "private_test"},
            action_method=lambda self=None: "test",
        ),
    ]
)
def example_test_class_initialised(request) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    test_instance = create_test_class(**request.param)
    return test_instance, request.param


@pytest.fixture()
def example_test_class_after_action(example_test_class_initialised) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    test_instance, params = example_test_class_initialised
    action_params = {
        **params["attributes"],
        **params["other_params"],
    }
    for k, v in action_params.items():
        setattr(test_instance, k, v)
    return test_instance, params


def test_get_action_method(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance._action_method == test_parameters["action_method_name"]
    if test_parameters["action_method"] is not None:
        assert instance._get_action_method()() == test_parameters["action_method"]()
    else:
        with pytest.raises(AttributeError):
            instance._get_action_method()


def test_get_attributes(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_attributes() == test_parameters["attributes"]


def test_get_parameter(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_params() == test_parameters["params"]


def test_get_other_parameter(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_other_params() == test_parameters["other_params"]


def test_normal_wrong_attr_still_raises_attr_error(example_test_class_initialised):
    instance, test_parameters = example_test_class_initialised

    key = "not_existend_without_underscore"

    with pytest.raises(AttributeError) as e:
        getattr(instance, key)

    assert "result" not in str(e.value)
    assert key in str(e.value)
    assert instance._action_method not in str(e.value)


@pytest.mark.parametrize("key", ["wrong_with_", "wrong_without"])
def test_attribute_helper_after_action_wrong(example_test_class_after_action, key):
    instance, test_parameters = example_test_class_after_action

    if not test_parameters["attributes"]:
        pytest.skip("Invalid fixture for this test")

    with pytest.raises(AttributeError) as e:
        getattr(instance, key)

    assert "result" not in str(e.value)
    assert key in str(e.value)
    assert instance._action_method not in str(e.value)


def test_action_is_not_applied(example_test_class_initialised):
    instance, _ = example_test_class_initialised

    assert instance._action_is_applied is False


def test_action_is_applied(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    if not test_parameters["attributes"]:
        pytest.skip("Invalid fixture for this test")

    assert instance._action_is_applied is True


def test_nested_get_params():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": default(nested_instance)})

    params = test_instance.get_params()

    assert isinstance(params["nested_class"], nested_instance.__class__)

    for k, v in nested_instance.get_params().items():
        assert params["nested_class__" + k] == v

    for k, v in top_level_params.items():
        assert params[k] == v


def test_nested_set_params():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": default(nested_instance)})
    # We get the actual object here again, because the meta class has created a copy of the nested instance.
    nested_instance = test_instance.nested_class
    new_params_top_level = {"test1": "new_t1"}
    new_params_nested = {"nested2": "new_n2"}
    test_instance.set_params(**new_params_top_level, **{"nested_class__" + k: v for k, v in new_params_nested.items()})

    params = test_instance.get_params()
    params_nested = nested_instance.get_params()

    for k, v in new_params_top_level.items():
        assert params[k] == v
    for k, v in new_params_nested.items():
        assert params["nested_class__" + k] == v
        assert params_nested[k] == v


def test_nested_clone():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": default(nested_instance)})

    cloned_instance = test_instance.clone()

    # Check that the ids are different
    assert test_instance is not cloned_instance
    assert test_instance.nested_class is not cloned_instance.nested_class

    params = test_instance._get_params_without_nested_class()
    cloned_params = cloned_instance._get_params_without_nested_class()

    for k, v in params.items():
        assert cloned_params[k] == v


def test_clone_mutable():
    """Test cloning with mutable paras.

    In this case, we expect that a deepcopy of the mutable objects are created.
    I.e. content stays the same, but memory address changes.
    """
    mutable = {"m1": "m1", "m2": "m2"}
    test_instance = create_test_class("mutable", params={"normal": "n1", "mutable": mutable})

    # deep cloning (the default)
    cloned_instance = test_instance.clone()

    # Check that the object is different, but the content is still the same
    assert cloned_instance is not test_instance
    assert cloned_instance.mutable is not test_instance.mutable
    assert test_instance.mutable is mutable
    assert cloned_instance.mutable is not mutable
    assert joblib.hash(cloned_instance.mutable) == joblib.hash(test_instance.mutable) == joblib.hash(mutable)


def test_mutable_default_nested_objects_error():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})

    with pytest.raises(MutableDefaultsError):
        create_test_class("mutable", params={"normal": "n1", "mutable": nested_instance})

    # When wrapped in default, no error is raised
    create_test_class("mutable", params={"normal": "n1", "mutable": default(nested_instance)})


@pytest.mark.parametrize("wrapper", (mdf, default))
def test_default_wrapper(wrapper, example_test_class_initialised):
    wrapped = wrapper(example_test_class_initialised[0])
    assert getattr(wrapped, _DEFAULT_PARA_NAME) is True


def test_nested_mutable_algorithm_copy():
    """When a object is wrapped with default, it will be copied on init."""
    nested_params = {"nested1": "n1", "nested2": "n2"}
    nested_instance = create_test_class("nested", params=nested_params)

    test_instance = create_test_class("mutable", params={"normal": "n1", "mutable": default(nested_instance)})

    assert nested_instance is not test_instance.mutable
    assert hasattr(test_instance.mutable, _DEFAULT_PARA_NAME) is False
    assert (
        joblib.hash(test_instance.mutable.get_params())
        == joblib.hash(nested_instance.get_params())
        == joblib.hash(nested_params)
    )
