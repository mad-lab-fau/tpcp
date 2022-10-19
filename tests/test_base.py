"""This tests the BaseAlgorithm and fundamental functionality."""
import dataclasses
from collections import namedtuple
from inspect import Parameter, signature
from typing import Any, ClassVar, Dict, Tuple
from unittest.mock import patch

import joblib
import pytest

from tests.conftest import _get_params_without_nested_class
from tpcp import Algorithm, OptimizablePipeline, OptiPara, Para, cf, clone
from tpcp._algorithm_utils import (
    get_action_method,
    get_action_methods_names,
    get_action_params,
    get_results,
    is_action_applied,
)
from tpcp._base import _get_tpcp_validated, _validate_parameter
from tpcp._parameters import _ParaTypes
from tpcp.exceptions import MutableDefaultsError, ValidationError


def _init_getter():
    def _fake_init(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    return _fake_init


def create_test_class(
    action_method_name, params=None, private_params=None, action_method_callable=None, **_
) -> Algorithm:
    params = params or {}
    private_params = private_params or {}

    class_dict = {"_action_methods": action_method_name, "__init__": _init_getter()}
    user_set_params = {**params, **private_params}
    if action_method_callable:
        class_dict = {**class_dict, action_method_name: action_method_callable}

    # We create the class once, then create a proper signature and then create the class again to trigger the
    # signature related checks of the metaclass
    test_class = type("TestClass", (object,), class_dict)

    # Set the signature to conform to the expected conventions
    sig = signature(test_class.__init__)
    sig = sig.replace(parameters=(Parameter(k, Parameter.KEYWORD_ONLY, default=v) for k, v in params.items()))
    test_class.__init__.__signature__ = sig
    class_dict = {**class_dict, "__init__": test_class.__init__}
    # Recreate the class with the correct init
    test_class = type("TestClass", (Algorithm,), class_dict)

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
            action_method_callable=None,
        ),
        dict(
            action_method_name="test",
            attributes={"attr1_": "test1", "attr2_": "test2"},
            params={"para1": "test1", "para2": "test2"},
            other_params={"other_para1": "other_test1", "other_para2": "other_test2"},
            private_params={"_private": "private_test"},
            action_method_callable=lambda self=None: "test",
        ),
    ]
)
def example_test_class_initialised(request) -> Tuple[Algorithm, Dict[str, Any]]:
    test_instance = create_test_class(**request.param)
    return test_instance, request.param


@pytest.fixture()
def example_test_class_after_action(example_test_class_initialised) -> Tuple[Algorithm, Dict[str, Any]]:
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

    assert instance._action_methods == test_parameters["action_method_name"]
    if test_parameters["action_method_callable"] is not None:
        assert get_action_method(instance)() == test_parameters["action_method_callable"]()
    else:
        with pytest.raises(AttributeError):
            get_action_method(instance)


def test_get_results(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert get_results(instance) == test_parameters["attributes"]


def test_get_parameter(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_params() == test_parameters["params"]


def test_get_action_params(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert get_action_params(instance) == test_parameters["other_params"]


def test_normal_wrong_attr_still_raises_attr_error(example_test_class_initialised):
    instance, test_parameters = example_test_class_initialised

    key = "not_existend_without_underscore"

    with pytest.raises(AttributeError) as e:
        getattr(instance, key)

    assert "result" not in str(e.value)
    assert key in str(e.value)
    assert get_action_methods_names(instance)[0] not in str(e.value)


@pytest.mark.parametrize("key", ["wrong_with_", "wrong_without"])
def test_attribute_helper_after_action_wrong(example_test_class_after_action, key):
    instance, test_parameters = example_test_class_after_action

    if not test_parameters["attributes"]:
        pytest.skip("Invalid fixture for this test")

    with pytest.raises(AttributeError) as e:
        getattr(instance, key)

    assert "result" not in str(e.value)
    assert key in str(e.value)
    assert get_action_methods_names(instance)[0] not in str(e.value)


def test_action_is_not_applied(example_test_class_initialised):
    instance, _ = example_test_class_initialised

    assert is_action_applied(instance) is False


def test_action_is_applied(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    if not test_parameters["attributes"]:
        pytest.skip("Invalid fixture for this test")

    assert is_action_applied(instance) is True


def test_nested_get_params():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": cf(nested_instance)})

    params = test_instance.get_params()

    assert isinstance(params["nested_class"], nested_instance.__class__)

    for k, v in nested_instance.get_params().items():
        assert params["nested_class__" + k] == v

    for k, v in top_level_params.items():
        assert params[k] == v


def test_nested_set_params():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": cf(nested_instance)})
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
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": cf(nested_instance)})

    cloned_instance = clone(test_instance)

    # Check that the ids are different
    assert test_instance is not cloned_instance
    assert test_instance.nested_class is not cloned_instance.nested_class

    params = _get_params_without_nested_class(test_instance)
    cloned_params = _get_params_without_nested_class(cloned_instance)

    for k, v in params.items():
        assert cloned_params[k] == v


def test_clone_mutable():
    """Test cloning with mutable paras.

    In this case, we expect that a deepcopy of the mutable objects are created.
    I.e. content stays the same, but memory address changes.
    """
    mutable = {"m1": "m1", "m2": "m2"}
    test_instance = create_test_class("mutable", params={"mutable": None, "normal": "n1"})
    test_instance.mutable = mutable

    # deep cloning (the default)
    cloned_instance = test_instance.clone()

    # Check that the object is different, but the content is still the same
    assert cloned_instance is not test_instance
    assert cloned_instance.mutable is not test_instance.mutable
    assert test_instance.mutable is mutable
    assert cloned_instance.mutable is not mutable
    assert joblib.hash(cloned_instance.mutable) == joblib.hash(test_instance.mutable) == joblib.hash(mutable)


@pytest.mark.parametrize("mutable", [True, False])
def test_clone_namedtuple(mutable):
    """Test that objects with namedtuples are cloned correctly.

    We expect a deepcopy of the named tuple and all its elements
    """
    # create a new namedtuple
    namedtuple_class = namedtuple("namedtuple_class", ["a", "b"])
    # Create named tuple with either mutable or imutable elements
    if mutable:
        namedtuple_instance = namedtuple_class(a={"a": "a"}, b={"b": "b"})
    else:
        # Note: we use tuples here, as they are immutable, but we can still use `is` to check the mem address
        namedtuple_instance = namedtuple_class(a=("a",), b=("b",))

    test_class = create_test_class("test", params={"namedtuple": namedtuple_instance})

    # Test clone with instance
    # We expect that the cloned namedtuple has the same content as the original, but the memory address is different
    cloned_class = clone(test_class)
    cloned_named_tuble = cloned_class.namedtuple
    assert namedtuple_instance is not cloned_named_tuble
    assert namedtuple_instance.a is not cloned_named_tuble.a
    assert namedtuple_instance.b is not cloned_named_tuble.b
    assert namedtuple_instance.a == cloned_named_tuble.a
    assert namedtuple_instance.b == cloned_named_tuble.b


def test_mutable_default_nested_objects_error():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})

    with pytest.raises(MutableDefaultsError):
        create_test_class("test", params={"normal": "n1", "mutable": nested_instance}).get_params()

    # When wrapped in default, no error is raised
    create_test_class("mutable", params={"normal": "n1", "mutable": cf(nested_instance)}).get_params()


def test_nested_mutable_algorithm_copy():
    """When a object is wrapped with default, it will be copied on init."""
    nested_params = {"nested1": "n1", "nested2": "n2"}
    nested_instance = create_test_class("nested", params=nested_params)

    test_instance = create_test_class("mutable", params={"normal": "n1", "mutable": cf(nested_instance)})

    assert nested_instance is not test_instance.mutable
    assert (
        joblib.hash(test_instance.mutable.get_params())
        == joblib.hash(nested_instance.get_params())
        == joblib.hash({k: f for k, f in nested_params.items()})
    )


def test_invalid_parameter_names():
    class Test(Algorithm):
        def __init__(self, invalid__name):
            self.invalid__name = invalid__name

    with pytest.raises(ValidationError) as e:
        Test(invalid__name="test").get_params()

    assert "double-underscore" in str(e)


@patch("tpcp._base._validate_parameter", wraps=_validate_parameter)
def test_processing_is_only_run_once_on_get_params(mock_process):
    class Test(Algorithm):
        def __init__(self, name):
            self.name = name

    Test("test").get_params()
    Test("test2").get_params()

    assert _get_tpcp_validated(Test) is True
    assert mock_process.call_count == 1


@patch("tpcp._base._validate_parameter", wraps=_validate_parameter)
def test_processing_is_correctly_called_on_all_subclasses(mock_process):
    class Test(Algorithm):
        def __init__(self, name):
            self.name = name

    class Test2(Test):
        pass

    Test("test").get_params()
    assert _get_tpcp_validated(Test2) is False
    Test2("test").get_params()
    assert _get_tpcp_validated(Test2) is True

    assert mock_process.call_count == 2


@patch("tpcp._base._validate_parameter", wraps=_validate_parameter)
def test_processing_is_correctly_called_on_all_subclasses_2(mock_process):
    class Test(Algorithm):
        def __init__(self, name):
            self.name = name

    # We force processing, before the second class is created
    Test("test").get_params()

    class Test2(Test):
        pass

    assert _get_tpcp_validated(Test2) is False
    Test2("test").get_params()
    assert _get_tpcp_validated(Test2) is True

    assert mock_process.call_count == 2


def test_processing_works_with_class_without_init():
    class Test(Algorithm):
        pass

    Test().get_params()


def test_basic_dataclass_support():
    @dataclasses.dataclass
    class Test(Algorithm):
        a: OptiPara[int]
        b: int

    Test(a=1, b=2)

    class Test2(Test):
        c: OptiPara[int]

    DTest = dataclasses.dataclass(Test2)
    assert dataclasses.is_dataclass(DTest)
    dtest = DTest(a=1, b=2, c=3)
    assert dtest.get_params() == {"a": 1, "b": 2, "c": 3}


def test_dataclass_work_with_custom_annot():
    @dataclasses.dataclass
    class Test(Algorithm):
        a: OptiPara[int]
        b: int

    Test(a=1, b=2)

    @dataclasses.dataclass
    class Test2(Algorithm):
        c: Para[Test]
        c__a: ClassVar[OptiPara[int]]

    assert dataclasses.is_dataclass(Test2)
    dtest = Test2(c=Test(a=1, b=2))
    assert dtest.__field_annotations__ == {"c": _ParaTypes.SIMPLE, "c__a": _ParaTypes.OPTI}
    paras = dtest.get_params()
    paras.pop("c")
    assert paras == {"c__a": 1, "c__b": 2}


def test_dataclass_with_nested_para_is_invalid():
    class Test(Algorithm):
        c__a: int = 3

    with pytest.raises(ValidationError):
        dataclasses.dataclass(Test)().get_params()


def test_dataclass_warns_when_cf_is_used():
    @dataclasses.dataclass
    class Test(Algorithm):
        b: int
        a: OptiPara[int] = cf("test")

    with pytest.raises(ValidationError) as e:
        Test(b=2).get_params()


def test_self_optimize_calls_with_info():
    class OnlyOptimize(OptimizablePipeline):
        def self_optimize(self, dataset, **kwargs):
            self.result_ = "optimized"
            return self

    test = OnlyOptimize()
    test.self_optimize_with_info(None)

    assert test.result_ == "optimized"


def test_with_info_calls_self_optimize():
    class OnlyWithInfo(OptimizablePipeline):
        def self_optimize_with_info(self, dataset, **kwargs):
            self.result_ = "optimized"
            return self, None

    test = OnlyWithInfo()
    test.self_optimize(None)

    assert test.result_ == "optimized"
