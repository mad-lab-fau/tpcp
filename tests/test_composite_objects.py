import pytest

from tests.test_base import create_test_class
from tpcp import cf
from tpcp.exceptions import ValidationError


def test_composite_deep_get_params():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}

    nested_default = cf(
        [("nc1", nested_instance.clone().set_params(nested1=1)), ("nc2", nested_instance.clone().set_params(nested1=2))]
    )

    test_instance = create_test_class(
        "test",
        params={**top_level_params, "nested_class_list": nested_default},
        private_params={"_composite_params": ("nested_class_list",)},
    )
    params = test_instance.get_params()

    nested_class_list = params["nested_class_list"]
    expected_values = iter((1, 2))
    for k, v in nested_class_list:
        assert isinstance(v, nested_instance.__class__)
        assert isinstance(params[f"nested_class_list__{k}"], nested_instance.__class__)
        expected_value = next(expected_values)
        assert v.nested1 == expected_value
        assert params[f"nested_class_list__{k}"].nested1 == expected_value
        assert params[f"nested_class_list__{k}__nested1"] == expected_value


@pytest.mark.parametrize(
    "values",
    (
        (None, True),
        ([("name", "something")], True),
        ("something", False),
        ([(1, "something")], False),
        ([("name", "something"), (1, "something")], False),
        ([("name", "something", "some_other")], False),
    ),
)
def test_raises_with_invalid_composite(values):
    test_instance = create_test_class(
        "test",
        params={"nested": None},
        private_params={"_composite_params": ("nested",)},
    )
    test_instance.set_params(nested=values[0])
    if values[1]:
        assert test_instance.get_params()["nested"] == values[0]
    else:
        with pytest.raises(ValidationError):
            test_instance.get_params()


def test_set_params_simple():
    test_instance = create_test_class(
        "test",
        params={"nested": None},
        private_params={"_composite_params": ("nested",)},
    )

    test_instance.nested = [("bla", 2)]

    assert test_instance.nested[0][1] == 2
    test_instance.set_params(nested__bla=3)
    assert test_instance.nested[0][1] == 3


def test_set_params_nested_obj():
    test_instance = create_test_class(
        "test",
        params={"nested": None},
        private_params={"_composite_params": ("nested",)},
    )
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})

    test_instance.nested = [("bla", nested_instance)]

    assert test_instance.nested[0][1] == nested_instance
    test_instance.set_params(nested__bla__nested1="changed_value")
    assert test_instance.nested[0][1].nested1 == "changed_value"


def test_set_params_order():
    """This tests that we always set the least nested values first."""
    test_instance = create_test_class(
        "test",
        params={"nested": None},
        private_params={"_composite_params": ("nested",)},
    )
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})

    test_instance.nested = [("bla", nested_instance)]

    assert test_instance.nested[0][1] == nested_instance
    test_instance.set_params(
        nested__bla=nested_instance.clone().set_params(nested1="first_changed_value", nested2="first_changed_value"),
        nested__bla__nested1="changed_value",
    )
    assert test_instance.nested[0][1].nested1 == "changed_value"
    assert test_instance.nested[0][1].nested2 == "first_changed_value"


def test_raises_when_setting_unknown():
    test_instance = create_test_class(
        "test",
        params={"nested": None},
        private_params={"_composite_params": ("nested",)},
    )

    test_instance.nested = [("bla", 2)]

    with pytest.raises(ValueError):
        test_instance.set_params(nested__blub=3)
