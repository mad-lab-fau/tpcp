"""Tests for warning and exception context metadata."""

import re
import warnings

import pytest

from tpcp.misc import warning_error_context


class _CustomWarning(UserWarning):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail


def _emit_warning():
    warnings.warn("low level warning", UserWarning, stacklevel=1)


def test_nested_contexts_are_added_to_warnings():
    """Nested contexts are rendered in emitted warning messages."""
    with (
        warning_error_context("datapoint", group="patient-1"),
        warning_error_context("region", start=10, end=20),
        pytest.warns(
            UserWarning,
            match=re.escape("[datapoint: group='patient-1' > region: start=10, end=20] low level warning"),
        ),
    ):
        _emit_warning()


def test_nested_contexts_are_added_to_exception_notes():
    """Nested contexts are attached as exception notes without changing the exception."""
    with (
        pytest.raises(ValueError, match="failed") as error,
        warning_error_context("datapoint", group="patient-1"),
        warning_error_context("region", start=10, end=20),
    ):
        raise ValueError("failed")

    assert error.value.args == ("failed",)
    assert error.value.__notes__ == [
        "Context: region: start=10, end=20",
        "Context: datapoint: group='patient-1'",
    ]


def test_context_is_cleaned_up_after_exception():
    """Contexts do not leak after an exception leaves the context manager."""
    with pytest.raises(ValueError, match="failed"), warning_error_context("datapoint", group="patient-1"):
        raise ValueError("failed")

    with pytest.warns(UserWarning, match="low level warning") as warning:
        _emit_warning()

    assert str(warning[0].message) == "low level warning"


def test_sibling_contexts_do_not_leak_into_each_other():
    """Sibling contexts under the same parent are pushed and popped independently."""
    with warning_error_context("parent", item="recording-1"):
        with (
            warning_error_context("child", item="first"),
            pytest.warns(
                UserWarning,
                match=re.escape("[parent: item='recording-1' > child: item='first'] low level warning"),
            ),
        ):
            _emit_warning()

        with (
            warning_error_context("child", item="second"),
            pytest.warns(
                UserWarning,
                match=re.escape("[parent: item='recording-1' > child: item='second'] low level warning"),
            ) as warning,
        ):
            _emit_warning()

    assert "first" not in str(warning[0].message)


def test_warning_instances_keep_their_type_and_data():
    """Context metadata does not replace warning instances with plain strings."""
    with (
        warning_error_context("datapoint", group="patient-1"),
        pytest.warns(
            _CustomWarning,
            match=re.escape("[datapoint: group='patient-1'] custom warning"),
        ) as warning,
    ):
        warnings.warn(_CustomWarning("custom warning", detail="kept"), stacklevel=1)

    assert warning[0].message.detail == "kept"
