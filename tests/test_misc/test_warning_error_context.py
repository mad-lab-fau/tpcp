"""Tests for warning and exception context metadata."""

import _warnings
import inspect
import re
import subprocess
import sys
import warnings
from textwrap import dedent

import pytest
from joblib.externals import cloudpickle

_warn_alias_imported_before_tpcp = warnings.warn

from tpcp.misc import (  # noqa: E402  (import must happen after capturing warnings.warn)
    iter_with_warning_error_context,
    warning_error_context,
)


class _CustomWarning(UserWarning):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail


class _RequiredNewSlotWarning(UserWarning):
    __slots__ = ("detail",)

    def __new__(cls, message, detail):
        instance = super().__new__(cls, message)
        instance.detail = detail
        return instance

    def __init__(self, message, detail):
        super().__init__(message)
        self.payload = {"detail": detail}


def _emit_warning():
    warnings.warn("low level warning", UserWarning, stacklevel=1)


def _emit_warning_with_line():
    lineno = inspect.currentframe().f_lineno + 1
    warnings.warn("location warning", UserWarning, stacklevel=1)
    return lineno


def test_nested_contexts_are_added_to_warnings():
    """Nested contexts are rendered in emitted warning messages."""
    with (
        warning_error_context("datapoint", {"group": "patient-1"}),
        warning_error_context("region", {"start": 10, "end": 20}),
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
        warning_error_context("datapoint", {"group": "patient-1"}),
        warning_error_context("region", {"start": 10, "end": 20}),
    ):
        raise ValueError("failed")

    assert error.value.args == ("failed",)
    assert error.value.__notes__ == [
        "Context: region: start=10, end=20",
        "Context: datapoint: group='patient-1'",
    ]


def test_iter_with_warning_error_context_scopes_nested_loop_bodies():
    """Iterator contexts nest explicitly and do not leak after a loop body."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for make_outer_context, outer_item in iter_with_warning_error_context([1]):
            with make_outer_context("outer", {"item": outer_item}):
                for make_inner_context, inner_item in iter_with_warning_error_context([2]):
                    with make_inner_context("inner", {"item": inner_item}):
                        warnings.warn("inner warning", UserWarning, stacklevel=1)
                warnings.warn("outer warning", UserWarning, stacklevel=1)
        warnings.warn("unscoped warning", UserWarning, stacklevel=1)

    assert [str(warning.message) for warning in caught] == [
        "[outer: i=0, item=1 > inner: i=0, item=2] inner warning",
        "[outer: i=0, item=1] outer warning",
        "unscoped warning",
    ]


def test_iter_with_warning_error_context_injects_a_stable_iteration_index():
    """Every yielded creator retains its own zero-based index."""
    factories = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for make_context, item in iter_with_warning_error_context(["first", "second"]):
            factories.append(make_context)
            with make_context("item", {"value": item}):
                warnings.warn("item warning", UserWarning, stacklevel=1)

        with factories[0]("saved", context_provider=lambda: {"dynamic": True}):
            warnings.warn("saved warning", UserWarning, stacklevel=1)

    assert [str(warning.message) for warning in caught] == [
        "[item: i=0, value='first'] item warning",
        "[item: i=1, value='second'] item warning",
        "[saved: i=0, dynamic=True] saved warning",
    ]


def test_warning_error_context_copies_the_explicit_context_dict():
    """Mutating the input dictionary does not change an active fixed context."""
    context = {"value": "initial"}
    with warning_error_context("copy", context), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        context["value"] = "changed"
        warnings.warn("copied warning", UserWarning, stacklevel=1)

    assert str(caught[0].message) == "[copy: value='initial'] copied warning"


def test_context_provider_is_resolved_for_each_warning():
    """Dynamic metadata reflects state at the time each warning is rendered."""
    state = {"step": 1}

    with (
        warning_error_context(
            "iteration",
            {"fixed": "value"},
            context_provider=lambda: {"step": state["step"]},
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        warnings.warn("first", UserWarning, stacklevel=1)
        state["step"] = 2
        warnings.warn("second", UserWarning, stacklevel=1)

    assert [str(warning.message) for warning in caught] == [
        "[iteration: fixed='value', step=1] first",
        "[iteration: fixed='value', step=2] second",
    ]


def test_context_provider_can_be_used_without_fixed_context():
    """A dynamic provider does not require an empty context dictionary."""
    with (
        warning_error_context("iteration", context_provider=lambda: {"step": 1}),
        pytest.warns(UserWarning) as caught,
    ):
        warnings.warn("provider-only warning", UserWarning, stacklevel=1)

    assert str(caught[0].message) == "[iteration: step=1] provider-only warning"


def test_context_provider_is_resolved_when_exception_leaves_context():
    """Exception notes contain dynamic metadata from the point of failure."""
    state = {"step": 1}

    def fail_after_state_change():
        state["step"] = 2
        raise ValueError("failed")

    with (
        pytest.raises(ValueError, match="failed") as error,
        warning_error_context("iteration", {}, context_provider=lambda: {"step": state["step"]}),
    ):
        fail_after_state_change()

    assert error.value.__notes__ == ["Context: iteration: step=2"]


@pytest.mark.parametrize(
    ("context_provider", "error_text"),
    [
        (lambda: 1, "TypeError: context_provider must return a mapping"),
        (lambda: {"fixed": "dynamic"}, "ValueError: context_provider returned duplicate keys: fixed"),
        (lambda: 1 / 0, "ZeroDivisionError: division by zero"),
    ],
    ids=["not-a-mapping", "duplicate-key", "provider-raises"],
)
def test_context_provider_failure_does_not_mask_warning(context_provider, error_text):
    """Invalid dynamic metadata is reported without replacing the warning."""
    with (
        warning_error_context("iteration", {"fixed": "value"}, context_provider=context_provider),
        pytest.warns(UserWarning, match="original warning") as caught,
    ):
        warnings.warn("original warning", UserWarning, stacklevel=1)

    assert str(caught[0].message) == (
        f"[iteration: fixed='value', context_provider_error={error_text!r}] original warning"
    )


def test_context_provider_failure_does_not_mask_exception():
    """A provider failure is diagnostic context on the original exception."""
    with (
        pytest.raises(RuntimeError, match="original error") as error,
        warning_error_context("iteration", {}, context_provider=lambda: 1 / 0),
    ):
        raise RuntimeError("original error")

    assert error.value.__notes__ == ["Context: iteration: context_provider_error='ZeroDivisionError: division by zero'"]


def test_context_value_repr_failure_does_not_mask_warning():
    """An unrenderable context value does not replace the original warning."""

    class BrokenRepr:
        def __repr__(self):
            raise RuntimeError("repr failed")

    with (
        warning_error_context("iteration", {"value": BrokenRepr()}),
        pytest.warns(UserWarning, match="original warning") as caught,
    ):
        warnings.warn("original warning", UserWarning, stacklevel=1)

    assert str(caught[0].message) == "[iteration: value=<repr failed: RuntimeError>] original warning"


def test_context_provider_error_str_failure_does_not_mask_warning():
    """An unrenderable provider error does not replace the original warning."""

    class BrokenStrError(RuntimeError):
        def __str__(self):
            raise RuntimeError("str failed")

    def broken_provider():
        raise BrokenStrError

    with (
        warning_error_context("iteration", context_provider=broken_provider),
        pytest.warns(UserWarning, match="original warning") as caught,
    ):
        warnings.warn("original warning", UserWarning, stacklevel=1)

    assert str(caught[0].message) == (
        "[iteration: context_provider_error='BrokenStrError: <str failed: RuntimeError>'] original warning"
    )


def test_context_is_cleaned_up_after_exception():
    """Contexts do not leak after an exception leaves the context manager."""
    with pytest.raises(ValueError, match="failed"), warning_error_context("datapoint", {"group": "patient-1"}):
        raise ValueError("failed")

    with pytest.warns(UserWarning, match="low level warning") as warning:
        _emit_warning()

    assert str(warning[0].message) == "low level warning"


def test_sibling_contexts_do_not_leak_into_each_other():
    """Sibling contexts under the same parent are pushed and popped independently."""
    with warning_error_context("parent", {"item": "recording-1"}):
        with (
            warning_error_context("child", {"item": "first"}),
            pytest.warns(
                UserWarning,
                match=re.escape("[parent: item='recording-1' > child: item='first'] low level warning"),
            ),
        ):
            _emit_warning()

        with (
            warning_error_context("child", {"item": "second"}),
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
        warning_error_context("datapoint", {"group": "patient-1"}),
        pytest.warns(
            _CustomWarning,
            match=re.escape("[datapoint: group='patient-1'] custom warning"),
        ) as warning,
    ):
        warnings.warn(_CustomWarning("custom warning", detail="kept"), stacklevel=1)

    assert warning[0].message.detail == "kept"


def test_warning_instances_with_required_new_arguments_and_slots_keep_their_type_and_data():
    """Cloning a warning does not invoke its potentially custom constructor."""
    original_warning = _RequiredNewSlotWarning("custom warning", detail="kept")

    with (
        warning_error_context("datapoint", {"group": "patient-1"}),
        pytest.warns(
            _RequiredNewSlotWarning,
            match=re.escape("[datapoint: group='patient-1'] custom warning"),
        ) as warning,
    ):
        warnings.warn(original_warning, stacklevel=1)

    assert len(warning) == 1
    assert warning[0].category is _RequiredNewSlotWarning
    assert type(warning[0].message) is _RequiredNewSlotWarning
    assert warning[0].message.detail == "kept"
    assert warning[0].message.payload == {"detail": "kept"}
    assert str(original_warning) == "custom warning"


@pytest.mark.parametrize(
    "emit_warning",
    [
        lambda: _warn_alias_imported_before_tpcp("aliased warning", UserWarning, stacklevel=1),
        lambda: warnings.warn_explicit("explicit warning", UserWarning, "synthetic_warning_source.py", 123),
        lambda: _warnings.warn("C warning", UserWarning, stacklevel=1),
    ],
    ids=["pre-imported-warn-alias", "warn-explicit", "C-warnings-api"],
)
def test_context_is_added_for_all_warning_emitters(emit_warning):
    """The dispatcher sees warning sources that do not call the current warnings.warn binding."""
    with warning_error_context("datapoint", {"item": 3}), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_warning()

    assert len(caught) == 1
    assert str(caught[0].message).startswith("[datapoint: item=3] ")
    assert caught[0].category is UserWarning


def test_warn_explicit_location_is_preserved():
    """Contextualizing an explicit warning keeps its synthetic location."""
    with warning_error_context("datapoint", {"item": 3}), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn_explicit("explicit warning", UserWarning, "synthetic_warning_source.py", 123)

    assert len(caught) == 1
    assert caught[0].filename == "synthetic_warning_source.py"
    assert caught[0].lineno == 123


def test_warning_message_is_forwarded_exactly_once(monkeypatch):
    """The dispatcher forwards instead of re-emitting the warning."""
    forwarded_messages = []
    monkeypatch.setattr(warnings, "_showwarnmsg_impl", forwarded_messages.append)

    with warning_error_context("datapoint", {"item": 3}):
        warnings.warn_explicit("explicit warning", UserWarning, "synthetic_warning_source.py", 123)

    assert len(forwarded_messages) == 1
    assert str(forwarded_messages[0].message) == "[datapoint: item=3] explicit warning"
    assert forwarded_messages[0].filename == "synthetic_warning_source.py"
    assert forwarded_messages[0].lineno == 123


def test_warning_hook_installation_is_reload_safe():
    """Reloading the module does not recurse or leave an inconsistent hook."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent(
                """
                import importlib
                import warnings

                module = importlib.import_module("tpcp.misc._warning_error_context")
                reloaded_module = importlib.reload(module)
                with (
                    reloaded_module.warning_error_context("reload", {"attempt": 1}),
                    warnings.catch_warnings(record=True) as caught,
                ):
                    warnings.simplefilter("always")
                    warnings.warn("reloaded warning", UserWarning, stacklevel=1)

                assert len(caught) == 1
                assert str(caught[0].message) == "[reload: attempt=1] reloaded warning"
                assert warnings._showwarnmsg is reloaded_module._showwarnmsg_with_context
                """
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_warning_hook_chains_to_preexisting_dispatcher_on_reload():
    """Installation preserves a dispatcher that was not installed by tpcp."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent(
                """
                import importlib
                import warnings

                module = importlib.import_module("tpcp.misc._warning_error_context")
                hook_before_reload = warnings._showwarnmsg
                original_attribute = module._TPCP_ORIGINAL_DISPATCHER
                downstream_dispatcher = getattr(hook_before_reload, original_attribute, hook_before_reload)
                third_party_messages = []

                def third_party_dispatcher(message):
                    third_party_messages.append(message)
                    downstream_dispatcher(message)

                warnings._showwarnmsg = third_party_dispatcher
                reloaded_module = importlib.reload(module)
                with (
                    reloaded_module.warning_error_context("third-party", {}),
                    warnings.catch_warnings(record=True) as caught,
                ):
                    warnings.simplefilter("always")
                    warnings.warn("chained warning", UserWarning, stacklevel=1)

                assert len(third_party_messages) == 1
                assert str(third_party_messages[0].message) == "[third-party] chained warning"
                assert len(caught) == 1
                """
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_warning_dispatcher_is_cloudpickleable():
    """The global hook remains import-addressable for multiprocessing workers."""
    dispatcher = warnings._showwarnmsg

    restored_dispatcher = cloudpickle.loads(cloudpickle.dumps(dispatcher))

    assert restored_dispatcher is dispatcher


def test_warning_location_is_preserved_with_context():
    """Adding context does not change a stacklevel-derived warning location."""
    with (
        warning_error_context("datapoint", {"item": 3}),
        pytest.warns(UserWarning, match="location warning") as warning,
    ):
        lineno = _emit_warning_with_line()

    assert warning[0].filename == __file__
    assert warning[0].lineno == lineno


def test_warning_location_is_preserved_without_context():
    """The global warning wrapper does not change locations for unrelated warnings."""
    with pytest.warns(UserWarning, match="location warning") as warning:
        lineno = _emit_warning_with_line()

    assert warning[0].filename == __file__
    assert warning[0].lineno == lineno
