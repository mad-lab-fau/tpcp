# ruff: noqa: D205, D400
"""
Warning and Error Context
=========================

The :func:`~tpcp.misc.warning_error_context` helper adds structured context to
warnings and exceptions without changing the code that emits them. This is
useful when a low-level warning or error does not know which dataset item,
optimization trial, or iteration triggered it.

This example covers fixed and dynamic context, nested contexts, retained event
records, contextual prints, manual lifecycle control, quiet record-only
execution, arbitrary iterators, :class:`~tpcp.misc.TypedIterator`, and typed
sub-iterations.

Simple and dynamic context
--------------------------
Pass fixed values as a dictionary. Warning messages retain their original
warning type and source location. The original message is followed by a
contextual copy on a separate line.
"""

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from tpcp.misc import (
    BaseTypedIterator,
    TypedIterator,
    iter_with_warning_error_context,
    print_with_context,
    warning_error_context,
)

with warning_error_context("record", {"participant": "p1"}):
    warnings.warn("signal is short", UserWarning, stacklevel=1)

# %%
# A ``context_provider`` is evaluated only when a warning or exception is
# rendered. It can be used without passing an empty context dictionary. This is
# useful for state that changes while the context is active.
state = {"step": 0}

with warning_error_context(
    "processing",
    context_provider=lambda: {"step": state["step"]},
):
    state["step"] = 2
    warnings.warn("processing warning", UserWarning, stacklevel=1)

# %%
# Fixed context and a provider can also be combined. Provider keys must not
# duplicate keys from the fixed dictionary.
with warning_error_context(
    "processing",
    {"participant": "p1"},
    context_provider=lambda: {"step": state["step"]},
):
    warnings.warn("combined warning", UserWarning, stacklevel=1)

# %%
# Nested contexts
# ---------------
# Contexts compose from outermost to innermost. Each context is removed when its
# ``with`` block ends.
with warning_error_context("participant", {"id": "p1"}):  # noqa: SIM117 - Python 3.9 compatibility
    with warning_error_context("recording", {"id": 3}):
        warnings.warn("nested warning", UserWarning, stacklevel=1)

# %%
# Errors
# ------
# The context manager re-raises the original error, so it can be caught and
# handled normally. The active context is attached to the caught exception
# through exception notes. On Python 3.11 and newer, the standard traceback
# renderer displays these notes. On Python 3.9 and 3.10, renderers with
# exception-note support, such as Rich, can display them.
try:
    with warning_error_context("recording", {"id": 3}):
        raise ValueError("invalid signal")
except ValueError as error:
    print(f"Caught error: {error}")
    print(f"Context: {getattr(error, '__notes__', [])}")

# %%
# Manual lifecycle
# ----------------
# For a long top-level script, the object returned by
# :func:`~tpcp.misc.warning_error_context` can be started and stopped explicitly
# to avoid indenting the entire guarded section.
manual_context = warning_error_context("script", {"phase": "processing"})
manual_context.start()
warnings.warn("manual warning", UserWarning, stacklevel=1)
print_with_context("Manual context is active")
manual_context.stop()

print([record.type for record in manual_context.records])

# %%
# Manual lifecycle control has an important exception caveat. ``stop()`` does
# not receive an active exception, so it cannot record or annotate one. If an
# exception prevents ``stop()`` from running, the context also remains active.
# A ``try``/``finally`` can guarantee cleanup, but the bare ``stop()`` call
# still cannot add exception context. Use the ``with`` form whenever exceptions
# must be recorded, annotated, or cleanly unwind the context.

# %%
# Recording events and contextual prints
# ---------------------------------------
# Every context manager yields a persistent result whose ``records`` list can
# be inspected after the context exits. Each named-tuple record contains the
# event type (``"warning"``, ``"error"``, or ``"print"``), the fully resolved
# context stack, and the original warning/exception object or print message.
# Nested events are included in every active context result.
#
# :func:`~tpcp.misc.print_with_context` accepts the normal ``print`` arguments.
# With an active context, it prepends the rendered context and adds a record.
# Without an active context, it behaves like a normal ``print``.
items = ["ok", "short", "invalid"]

with warnings.catch_warnings():
    # Warning filters run before tpcp can record a warning. Use ``always`` when
    # every repeated occurrence is relevant to the later analysis.
    warnings.simplefilter("always")
    with warning_error_context(
        "dataset", {"name": "validation"}
    ) as dataset_context:
        for make_context, item in iter_with_warning_error_context(items):
            try:
                with make_context("datapoint", {"value": item}):
                    print_with_context("Processing", item)
                    if item != "ok":
                        warnings.warn(
                            "signal quality issue", UserWarning, stacklevel=1
                        )
                    if item == "invalid":
                        raise ValueError("invalid signal")
            except ValueError:
                # Errors are recorded when they leave a context. They still
                # propagate normally and can be handled by the application.
                pass

print([record.type for record in dataset_context.records])

warning_contexts = [
    record.context
    for record in dataset_context.records
    if record.type == "warning" and isinstance(record.message, UserWarning)
]
print(warning_contexts)

# %%
# Quiet record-only mode
# ----------------------
# A high-level ``record_only=True`` context suppresses contextual warning and
# ``print_with_context`` output from all nested contexts while retaining the
# same records. It does not suppress errors or ordinary ``print`` calls. This is
# useful for wrapping a long-running program and printing only a final summary.
with warnings.catch_warnings():
    warnings.simplefilter("always")
    with warning_error_context(
        "program", {"mode": "batch"}, record_only=True
    ) as program_context:
        for make_context, item in iter_with_warning_error_context(
            ["left", "right"]
        ):
            with make_context("datapoint", {"value": item}):
                warnings.warn("quiet warning", UserWarning, stacklevel=1)
                print_with_context("Processed", item)

print([record.type for record in program_context.records])

# %%
# Arbitrary iterators
# -------------------
# :func:`~tpcp.misc.iter_with_warning_error_context` pairs each item with a
# context creator. Enter the returned context explicitly inside the loop body.
# The creator automatically adds the zero-based iteration index as ``i``; all
# other values remain explicit.
#
# Python normally suppresses repeated warnings from the same source line. This
# filtering happens before context is attached, so different context values do
# not make the warnings distinct. Applications that need every contextual
# occurrence can enable an appropriate warning filter. Here, ``catch_warnings``
# only scopes that filter; warnings still emit normally for Sphinx-Gallery.
with warnings.catch_warnings():
    warnings.simplefilter("always")
    for make_context, item in iter_with_warning_error_context(
        ["left", "right"]
    ):
        with make_context("item", {"value": item}):
            warnings.warn("iterator warning", UserWarning, stacklevel=1)

# %%
# Combining regular and iterator contexts
# ---------------------------------------
# An iterator context composes with any regular context that is already active.
# The regular context is rendered first, followed by the per-iteration context
# and its automatically injected ``i``.
with warnings.catch_warnings():
    warnings.simplefilter("always")
    with warning_error_context("dataset", {"name": "validation"}):
        for make_context, item in iter_with_warning_error_context(
            ["left", "right"]
        ):
            with make_context("item", {"value": item}):
                warnings.warn("stacked warning", UserWarning, stacklevel=1)

# %%
# TypedIterator
# -------------
# :class:`~tpcp.misc.TypedIterator` exposes the same context-creation interface.
# While an item is yielded, ``warning_error_context`` reads the current
# zero-based index from the iterator and injects it as ``i``. No other item
# metadata is inferred.


@dataclass
class Result:
    """Result created for one iteration."""

    doubled: int


typed_iterator = TypedIterator[int, Result](Result)

with warnings.catch_warnings():
    warnings.simplefilter("always")
    for item, result in typed_iterator.iterate([2, 4]):
        with typed_iterator.warning_error_context("item", {"value": item}):
            warnings.warn("typed warning", UserWarning, stacklevel=1)
            result.doubled = item * 2

print(typed_iterator.results_.doubled)

# %%
# TypedIterator sub-iterations
# ----------------------------
# Custom typed iterators can expose sub-iterations by wrapping the protected
# ``_iterate`` helper. The same ``warning_error_context`` method then binds to
# whichever iteration is currently yielding. Every sub-iteration starts its own
# ``i`` at zero, and completing it restores the outer iteration's ``i``.


class NestedIterator(BaseTypedIterator[int, Result]):
    """Typed iterator with an outer iteration and an explicit sub-iteration."""

    def iterate(self, values: Iterable[int]) -> Iterator[tuple[int, Result]]:
        """Iterate over outer values."""
        yield from self._iterate(values)

    def iterate_subitems(
        self, values: Iterable[int]
    ) -> Iterator[tuple[int, Result]]:
        """Iterate over values belonging to the current outer item."""
        yield from self._iterate(values, iteration_name="subitems")


nested_iterator = NestedIterator(Result)

with warnings.catch_warnings():
    warnings.simplefilter("always")
    for outer, outer_result in nested_iterator.iterate([10]):
        with nested_iterator.warning_error_context("outer", {"value": outer}):
            warnings.warn("outer before", UserWarning, stacklevel=1)

        for inner, inner_result in nested_iterator.iterate_subitems([20, 21]):
            with nested_iterator.warning_error_context(
                "inner",
                context_provider=lambda inner=inner: {"value": inner},
            ):
                warnings.warn("inner", UserWarning, stacklevel=1)
                inner_result.doubled = inner * 2

        # No fixed dictionary is required. This context contains only the
        # restored outer ``i``.
        with nested_iterator.warning_error_context("outer_after"):
            warnings.warn("outer after", UserWarning, stacklevel=1)
            outer_result.doubled = outer * 2

# %%
# In all iterator variants, the context manager is created and entered inside
# the loop body. This keeps its lifetime explicit and prevents context from
# leaking across a generator's ``yield`` boundary.
