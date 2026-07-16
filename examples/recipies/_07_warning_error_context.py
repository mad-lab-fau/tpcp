# ruff: noqa: D205, D400
"""
Warning and Error Context
=========================

The :func:`~tpcp.misc.warning_error_context` helper adds structured context to
warnings and exceptions without changing the code that emits them. This is
useful when a low-level warning or error does not know which dataset item,
optimization trial, or iteration triggered it.

This example covers fixed and dynamic context, nested contexts, arbitrary
iterators, :class:`~tpcp.misc.TypedIterator`, and typed sub-iterations.

Simple and dynamic context
--------------------------
Pass fixed values as a dictionary. Warning messages retain their original
warning type and source location, with the active context prepended to the
message.
"""

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from tpcp.misc import (
    BaseTypedIterator,
    TypedIterator,
    iter_with_warning_error_context,
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
# Arbitrary iterators
# -------------------
# :func:`~tpcp.misc.iter_with_warning_error_context` pairs each item with a
# context creator. Enter the returned context explicitly inside the loop body.
# The creator automatically adds the zero-based iteration index as ``i``; all
# other values remain explicit.

for make_context, item in iter_with_warning_error_context(["left", "right"]):
    with make_context("item", {"value": item}):
        warnings.warn("iterator warning", UserWarning, stacklevel=1)

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

    # No fixed dictionary is required. This context contains only the restored
    # outer ``i``.
    with nested_iterator.warning_error_context("outer_after"):
        warnings.warn("outer after", UserWarning, stacklevel=1)
        outer_result.doubled = outer * 2

# %%
# In all iterator variants, the context manager is created and entered inside
# the loop body. This keeps its lifetime explicit and prevents context from
# leaking across a generator's ``yield`` boundary.
