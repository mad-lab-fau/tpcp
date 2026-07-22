from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

T = TypeVar("T")
_WarningErrorContextFactory = Callable[..., AbstractContextManager[None]]


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except BaseException as exc:  # noqa: BLE001 - diagnostic rendering must never replace the original event
        return f"<repr failed: {type(exc).__name__}>"


def _render_context_values(context: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={_safe_repr(value)}" for key, value in context.items())


def _safe_exception_description(exc: BaseException) -> str:
    try:
        message = str(exc)
    except BaseException as render_exc:  # noqa: BLE001 - diagnostic rendering must never replace the original event
        message = f"<str failed: {type(render_exc).__name__}>"
    return f"{type(exc).__name__}: {message}"


@dataclass(frozen=True)
class _ContextFrame:
    name: str
    context: dict[str, Any]
    context_provider: Optional[Callable[[], Mapping[str, Any]]] = None

    def _render_context(self) -> str:
        context = self.context
        if self.context_provider is not None:
            try:
                provided_context = self.context_provider()
                if not isinstance(provided_context, Mapping):
                    raise TypeError("context_provider must return a mapping")
                duplicate_keys = context.keys() & provided_context.keys()
                if duplicate_keys:
                    duplicates = ", ".join(sorted(duplicate_keys))
                    raise ValueError(f"context_provider returned duplicate keys: {duplicates}")
                context = {**context, **provided_context}
            except BaseException as exc:  # noqa: BLE001 - context rendering must not mask the original event
                provider_error = _safe_exception_description(exc)
                rendered_context = _render_context_values(context)
                rendered_error = f"context_provider_error={provider_error!r}"
                return ", ".join(filter(None, (rendered_context, rendered_error)))

        return _render_context_values(context)

    def render(self) -> str:
        rendered_context = self._render_context()
        if not rendered_context:
            return self.name
        return f"{self.name}: {rendered_context}"


_context_stack: ContextVar[tuple[_ContextFrame, ...]] = ContextVar("tpcp_warning_error_context", default=())


def _render_context_stack() -> str:
    return " > ".join(frame.render() for frame in _context_stack.get())


def _render_contexts(contexts: Sequence[tuple[str, dict[str, Any]]]) -> str:
    return " > ".join(_ContextFrame(name=name, context=context).render() for name, context in contexts)


def _warning_message_with_context(message: str, context: str) -> str:
    return f"{message}\n[{context}] {message}"


def _warning_with_context(message: Warning, context: str) -> Warning:
    # Calling the concrete type's constructor is not safe: warning subclasses can
    # require extra arguments in __new__. BaseException can allocate every Warning
    # subtype without invoking the subtype's constructor.
    contextualized_warning = BaseException.__new__(type(message))
    contextualized_warning.__dict__.update(getattr(message, "__dict__", {}))
    for base_class in type(message).__mro__:
        slots = base_class.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot_name in slots:
            if slot_name in {"__dict__", "__weakref__"}:
                continue
            actual_slot_name = slot_name
            if slot_name.startswith("__") and not slot_name.endswith("__"):
                actual_slot_name = f"_{base_class.__name__.lstrip('_')}{slot_name}"
            slot_descriptor = base_class.__dict__[actual_slot_name]
            try:
                slot_value = slot_descriptor.__get__(message, type(message))
            except AttributeError:
                continue
            slot_descriptor.__set__(contextualized_warning, slot_value)
    contextualized_warning.args = (_warning_message_with_context(str(message), context), *message.args[1:])
    return contextualized_warning


def _contextualize_warning(message: Union[Warning, str]) -> Union[Warning, str]:
    context = _render_context_stack()
    if not context:
        return message
    if isinstance(message, str):
        return _warning_message_with_context(message, context)
    return _warning_with_context(message, context)


_TPCP_DISPATCHER_MARKER = "_tpcp_warning_context_dispatcher"
_TPCP_ORIGINAL_DISPATCHER = "_tpcp_warning_context_original_dispatcher"

_installed_showwarnmsg = getattr(warnings, "_showwarnmsg", None)
if getattr(_installed_showwarnmsg, _TPCP_DISPATCHER_MARKER, False):
    # Module reloads must unwrap our existing dispatcher instead of chaining it
    # again. Otherwise the old function resolves reloaded module globals and recurses.
    _original_showwarnmsg = getattr(_installed_showwarnmsg, _TPCP_ORIGINAL_DISPATCHER)
else:
    _original_showwarnmsg = _installed_showwarnmsg
_original_showwarning = warnings.showwarning


def _showwarnmsg_with_context(message: warnings.WarningMessage) -> None:
    """Add active context before forwarding a warning to Python's dispatcher."""
    if _context_stack.get():
        message = copy(message)
        message.message = _contextualize_warning(message.message)

    # Patch this private dispatcher intentionally: warnings.warn misses aliases
    # imported earlier, warn_explicit, and C warnings, while showwarning is bypassed
    # by catch_warnings(record=True) and pytest.warns. Forwarding the WarningMessage
    # exactly once avoids re-filtering, recursion, and distorted source locations.
    original_showwarnmsg = getattr(_showwarnmsg_with_context, _TPCP_ORIGINAL_DISPATCHER)
    original_showwarnmsg(message)


if callable(_original_showwarnmsg):
    setattr(_showwarnmsg_with_context, _TPCP_DISPATCHER_MARKER, True)
    setattr(_showwarnmsg_with_context, _TPCP_ORIGINAL_DISPATCHER, _original_showwarnmsg)
    setattr(warnings, "_showwarnmsg", _showwarnmsg_with_context)
else:

    def _showwarning_with_context(
        message: Warning,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: Optional[str] = None,
    ) -> None:
        """Fallback for Python implementations without warnings._showwarnmsg."""
        _original_showwarning(_contextualize_warning(message), category, filename, lineno, file, line)

    warnings.showwarning = _showwarning_with_context


def _add_note(exc: BaseException, note: str) -> None:
    try:
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(note)
            return
    except BaseException:  # noqa: BLE001 - diagnostic context must never replace the original exception
        pass

    try:
        notes = list(getattr(exc, "__notes__", []))
        if not notes or notes[-1] != note:
            notes.append(note)
        setattr(exc, "__notes__", notes)
    except BaseException:  # noqa: BLE001 - re-raising the original exception is more important than its note
        pass


@contextmanager
def warning_error_context(
    name: str,
    context: Optional[dict[str, Any]] = None,
    /,
    *,
    context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Generator[None, None, None]:
    """Add structured context information to warnings and exceptions raised in the context.

    If provided, ``context`` is copied when the context is entered.
    ``context_provider`` is evaluated whenever context is rendered, allowing
    diagnostics to include state that changes while the context is active. It can be
    used without a fixed context dictionary. The provider must return a mapping and
    must not repeat fixed context keys.

    A failing provider is represented in the rendered context instead of masking the
    warning or exception that caused context rendering.

    Python applies warning filters before this context manager attaches its metadata.
    Consequently, different context values do not make otherwise identical warnings
    count as distinct occurrences. If every contextual occurrence must be shown, use
    an appropriate warning filter such as ``warnings.simplefilter("always")``.

    On Python 3.9 and 3.10, exception context is stored in ``__notes__`` but is not
    displayed by Python's standard traceback renderer. Traceback renderers that
    support exception notes, such as Rich, display this context on those versions.
    """
    frame = _ContextFrame(
        name=name, context=dict({} if context is None else context), context_provider=context_provider
    )
    token = _context_stack.set((*_context_stack.get(), frame))
    try:
        yield
    except BaseException as exc:
        _add_note(exc, f"Context: {frame.render()}")
        raise
    finally:
        _context_stack.reset(token)


def _make_iteration_context_factory(i: int) -> _WarningErrorContextFactory:
    def make_context(
        name: str,
        context: Optional[dict[str, Any]] = None,
        /,
        *,
        context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> AbstractContextManager[None]:
        context = {} if context is None else context
        if "i" in context:
            raise ValueError("The context key 'i' is reserved by iter_with_warning_error_context.")
        return warning_error_context(name, {"i": i, **context}, context_provider=context_provider)

    return make_context


def iter_with_warning_error_context(
    iterable: Iterable[T],
) -> Iterator[tuple[_WarningErrorContextFactory, T]]:
    """Pair every item with a warning/error context creator.

    The creator has the same interface as :func:`warning_error_context`, with the
    zero-based iteration index added as ``i``. It does not infer any other context
    from the item. Enter it explicitly in the loop body so the context is closed
    before advancing or suspending the iterator.
    """
    for i, item in enumerate(iterable):
        yield _make_iteration_context_factory(i), item
