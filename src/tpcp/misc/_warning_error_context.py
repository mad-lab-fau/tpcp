from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


@dataclass(frozen=True)
class _ContextFrame:
    name: str
    metadata: dict[str, Any]
    metadata_provider: Optional[Callable[[], Mapping[str, Any]]] = None

    def _render_metadata(self) -> str:
        metadata = self.metadata
        if self.metadata_provider is not None:
            try:
                provided_metadata = self.metadata_provider()
                if not isinstance(provided_metadata, Mapping):
                    raise TypeError("metadata_provider must return a mapping")
                duplicate_keys = metadata.keys() & provided_metadata.keys()
                if duplicate_keys:
                    duplicates = ", ".join(sorted(duplicate_keys))
                    raise ValueError(f"metadata_provider returned duplicate keys: {duplicates}")
                metadata = {**metadata, **provided_metadata}
            except BaseException as exc:  # noqa: BLE001 - context rendering must not mask the original event
                provider_error = f"{type(exc).__name__}: {exc}"
                rendered_metadata = ", ".join(f"{key}={value!r}" for key, value in metadata.items())
                rendered_error = f"metadata_provider_error={provider_error!r}"
                return ", ".join(filter(None, (rendered_metadata, rendered_error)))

        return ", ".join(f"{key}={value!r}" for key, value in metadata.items())

    def render(self) -> str:
        rendered_metadata = self._render_metadata()
        if not rendered_metadata:
            return self.name
        return f"{self.name}: {rendered_metadata}"


_context_stack: ContextVar[tuple[_ContextFrame, ...]] = ContextVar("tpcp_warning_error_context", default=())


def _render_context_stack() -> str:
    return " > ".join(frame.render() for frame in _context_stack.get())


def _render_contexts(contexts: Sequence[tuple[str, dict[str, Any]]]) -> str:
    return " > ".join(_ContextFrame(name=name, metadata=metadata).render() for name, metadata in contexts)


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
    contextualized_warning.args = (f"[{context}] {message}", *message.args[1:])
    return contextualized_warning


def _contextualize_warning(message: Union[Warning, str]) -> Union[Warning, str]:
    context = _render_context_stack()
    if not context:
        return message
    if isinstance(message, str):
        return f"[{context}] {message}"
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
    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(note)
        return

    notes = getattr(exc, "__notes__", [])
    notes.append(note)
    setattr(exc, "__notes__", notes)


@contextmanager
def warning_error_context(
    name: str,
    /,
    *,
    metadata_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    **metadata: Any,
) -> Generator[None, None, None]:
    """Add structured context information to warnings and exceptions raised in the context.

    ``metadata`` values are fixed when the context is entered. ``metadata_provider``
    is evaluated whenever context is rendered, allowing diagnostics to include state
    that changes while the context is active. The provider must return a mapping and
    must not repeat fixed metadata keys.

    A failing provider is represented in the rendered context instead of masking the
    warning or exception that caused context rendering.
    """
    frame = _ContextFrame(name=name, metadata=metadata, metadata_provider=metadata_provider)
    token = _context_stack.set((*_context_stack.get(), frame))
    try:
        yield
    except BaseException as exc:
        _add_note(exc, f"Context: {frame.render()}")
        raise
    finally:
        _context_stack.reset(token)
