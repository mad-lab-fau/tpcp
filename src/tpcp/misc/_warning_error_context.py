from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class _ContextFrame:
    name: str
    metadata: dict[str, Any]

    def render(self) -> str:
        if not self.metadata:
            return self.name
        metadata = ", ".join(f"{key}={value!r}" for key, value in self.metadata.items())
        return f"{self.name}: {metadata}"


_context_stack: ContextVar[tuple[_ContextFrame, ...]] = ContextVar("tpcp_warning_error_context", default=())
_warn = warnings.warn


def _render_context_stack() -> str:
    return " > ".join(frame.render() for frame in _context_stack.get())


def _warn_with_context(
    message: Union[Warning, str],
    category: Optional[type[Warning]] = None,
    stacklevel: int = 1,
    source: Any = None,
) -> None:
    context = _render_context_stack()
    if not context:
        _warn(message, category=category, stacklevel=stacklevel, source=source)
        return

    if isinstance(message, Warning):
        category = category or type(message)
        message = str(message)

    _warn(f"[{context}] {message}", category=category, stacklevel=stacklevel + 1, source=source)


warnings.warn = _warn_with_context


def _add_note(exc: BaseException, note: str) -> None:
    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(note)
        return

    notes = getattr(exc, "__notes__", [])
    notes.append(note)
    exc.__notes__ = notes


@contextmanager
def warning_error_context(name: str, /, **metadata: Any) -> Generator[None, None, None]:
    """Add structured context information to warnings and exceptions raised in the context."""
    frame = _ContextFrame(name=name, metadata=metadata)
    token = _context_stack.set((*_context_stack.get(), frame))
    try:
        yield
    except BaseException as exc:
        _add_note(exc, f"Context: {frame.render()}")
        raise
    finally:
        _context_stack.reset(token)
