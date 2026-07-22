from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar, Token
from copy import copy
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, TypeVar, Union
from uuid import uuid4
from weakref import WeakValueDictionary

from tpcp.parallel import _register_tpcp_parallel_side_channel

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class WarningErrorContextRecord(NamedTuple):
    """A warning, error, or contextual print captured by an active context.

    The original warning or exception object is retained as ``message`` so its
    concrete type and custom attributes remain available for later inspection.
    Contextual prints store their unmodified rendered text.

    Attributes
    ----------
    type
        The kind of captured event.
    context
        The fully rendered context stack at the time of the event.
    message
        The original warning or exception object, or the rendered print text.

    """

    type: Literal["warning", "error", "print"]
    context: str
    message: Union[Warning, BaseException, str]


class WarningErrorContext(AbstractContextManager["WarningErrorContext"]):
    """A warning/error context and its persistent event records.

    Attributes
    ----------
    records
        Events observed while the context was active. Events from nested contexts
        are included with their fully rendered context stack.

    """

    def __init__(
        self,
        name: str,
        context: Optional[dict[str, Any]] = None,
        *,
        context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
        record_only: bool = False,
    ) -> None:
        self.records: list[WarningErrorContextRecord] = []
        self._name = name
        self._context = context
        self._context_provider = context_provider
        self._record_only = record_only
        self._frame: Optional[_ContextFrame] = None
        self._token: Optional[Token] = None
        self._started = False
        self._parallel_target_id: Optional[str] = None

    def start(self) -> WarningErrorContext:
        """Activate the context and return this object.

        Use the context-manager form when exceptions must be recorded and
        annotated reliably.
        """
        if self._frame is not None:
            raise RuntimeError("The warning/error context is already active.")
        if self._started:
            raise RuntimeError("The warning/error context cannot be restarted.")

        frame = _ContextFrame(
            name=self._name,
            context=dict({} if self._context is None else self._context),
            context_provider=self._context_provider,
            result=self,
            record_only=self._record_only,
        )
        self._token = _context_stack.set((*_context_stack.get(), frame))
        self._frame = frame
        self._started = True
        return self

    def stop(self) -> None:
        """Deactivate the context without recording an exception."""
        if self._frame is None or self._token is None:
            raise RuntimeError("The warning/error context is not active.")
        stack = _context_stack.get()
        if not stack or stack[-1] is not self._frame:
            raise RuntimeError("Nested warning/error contexts must be stopped in reverse order.")

        is_outermost_context = len(stack) == 1
        _context_stack.reset(self._token)
        self._frame = None
        self._token = None
        if is_outermost_context:
            _recorded_error.set(None)

    def __enter__(self) -> WarningErrorContext:
        return self.start()

    def __exit__(self, _exc_type: Any, exc: Optional[BaseException], _traceback: Any) -> Optional[bool]:
        frame = self._frame
        if frame is None:
            raise RuntimeError("The warning/error context is not active.")
        try:
            if exc is not None:
                if _recorded_error.get() is not exc:
                    _record_event("error", _render_context_stack(), exc)
                    _recorded_error.set(exc)
                _add_note(exc, f"Context: {frame.render()}")
        finally:
            self.stop()
        return None


_WarningErrorContextFactory = Callable[..., WarningErrorContext]


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except BaseException as exc:  # noqa: BLE001 - diagnostic rendering must never replace the original event
        return f"<repr failed: {type(exc).__name__}>"


def _render_context_values(context: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={_safe_repr(value)}" for key, value in context.items())


def _render_named_context(name: str, context: Mapping[str, Any]) -> str:
    rendered_context = _render_context_values(context)
    if not rendered_context:
        return name
    return f"{name}: {rendered_context}"


def _safe_exception_description(exc: BaseException) -> str:
    try:
        message = str(exc)
    except BaseException as render_exc:  # noqa: BLE001 - diagnostic rendering must never replace the original event
        message = f"<str failed: {type(render_exc).__name__}>"
    return f"{type(exc).__name__}: {message}"


def _render_context_with_provider(
    context: dict[str, Any], context_provider: Optional[Callable[[], Mapping[str, Any]]]
) -> str:
    if context_provider is not None:
        try:
            provided_context = context_provider()
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


@dataclass(frozen=True)
class _ContextFrame:
    name: str
    context: dict[str, Any]
    result: WarningErrorContext
    context_provider: Optional[Callable[[], Mapping[str, Any]]] = None
    record_only: bool = False

    def _render_context(self) -> str:
        return _render_context_with_provider(self.context, self.context_provider)

    def render(self) -> str:
        rendered_context = self._render_context()
        return self.name if not rendered_context else f"{self.name}: {rendered_context}"


@dataclass(frozen=True)
class _ParallelContextFrame:
    name: str
    context: dict[str, Any]
    context_provider: Optional[Callable[[], Mapping[str, Any]]]
    record_only: bool
    target_id: str

    def render(self) -> str:
        rendered_context = _render_context_with_provider(self.context, self.context_provider)
        return self.name if not rendered_context else f"{self.name}: {rendered_context}"


_context_stack: ContextVar[tuple[_ContextFrame, ...]] = ContextVar("tpcp_warning_error_context", default=())
_recorded_error: ContextVar[Optional[BaseException]] = ContextVar("tpcp_warning_error_recorded_error", default=None)
_parallel_context_targets: WeakValueDictionary[str, WarningErrorContext] = WeakValueDictionary()


def _render_context_stack() -> str:
    return " > ".join(frame.render() for frame in _context_stack.get())


def _capture_parallel_context() -> Optional[tuple[_ParallelContextFrame, ...]]:
    stack = _context_stack.get()
    if not stack:
        return None
    captured_frames = []
    for frame in stack:
        target_id = frame.result._parallel_target_id
        if target_id is None:
            target_id = uuid4().hex
            frame.result._parallel_target_id = target_id
        _parallel_context_targets[target_id] = frame.result
        captured_frames.append(
            _ParallelContextFrame(
                name=frame.name,
                context=dict(frame.context),
                context_provider=frame.context_provider,
                record_only=frame.record_only,
                target_id=target_id,
            )
        )
    return tuple(captured_frames)


class _ParallelContextSideChannelData(NamedTuple):
    target_ids: tuple[str, ...]
    records: tuple[WarningErrorContextRecord, ...]


@contextmanager
def _restore_parallel_context(
    context: tuple[_ParallelContextFrame, ...],
) -> Iterator[Callable[[], _ParallelContextSideChannelData]]:
    active_target_ids = tuple(frame.result._parallel_target_id for frame in _context_stack.get())
    captured_target_ids = tuple(frame.target_id for frame in context)
    if active_target_ids == captured_target_ids:
        yield lambda: _ParallelContextSideChannelData((), ())
        return

    with ExitStack() as stack:
        restored_contexts = []
        for frame in context:
            restored_contexts.append(
                stack.enter_context(warning_error_context(frame.render(), record_only=frame.record_only))
            )

        def collect_side_channel_data() -> _ParallelContextSideChannelData:
            records = () if not restored_contexts else tuple(restored_contexts[0].records)
            return _ParallelContextSideChannelData(tuple(frame.target_id for frame in context), records)

        yield collect_side_channel_data


def _merge_parallel_context_side_channel_data(side_channel_data: _ParallelContextSideChannelData) -> None:
    for target_id in side_channel_data.target_ids:
        target = _parallel_context_targets.get(target_id)
        if target is not None:
            target.records.extend(side_channel_data.records)


def _record_event(
    event_type: Literal["warning", "error", "print"],
    context: str,
    message: Union[Warning, BaseException, str],
) -> None:
    record = WarningErrorContextRecord(event_type, context, message)
    for frame in _context_stack.get():
        frame.result.records.append(record)


def _is_record_only() -> bool:
    return any(frame.record_only for frame in _context_stack.get())


def _render_contexts(contexts: Sequence[tuple[str, dict[str, Any]]]) -> str:
    return " > ".join(_render_named_context(name, context) for name, context in contexts)


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


def _contextualize_warning(message: Union[Warning, str], context: Optional[str] = None) -> Union[Warning, str]:
    context = _render_context_stack() if context is None else context
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
        context = _render_context_stack()
        _record_event("warning", context, message.message)
        if _is_record_only():
            return
        message = copy(message)
        message.message = _contextualize_warning(message.message, context)

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
        context = _render_context_stack()
        if context:
            _record_event("warning", context, message)
            if _is_record_only():
                return
        _original_showwarning(_contextualize_warning(message, context), category, filename, lineno, file, line)

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


def warning_error_context(
    name: str,
    context: Optional[dict[str, Any]] = None,
    /,
    *,
    context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    record_only: bool = False,
) -> WarningErrorContext:
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
    count as distinct occurrences. If every contextual occurrence must be shown or
    recorded, use an appropriate warning filter such as
    ``warnings.simplefilter("always")``.

    When :func:`tpcp.parallel.delayed` captures an active context for execution in a
    worker process, its fixed metadata and ``record_only`` setting are restored for
    that task. A propagated ``context_provider`` is evaluated once when the worker
    task enters the restored context, so all events in that task use the same
    snapshot. Context providers created inside the worker retain their normal dynamic
    behavior. When :class:`tpcp.parallel.Parallel` is paired with
    :func:`tpcp.parallel.delayed`, records from successful tasks are merged back into
    the original context. Records from a task that raises are not recovered from a
    separate worker. With ``n_jobs=1``, execution remains inline: providers retain
    their normal dynamic behavior, and the active parent context directly records
    exceptions that escape the task.

    The context manager yields a :class:`WarningErrorContext` whose ``records`` list
    remains available after the context exits. It contains warnings that reached
    Python's warning dispatcher and exceptions that escaped the context. Original
    warning and exception objects are retained so callers can inspect their concrete
    types and custom attributes.

    The returned object can also be activated with :meth:`WarningErrorContext.start`
    and deactivated with :meth:`WarningErrorContext.stop`, which avoids indenting a
    long script. Manual use cannot automatically record or annotate an exception:
    ``stop()`` does not receive exception information, and an exception that skips
    the call leaves the context active. Use the ``with`` form whenever exception-safe
    cleanup or contextualized exceptions are required.

    Setting ``record_only=True`` on an outer context suppresses warning forwarding
    and :func:`print_with_context` output throughout its nested contexts while still
    recording those events. It does not suppress exceptions or ordinary
    :func:`print` calls. Warning filters still run before recording, including in
    record-only mode.

    On Python 3.9 and 3.10, exception context is stored in ``__notes__`` but is not
    displayed by Python's standard traceback renderer. Traceback renderers that
    support exception notes, such as Rich, display this context on those versions.
    """
    return WarningErrorContext(
        name,
        context,
        context_provider=context_provider,
        record_only=record_only,
    )


def print_with_context(
    *values: Any,
    sep: Optional[str] = " ",
    end: Optional[str] = "\n",
    file: Any = None,
    flush: bool = False,
) -> None:
    """Print values with the active context and retain the original message.

    Without an active :func:`warning_error_context`, this behaves exactly like
    :func:`print`. With an active context, the rendered context stack is prepended
    to the output and a ``"print"`` record is added to every active context result.
    Output is suppressed when any active context uses ``record_only=True``.
    """
    if not _context_stack.get():
        print(*values, sep=sep, end=end, file=file, flush=flush)
        return

    if sep is not None and not isinstance(sep, str):
        raise TypeError(f"sep must be None or a string, not {type(sep).__name__}")
    if end is not None and not isinstance(end, str):
        raise TypeError(f"end must be None or a string, not {type(end).__name__}")

    message_buffer = StringIO()
    print(*values, sep=sep, end="", file=message_buffer)
    message = message_buffer.getvalue()
    context = _render_context_stack()
    _record_event("print", context, message)
    if _is_record_only():
        return
    print(f"[{context}] {message}", end=end, file=file, flush=flush)


def _make_iteration_context_factory(i: int) -> _WarningErrorContextFactory:
    def make_context(
        name: str,
        context: Optional[dict[str, Any]] = None,
        /,
        *,
        context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
        record_only: bool = False,
    ) -> WarningErrorContext:
        context = {} if context is None else context
        if "i" in context:
            raise ValueError("The context key 'i' is reserved by iter_with_warning_error_context.")
        return warning_error_context(
            name,
            {"i": i, **context},
            context_provider=context_provider,
            record_only=record_only,
        )

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


_register_tpcp_parallel_side_channel(
    "warning_error_context",
    _capture_parallel_context,
    _restore_parallel_context,
    _merge_parallel_context_side_channel_data,
)
