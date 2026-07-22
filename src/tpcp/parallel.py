"""Helpers for restoring TPCP runtime context in joblib workers.

The worker-state helpers are required as long as https://github.com/joblib/joblib/issues/1071 is not resolved.
The :func:`delayed` wrapper restores registered global state, Python warning filters,
and active warning/error contexts for each worker task. :class:`Parallel` additionally
recovers side-channel data from successful tasks without changing their return values.
Applications can add their own round-trip state with :func:`register_parallel_side_channel`.

The provided workarounds are similar to the ones done in scikit-learn
(https://github.com/scikit-learn/scikit-learn/pull/25363).

Note, that these fixes are not necessarily compatible with each other.
This means you can not forward custom callbacks through scikit-learn Parallel calls.
The same way, by default tpcp will not forward sklearn global configs through tpcp Parallel calls.
However, you can likely configure the callbacks in tpcp to make that work.
"""

from __future__ import annotations

import functools
import multiprocessing
import warnings
from contextlib import AbstractContextManager, ExitStack
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import joblib

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T")
CalbackReturnType = tuple[T, Callable[[T], None]]
_PARALLEL_CONTEXT_CALLBACKS: dict[str, [Callable[[], CalbackReturnType]]] = {}
_TPCP_PARALLEL_NAME_PREFIX = "__tpcp_internal__."


@dataclass(frozen=True)
class _ParallelSideChannel:
    capture: Callable[[], Any]
    restore: Callable[[Any], AbstractContextManager[Callable[[], Any]]]
    merge: Callable[[Any], None]


_PARALLEL_SIDE_CHANNELS: dict[str, _ParallelSideChannel] = {}


def _tpcp_parallel_name(name: str) -> str:
    return f"{_TPCP_PARALLEL_NAME_PREFIX}{name}"


def _register_tpcp_global_parallel_callback(name: str, callback: Callable[[], CalbackReturnType]) -> None:
    """Register a TPCP-owned one-way callback in the reserved namespace."""
    _PARALLEL_CONTEXT_CALLBACKS[_tpcp_parallel_name(name)] = callback


def _remove_tpcp_global_parallel_callback(name: str) -> None:
    """Remove a TPCP-owned one-way callback from the reserved namespace."""
    del _PARALLEL_CONTEXT_CALLBACKS[_tpcp_parallel_name(name)]


def _register_tpcp_parallel_side_channel(
    name: str,
    capture: Callable[[], Any],
    restore: Callable[[Any], AbstractContextManager[Callable[[], Any]]],
    merge: Callable[[Any], None],
) -> None:
    """Register a TPCP-owned side channel, replacing it on module reload."""
    _PARALLEL_SIDE_CHANNELS[_tpcp_parallel_name(name)] = _ParallelSideChannel(capture, restore, merge)


@dataclass(frozen=True)
class _ParallelResult:
    value: Any
    side_channel_data: tuple[tuple[str, Any], ...]


_parallel_side_channels_enabled: ContextVar[bool] = ContextVar("tpcp_parallel_side_channels_enabled", default=False)


def _are_parallel_side_channels_enabled() -> bool:
    return _parallel_side_channels_enabled.get()


def _run_with_parallel_side_channels(func: Callable, args: tuple, kwargs: dict) -> Any:
    token = _parallel_side_channels_enabled.set(True)
    try:
        return func(*args, **kwargs)
    finally:
        _parallel_side_channels_enabled.reset(token)


def _call_with_parallel_side_channels(func: Callable, args: tuple, kwargs: dict, side_channels: Iterable) -> Any:
    with ExitStack() as stack:
        collectors = [
            (name, stack.enter_context(restore(captured_state))) for name, captured_state, restore in side_channels
        ]
        result = func(*args, **kwargs)
        if _are_parallel_side_channels_enabled():
            side_channel_data = tuple((name, collect()) for name, collect in collectors)
            return _ParallelResult(result, side_channel_data)
        return result


class _ParallelTaskIterator:
    def __init__(self, iterable: Iterable, context: Context) -> None:
        self._iterator = iter(iterable)
        self._context = context

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        func, args, kwargs = self._context.copy().run(next, self._iterator)
        return _run_with_parallel_side_channels, (func, args, kwargs), {}


def _merge_parallel_side_channel_data(result: Any) -> Any:
    if not isinstance(result, _ParallelResult):
        return result
    for name, side_channel_data in result.side_channel_data:
        _PARALLEL_SIDE_CHANNELS[name].merge(side_channel_data)
    return result.value


class Parallel(joblib.Parallel):
    """Run joblib tasks and recover side-channel data from TPCP worker contexts.

    This is a drop-in :class:`joblib.Parallel` subclass. When paired with
    :func:`delayed`, side-channel data produced by a successfully completed worker
    task is merged back into parent-process state. This includes records produced by
    restored warning/error contexts. Generator results are merged lazily as they are
    consumed.

    Side-channel data from a task that raises is not recovered; joblib propagates the original
    exception unchanged. The exception still retains context added inside the worker.
    Plain :class:`joblib.Parallel` remains supported with :func:`delayed`, but does not
    recover worker side-channel data.

    """

    def __call__(self, iterable):
        """Dispatch delayed tasks and merge side-channel data in the parent."""
        results = super().__call__(_ParallelTaskIterator(iterable, copy_context()))
        if self.return_generator:
            return (_merge_parallel_side_channel_data(result) for result in results)
        return [_merge_parallel_side_channel_data(result) for result in results]


def delayed(func):
    """Wrap a function to be used in a parallel context.

    This is a modified version of joblib.delayed that can run arbitrary callbacks when `delayed` is called in the
    main process and when the function is called in the parallel process.

    This is useful for example to restore the global config in the parallel process.
    For this to work, callbacks must be registered using :func:`~tpcp.parallel.register_global_parallel_callback` first.

    All uses of `delayed` in tpcp are using this implementation.
    This means you can configure your custom callbacks and expect them to work in all tpcp functions that use
    multiprocessing.

    If you need to write your own multiprocessing method using joblib, refer to the example below.

    Examples
    --------
    This example shows how to use this to make sure the global scikit-learn config is restored in the parallel process
    used in tpcp.
    Note, sklearn has a custom workaround for this, which is not compatible with tpcp.

    >>> from contextlib import contextmanager
    >>> from tpcp.parallel import (
    ...     Parallel,
    ...     delayed,
    ...     register_global_parallel_callback,
    ...     register_parallel_side_channel,
    ...     remove_global_parallel_callback,
    ...     remove_parallel_side_channel,
    ... )
    >>> from sklearn import get_config, set_config
    >>>
    >>> set_config(assume_finite=True)
    >>> def callback():
    ...     def setter(config):
    ...         set_config(**config)
    ...
    ...     return get_config(), setter
    >>>
    >>> def worker_func():
    ...     # This is what would be called in the parallel process
    ...     # We just return the config here for demonstration purposes
    ...     config = get_config()
    ...     return config["assume_finite"]
    >>>
    >>> # register the callback
    >>> name = register_global_parallel_callback(callback)
    >>> # call the worker function in parallel
    >>> Parallel(n_jobs=2)(delayed(worker_func)() for _ in range(2))
    [True, True]
    >>> # remove the callback again
    >>> remove_global_parallel_callback(name)
    >>>
    >>> # A side channel can additionally return data from successful worker tasks.
    >>> worker_events = []
    >>> collected_events = []
    >>> @contextmanager
    ... def restore_events(run_name):
    ...     worker_events.clear()
    ...     try:
    ...         yield lambda: tuple((run_name, event) for event in worker_events)
    ...     finally:
    ...         worker_events.clear()
    >>> def worker_with_event(value):
    ...     worker_events.append(f"processed {value}")
    ...     return value * 2
    >>> side_channel = register_parallel_side_channel(
    ...     lambda: "experiment-1", restore_events, collected_events.extend
    ... )
    >>> Parallel(n_jobs=2)(delayed(worker_with_event)(value) for value in range(2))
    [0, 2]
    >>> collected_events
    [('experiment-1', 'processed 0'), ('experiment-1', 'processed 1')]
    >>> remove_parallel_side_channel(side_channel)

    Notes
    -----
    The getters are called as soon as the delayed function is called in the main process.
    This means, if you are calling `delayed` long before the actual parallel execution, the getters might not capture
    the correct state of the global variables.

    Setters might be called multiple times in the same process, if the process pool is reused by multiple jobs.
    The callbacks should be robust against this and not break.

    Active :func:`~tpcp.misc.warning_error_context` instances are restored automatically for each worker task.
    Their worker-side lifetime is limited to that task, even when joblib reuses the process.
    The Python warning filters active when the delayed task is created are restored in process workers.
    Thread workers share process-global warning filters and therefore use the filters active when the task executes.
    Use :func:`register_parallel_side_channel` for custom state that must be restored
    for one worker task and optionally returned to the parent process.

    """
    _parallel_setter = []
    for g in _PARALLEL_CONTEXT_CALLBACKS.values():
        _parallel_setter.append(g())

    @functools.wraps(func)
    def create_task(*task_args, **task_kwargs):
        warning_filters = list(warnings.filters)
        parallel_side_channels = []
        for name, side_channel in _PARALLEL_SIDE_CHANNELS.items():
            captured_state = side_channel.capture()
            if captured_state is not None:
                parallel_side_channels.append((name, captured_state, side_channel.restore))

        @functools.wraps(func)
        def inner(*args, **kwargs):
            if multiprocessing.parent_process() is not None:
                with warnings.catch_warnings():
                    warnings.filters[:] = warning_filters
                    warnings._filters_mutated()
                    for value, setter in _parallel_setter:
                        setter(value)
                    return _call_with_parallel_side_channels(func, args, kwargs, parallel_side_channels)
            if _are_parallel_side_channels_enabled():
                return _call_with_parallel_side_channels(func, args, kwargs, parallel_side_channels)
            return func(*args, **kwargs)

        return joblib.delayed(inner)(*task_args, **task_kwargs)

    return create_task


def register_parallel_side_channel(
    capture: Callable[[], Any],
    restore: Callable[[Any], AbstractContextManager[Callable[[], Any]]],
    merge: Callable[[Any], None],
    name: str | None = None,
) -> str:
    """Register state that can make a round trip through a parallel worker.

    ``capture`` is called when :func:`delayed` creates a task. If it returns
    ``None``, the side channel is inactive for that task. Otherwise, ``restore``
    receives the captured state in the worker and must return a context manager.
    The callable yielded by that context manager is invoked after a successful
    task to collect side-channel data. When :class:`Parallel` consumes the task
    result, ``merge`` receives that data in the parent process.

    Plain :class:`joblib.Parallel` still enters the worker context, but does not
    collect or merge side-channel data. A task that raises does not return its
    side-channel data either.

    Parameters
    ----------
    capture
        Capture parent-process state for one delayed task. Return ``None`` to
        disable this side channel for that task.
    restore
        Create the context manager that restores the captured state around one
        worker task. It must yield a zero-argument collector callable.
    merge
        Merge the collector result into parent-process state.
    name
        Optional registration name used to remove the side channel again. Names
        beginning with ``"__tpcp_internal__."`` are reserved for TPCP.

    Returns
    -------
    str
        The registration name for :func:`remove_parallel_side_channel`.

    """
    if name is None:
        name = str(id(capture))
    _validate_public_parallel_name(name)
    if name in _PARALLEL_SIDE_CHANNELS:
        raise ValueError(f"Parallel side channel with name {name} already registered.")
    _PARALLEL_SIDE_CHANNELS[name] = _ParallelSideChannel(capture, restore, merge)
    return name


def remove_parallel_side_channel(name: str) -> None:
    """Remove a side channel registered with :func:`register_parallel_side_channel`."""
    _validate_public_parallel_name(name)
    if name not in _PARALLEL_SIDE_CHANNELS:
        raise KeyError(f"Parallel side channel with name {name} not found.")
    del _PARALLEL_SIDE_CHANNELS[name]


def _validate_public_parallel_name(name: str) -> None:
    if name.startswith(_TPCP_PARALLEL_NAME_PREFIX):
        raise ValueError(f"Names beginning with {_TPCP_PARALLEL_NAME_PREFIX!r} are reserved for TPCP.")


def register_global_parallel_callback(callback: Callable[[], tuple[T, Callable[[T], None]]], name=None) -> str:
    """Register a callback to transfer information to parallel workers.

    This callback should be used together with :func:`~tpcp.parallel.delayed` in a joblib parallel context.
    All calbacks that are registered here will be called when :func:`~tpcp.parallel.delayed` is called in the main
    process.
    We expect that the callback returns a value and a setter function.
    In the worker process, the setter function will be called with the value that was returned by the callback.

    As all tpcp functions that use multiprocessing are using :func:`~tpcp.parallel.delayed`, registered callbacks will
    be called correctly in all tpcp functions.

    For details see the docs of :func:`~tpcp.parallel.delayed`.

    Parameters
    ----------
    callback
        The callback function that will be called when :func:`~tpcp.parallel.delayed` is called in the main process.
        The callback should return a value and a setter function.
    name
        Optional name of the callback function.
        This can be used to remove the callback again.
        If None a random name will be used.
        The random name can be obtained via the return value. Names beginning with
        ``"__tpcp_internal__."`` are reserved for TPCP.

    Returns
    -------
    str
        The name of the callback. This can be used to remove the callback

    """
    if name is None:
        # Generate random name
        name = str(id(callback))
    _validate_public_parallel_name(name)
    if name in _PARALLEL_CONTEXT_CALLBACKS:
        raise ValueError(f"Callback with name {name} already registered.")
    _PARALLEL_CONTEXT_CALLBACKS[name] = callback

    return name


def remove_global_parallel_callback(name: str):
    """Remove a registered callback.

    This can be used to remove a callback that was registered using
    :func:`~tpcp.parallel.register_global_parallel_callback`.

    Parameters
    ----------
    name : str
        The name of the callback that should be removed.

    """
    _validate_public_parallel_name(name)
    if name in _PARALLEL_CONTEXT_CALLBACKS:
        del _PARALLEL_CONTEXT_CALLBACKS[name]
    else:
        raise KeyError(f"Callback with name {name} not found.")


__all__ = [
    "Parallel",
    "delayed",
    "register_global_parallel_callback",
    "register_parallel_side_channel",
    "remove_global_parallel_callback",
    "remove_parallel_side_channel",
]
