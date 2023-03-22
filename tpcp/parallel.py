"""A set of helper functions to correctly handle global variables in parallel processes.

Both functions are required as long as https://github.com/joblib/joblib/issues/1071 is not resolved.

The provided workarounds are similar to the ones done in scikit-learn
(https://github.com/scikit-learn/scikit-learn/pull/25363).

Note, that these fixes are not necessarily compatible with each other.
This means you can not forward custom callbacks through scikit-learn Parallel calls.
The same way, by default tpcp will not forward sklearn global configs through tpcp Parallel calls.
However, you can likely configure the callbacks in tpcp to make that work.
"""
import functools
from typing import Callable, List, Tuple, TypeVar

import joblib

T = TypeVar("T")
_PARALLEL_CONTEXT_CALLBACKS: List[Callable[[], Tuple[T, Callable[[T], None]]]] = []


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

    >>> from tpcp.parallel import delayed, register_global_parallel_callback
    >>> from joblib import Parallel
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
    >>> register_global_parallel_callback(callback)
    >>> # call the worker function in parallel
    >>> Parallel(n_jobs=2)(delayed(worker_func)() for _ in range(2))
    [True, True]

    Notes
    -----
    The getters are called as soon as the delayed function is called in the main process.
    This means, if you are calling `delayed` long before the actual parallel execution, the getters might not capture
    the correct state of the global variables.

    """
    _parallel_setter = []
    for g in _PARALLEL_CONTEXT_CALLBACKS:
        _parallel_setter.append(g())

    @functools.wraps(func)
    def inner(*args, **kwargs):
        for value, setter in _parallel_setter:
            setter(value)
        return func(*args, **kwargs)

    return joblib.delayed(inner)


def register_global_parallel_callback(callback: Callable[[], Tuple[T, Callable[[T], None]]]):
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
    callback : Callable[[], Tuple[T, Callable[[T], None]]]
        The callback function that will be called when :func:`~tpcp.parallel.delayed` is called in the main process.
        The callback should return a value and a setter function.

    """
    _PARALLEL_CONTEXT_CALLBACKS.append(callback)


__all__ = ["delayed", "register_global_parallel_callback"]
