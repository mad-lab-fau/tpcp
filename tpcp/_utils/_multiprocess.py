"""Some helper to handle multiprocess progressbars."""
import functools
from typing import Callable, List, Tuple, TypeVar

import joblib

T = TypeVar("T")
_PARALLEL_CONTEXT_CALLBACKS: List[Callable[[], Tuple[T, Callable[[T], None]]]] = []


class TqdmParallel(joblib.Parallel):
    """Parallel Backend that can use a tqdm bar to display progress."""

    def __init__(
        self,
        n_jobs=None,
        backend=None,
        verbose=0,
        timeout=None,
        pre_dispatch="2 * n_jobs",
        batch_size="auto",
        temp_folder=None,
        max_nbytes="1M",
        mmap_mode="r",
        prefer=None,
        require=None,
        pbar=None,
    ):
        self.pbar = pbar
        super().__init__(
            n_jobs,
            backend,
            verbose,
            timeout,
            pre_dispatch,
            batch_size,
            temp_folder,
            max_nbytes,
            mmap_mode,
            prefer,
            require,
        )

    def __call__(self, *args, **kwargs):
        if self.pbar:
            with self.pbar:
                return joblib.Parallel.__call__(self, *args, **kwargs)
        else:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        """Update the tqdm bar after batch completion."""
        if self.pbar:
            self.pbar.n = self.n_completed_tasks
            self.pbar.refresh()
        super().print_progress()


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
    >>> from tpcp.parallel import delayed, register_global_parallel_callback
    >>> from joblib import Parallel
    >>> from my_package import set_config, config
    >>>
    >>> def callback():
    ...     def setter(config):
    ...         set_config(config)
    ...
    ...     return config(), setter
    >>>
    >>> def worker_func():
    ...     # This is what would be called in the parallel process
    ...     # We just retrun the config here for demonstration purposes
    ...     return config()
    >>>
    >>> # register the callback
    >>> register_global_parallel_callback(callback)
    >>> # call the worker function in parallel
    >>> Parallel(n_jobs=2)(delayed(worker_func)() for _ in range(2))

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

    Parameters
    ----------
    callback : Callable[[], Tuple[T, Callable[[T], None]]]
        The callback function that will be called when :func:`~tpcp.parallel.delayed` is called in the main process.
        The callback should return a value and a setter function.

    """
    _PARALLEL_CONTEXT_CALLBACKS.append(callback)
