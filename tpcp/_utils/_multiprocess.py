"""Some helper to handle mutliprocess progressbars."""
import contextlib
from typing import ContextManager, Iterable

import joblib
from tqdm.auto import tqdm
from tqdm.std import tqdm as tqdm_base


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm_base, iterable: Iterable):
    """Context manager to patch joblib to report into tqdm progress bar given as argument.

    Modified based on: https://stackoverflow.com/a/58936697/3707545
    """

    class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield iterable
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


@contextlib.contextmanager
def tqdm_dummy(tqdm_object: tqdm_base):
    """Context manager to simply return the tqdm object."""
    try:
        yield tqdm_object
    finally:
        tqdm_object.close()


def init_progressbar(progress_bar: bool, n_jobs, iterable, **kwargs) -> ContextManager:
    """Create a multiprocess context manager to create a progress bar."""
    if progress_bar is False:
        return contextlib.nullcontext()
    if n_jobs == 1:
        return tqdm_dummy(tqdm(iterable, **kwargs))
    return tqdm_joblib(tqdm(**kwargs), iterable)
