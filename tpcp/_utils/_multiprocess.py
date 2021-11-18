"""Some helper to handle mutliprocess progressbars."""
import contextlib
from typing import ContextManager, Union

import joblib
from tqdm.std import tqdm as tqdm_base


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm_base):
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
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def init_progressbar(progress_bar: Union[bool, tqdm_base], default_progress_bar: tqdm_base, **kwargs) -> ContextManager:
    """Create a multiprocess context manager to create a progress bar."""
    if progress_bar is False:
        return contextlib.nullcontext()
    if progress_bar is True:
        progress_bar = default_progress_bar
    for k, v in kwargs.items():
        setattr(progress_bar, k, v)
    return tqdm_joblib(progress_bar)
