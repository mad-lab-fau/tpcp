"""Some helper to handle mutliprocess progressbars."""
from concurrent.futures import Future
from typing import Optional

import joblib
from joblib.externals.loky import get_reusable_executor


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


class CustomLokyPool:
    def __init__(self, n_jobs: int, timeout: Optional[float] = None):
        self.pool = None
        if n_jobs != 1:
            self.pool = get_reusable_executor(max_workers=n_jobs, timeout=timeout)

    def submit(self, task, callback):
        if self.pool:
            future = self.pool.submit(task())
            future.add_done_callback(callback)
        else:
            fn, args, kwargs = task()
            result = fn(*args, **kwargs)
            result_fu = Future()
            result_fu.set_result(result)
            callback(result_fu)
