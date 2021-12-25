"""Some helper to handle mutliprocess progressbars."""
from concurrent.futures import Future, wait
from typing import Optional
from uuid import uuid4

import joblib
from joblib._parallel_backends import LokyBackend, SequentialBackend, FallbackToBackend
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
    def __init__(self, n_jobs: int, pbar=None):
        self.pbar = pbar
        self._backend = LokyBackend()
        self._n_jobs = n_jobs
        self.n_jobs = self._backend.effective_n_jobs(self._n_jobs)
        self.n_completed_tasks = 0
        self._id = uuid4().hex

    def _update_pbar(self, _=None):
        self.n_completed_tasks += 1
        if self.pbar:
            self.pbar.n = self.n_completed_tasks
            self.pbar.refresh()

    def submit(self, task) -> Future:
        fn, args, kwargs = task

        def _callback(ft):
            self._update_pbar()

        return self._backend.apply_async(lambda: fn(*args, **kwargs), _callback)

    def __enter__(self):
        self._initialize_backend()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._backend.terminate()

    def _initialize_backend(self):
        """Build a process or thread pool and return the number of workers"""
        try:
            n_jobs = self._backend.configure(n_jobs=self._n_jobs, parallel=self)
        except FallbackToBackend as e:
            # Recursively initialize the backend in case of requested fallback.
            self._backend = e.backend
            n_jobs = self._initialize_backend()

        return n_jobs
