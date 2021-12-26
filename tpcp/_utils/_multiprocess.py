"""Some helper to handle multiprocess progressbars."""

import joblib


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
