"""Joblib Memory integration for prehashed calls."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from joblib import dump
from joblib import hash as joblib_hash
from joblib.func_inspect import filter_args


def _injected_context():
    """Stable placeholder for the context supplied at dispatch."""


class CallArtifacts:
    def __init__(self, memory, task, scope):
        self.function = memory.cache(task.func)
        self.backend = self.function.store_backend
        self.cached = task.cached
        if task.cached and task.pass_context:
            raise ValueError("CachedTask cannot use pass_context.")
        args = (_injected_context, *task.args) if task.pass_context else task.args
        if hasattr(self.function, "_get_args_id"):
            self.call_id = (self.function.func_id, self.function._get_args_id(*args, **task.kwargs))
        else:  # joblib 1.3
            self.call_id = self.function._get_output_identifiers(*args, **task.kwargs)
        identity = ("cached",) if task.cached else ("scope", scope)
        self.task_id = json.dumps([os.path.realpath(self.backend.location), identity, self.call_id])
        self.args = task.args
        self.kwargs = task.kwargs

    def reference(self, attempt):
        if self.cached:
            return json.dumps(self.call_id)
        return json.dumps(["__tpcp_tasks__", joblib_hash(self.task_id), attempt])

    def available(self, reference):
        if self.cached:
            return self.function._is_in_cache_and_valid(self.call_id)
        return reference and self.backend.contains_item(json.loads(reference))

    @contextmanager
    def stage(self, value, duration):
        # Keep scratch on the artifact filesystem so promotion is an atomic rename.
        with TemporaryDirectory(dir=self.backend.location, prefix=".tpcp-stage-") as directory:
            stage = Path(directory)
            with (stage / "output.pkl").open("wb") as output:
                dump(value, output, compress=self.backend.compress)
            if self.cached:
                inputs = filter_args(self.function.func, self.function.ignore, self.args, self.kwargs)
                metadata = {
                    "time": time.time(),
                    "duration": duration,
                    "input_args": {k: repr(v) for k, v in inputs.items()},
                }
                (stage / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            yield stage

    def publish(self, reference, stage):
        destination = Path(self.backend.location).joinpath(*json.loads(reference))
        destination.mkdir(parents=True, exist_ok=True)
        if self.cached:
            (stage / "metadata.json").replace(destination / "metadata.json")
        (stage / "output.pkl").replace(destination / "output.pkl")

    def output_identity(self, reference):
        path = Path(self.backend.location).joinpath(*json.loads(reference), "output.pkl")
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def load(self, reference):
        return self.backend.load_item(tuple(json.loads(reference)), verbose=0)
