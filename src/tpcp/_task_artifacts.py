"""Joblib Memory integration for prehashed calls."""

import json
import os
import time

from joblib import hash as joblib_hash
from joblib.func_inspect import filter_args


class CallArtifacts:
    def __init__(self, memory, task, scope):
        self.function = memory.cache(task.func)
        self.backend = self.function.store_backend
        self.cached = task.cached
        if hasattr(self.function, "_get_args_id"):
            self.call_id = (self.function.func_id, self.function._get_args_id(*task.args, **task.kwargs))
        else:  # joblib 1.3
            self.call_id = self.function._get_output_identifiers(*task.args, **task.kwargs)
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

    def publish(self, reference, value, duration):
        key = tuple(json.loads(reference))
        self.backend.dump_item(key, value, verbose=0)
        if not self.backend.contains_item(key):
            raise OSError("Joblib failed to persist the task result.")
        if self.cached:
            inputs = filter_args(self.function.func, self.function.ignore, self.args, self.kwargs)
            self.backend.store_metadata(
                key, {"time": time.time(), "duration": duration, "input_args": {k: repr(v) for k, v in inputs.items()}}
            )

    def load(self, reference):
        return self.backend.load_item(tuple(json.loads(reference)), verbose=0)
