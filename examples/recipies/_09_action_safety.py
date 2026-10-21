# ruff: noqa: D400
r"""
.. _action_safety:

Action safety checks
====================

An action method must leave the algorithm's parameters unchanged, return the
algorithm instance, and store results in attributes ending in ``_``.
The :func:`~tpcp.make_action_safe` decorator checks these conventions at
runtime. Pipelines can also use :meth:`~tpcp.Pipeline.safe_run` to check
their ``run`` method.

Checking an action
------------------
This detector returns the positions of values above a threshold. It stores its
result in ``positions_`` and leaves the ``threshold`` parameter unchanged.
"""

from tpcp import (
    Algorithm,
    config_context,
    get_config,
    make_action_safe,
    set_config,
)
from tpcp.parallel import Parallel, delayed


class ThresholdDetector(Algorithm):
    _action_methods = "detect"

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    @make_action_safe
    def detect(self, data):
        self.positions_ = [
            i for i, value in enumerate(data) if value > self.threshold
        ]
        return self


data = [0.5, 1.5, 0.8, 2.0]
detector = ThresholdDetector().detect(data)
print(detector.positions_)

# %%
# Detecting parameter changes
# ---------------------------
# Changing a parameter inside the action makes subsequent runs depend on earlier
# inputs. Here, the decorator reports the mistake after the action finishes.


class ParameterChangingDetector(ThresholdDetector):
    @make_action_safe
    def detect(self, data):
        self.threshold = max(data)
        self.positions_ = [
            i for i, value in enumerate(data) if value > self.threshold
        ]
        return self


try:
    ParameterChangingDetector().detect(data)
except ValueError as error:
    print(error)

# %%
# The checks compare hashes of ``get_params(deep=False)`` before and after the
# action. Each parameter object's serialized nested state is included, including
# fitted sklearn model state and results on nested tpcp objects. The action's
# own results are excluded. Clone a nested algorithm before running it if it
# is a parameter of the outer algorithm.
#
# Custom serializers must include all state relevant to mutation detection;
# parameters omitted by serialization are not checked separately. For local and
# non-importable classes, annotation metadata and tpcp validation caches are
# excluded from hashing. Store hash-relevant settings as ordinary parameter
# state.

# %%
# Configuring checks
# ------------------
# Checks are enabled by default. Hashing large model parameters can take time.
# Checks can be disabled for an algorithm whose behavior has already been
# verified. Use :func:`~tpcp.config_context` to limit the change to a block.
print(get_config())
with config_context(action_safety_checks=False):
    detector = ThresholdDetector().detect(data)
    print(detector.positions_)
print(get_config())

# %%
# This skips all checks from ``make_action_safe`` and ``Pipeline.safe_run``:
# parameter mutation, return-type and result checks, and action-registration
# warnings. The action still runs normally, including any exceptions it raises.
# Cloning and optimization checks are unaffected. Nested contexts restore the
# previous setting, including when the block raises an exception.
#
# For a persistent setting, use :func:`~tpcp.set_config`. Here we restore the
# previous value so that running the example leaves the configuration unchanged.
previous_config = get_config()
try:
    set_config(action_safety_checks=False)
    print(get_config())
finally:
    set_config(**previous_config)

# %%
# Parallel tasks
# --------------
# Settings apply to the current thread and asynchronous execution context; new
# threads start with defaults. Pair :class:`tpcp.parallel.Parallel` with
# :func:`tpcp.parallel.delayed` to propagate settings to serial, process, and
# thread tasks. Each delayed task captures the setting when it is created.
# Workers restore their previous configuration after each task, including on
# failure.


def run_parallel_example():
    """Run process tasks without making workers import this gallery module."""

    def check_in_worker():
        return get_config()["action_safety_checks"]

    with config_context(action_safety_checks=False):
        return Parallel(n_jobs=2)(delayed(check_in_worker)() for _ in range(2))


print(run_parallel_example())

# %%
# ``joblib.Parallel`` also propagates settings to process workers when paired
# with ``tpcp.parallel.delayed``. Use both tpcp helpers for serial and thread
# execution as well.
