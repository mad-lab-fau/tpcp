r"""
.. _dataclasses:

Dataclasses support
===================

When using `tpcp` you have to write a lot of classes with a lot of parameters.
For each class you need to repeat all parameter names up to 3 times, even before writing any documentation.

Below you can see the relevant part of the `QRSDetection` algorithm we implemented in another example.
Even though it has only 3 parameters, it requires over 20 lines of code to define the basic initialization.
"""

import pandas as pd

from tpcp import Algorithm, Parameter


class QRSDetector(Algorithm):
    _action_methods = "detect"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float]
    max_heart_rate_bpm: Parameter[float]
    min_r_peak_height_over_baseline: Parameter[float]

    # Results
    r_peak_positions_: pd.Series

    # Some internal constants
    _HIGH_PASS_FILTER_ORDER: int = 4

    def __init__(
        self,
        max_heart_rate_bpm: float = 200.0,
        min_r_peak_height_over_baseline: float = 1.0,
        high_pass_filter_cutoff_hz: float = 0.5,
    ):
        self.max_heart_rate_bpm = max_heart_rate_bpm
        self.min_r_peak_height_over_baseline = min_r_peak_height_over_baseline
        self.high_pass_filter_cutoff_hz = high_pass_filter_cutoff_hz


from dataclasses import dataclass, field

# %%
# Luckily, Python has a built-in solution for that, called `dataclasses`.
# With that, we can write the class above much more compact.
#
# The only downside is that the annotation of result fields and constants is a little more verbose, and you **need** to
# make sure that these parameters are excluded from the init.
# Otherwise, tpcp will explode ;)
from typing import ClassVar


@dataclass
class QRSDetector(Algorithm):
    _action_methods: ClassVar[str] = "detect"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float] = 200.0
    max_heart_rate_bpm: Parameter[float] = 1.0
    min_r_peak_height_over_baseline: Parameter[float] = 0.5

    # Results
    # We need to add the special field annotation, to exclude the parameter from the init
    r_peak_positions_: pd.Series = field(init=False)

    # Some internal constants
    # Using the ClassVar annotation, will mark this value as a constant and dataclasses will ignore it.
    _HIGH_PASS_FILTER_ORDER: ClassVar[int] = 4


# %%
# We still get all parameters in the init:
QRSDetector(high_pass_filter_cutoff_hz=4, max_heart_rate_bpm=200, min_r_peak_height_over_baseline=1)

# %%
# Inheritance
# -----------
# Creating child classes of `dataclasses` is also simple.
# Instead of repeating all parameters, you just need to specify the new once.
# However, you need to make sure that you also apply the `dataclass` decorator to the child class!
#
# But, be aware that you can not subclass a class that is not a `dataclass` and also has a `__init__` method!
# For example, you can not subclass :class:`~tpcp.optimize.GridSearch` with a dataclass, as it already defines its own
# `__init__`.
# In this case you need to use a regular class and manually repeat all parent parameters
# (and call `super().__init__()`).
@dataclass
class ModifiedQRSDetector(QRSDetector):
    new_parameter: Parameter[float] = 3


ModifiedQRSDetector(
    high_pass_filter_cutoff_hz=4, max_heart_rate_bpm=200, min_r_peak_height_over_baseline=1, new_parameter=3
)


# %%
# Mutable Defaults
# ----------------
# In `tpcp` we usually deal with the issue of mutable defaults by using the :class:`~tpcp.CloneFactory` (
# :func:`~tpcp.cf`).
# However, when using dataclasses, we can use the (more elegant) `field` annotation to define mutable defaults.


@dataclass
class FilterAlgorithm(Algorithm):
    _action_methods = "filter"

    # Input Parameters
    cutoff_hz: Parameter[float] = 2
    order: Parameter[int] = 5

    # Results
    filtered_signal_: pd.Series = field(init=False)


@dataclass
class HigherLevelFiler(QRSDetector):
    filter_algorithm: Parameter[FilterAlgorithm] = field(default_factory=lambda: FilterAlgorithm(3, 2))


# %%
# We can see that each instance will get a copy of the default value.
v1 = HigherLevelFiler()
v2 = HigherLevelFiler()

nested_object_is_different = v1.filter_algorithm is not v2.filter_algorithm
nested_object_is_different
