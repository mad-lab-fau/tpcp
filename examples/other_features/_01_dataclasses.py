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
#
# Note, if you are using Python >=3.10, we highly recommend to use the `kw_only` option for dataclasses,
# which prevent some of the inheritance issues of dataclasses.
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
# ... warning :: New parameters will be added at the end in the positional order in the init method.
#                To avoid passing the wrong values to the wrong parameters, we highly recommend to pass parameters
#                only by name and not by position, or use the `kw_only` parameter of dataclasses supported in Python
#                >=3.10.
@dataclass
class ModifiedQRSDetector(QRSDetector):
    new_parameter: Parameter[float] = 3


ModifiedQRSDetector(
    high_pass_filter_cutoff_hz=4, max_heart_rate_bpm=200, min_r_peak_height_over_baseline=1, new_parameter=3
)

# %%
# Inheritance from complex tpcp classes
# --------------------------------------
# While inheriting from other dataclasses works without issues, be aware that you can not subclass a class that is
# not a `dataclass` and also has a `__init__` method!
# For example, you can not subclass :class:`~tpcp.optimize.GridSearch` with a dataclass, as it already defines its own
# `__init__`.
# In this case you need to use a regular class and manually repeat all parent parameters
# (and call `super().__init__()`).
#
# While this might not be a big deal for the GridSearch class, as you are not expected to subclass it on a regular, it
# can become annoying for classes like `~tpcp.Dataset` and `~tpcp.optimize.optuna.CustomOptunaOptimize`,
# which already have an init and you need to subclass to work with them.
# For these two classes (and other classes with predefined inits, we expect you to subclass from), we provide a
# `as_dataclass` class method that returns a data class version of the respective class:
from tpcp import Dataset
from itertools import product


@dataclass()
class CustomDataset(Dataset.as_dataclass()):  # Note the `as_dataclass` call here!
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            list(product(("patient_1", "patient_2", "patient_3"), ("test_1", "test_2"), ("1", "2"))),
            columns=["patient", "test", "extra"],
        )

    custom_param: float = 2  # This must have a default value, as the baseclass has parameters with defautls


CustomDataset(custom_param=3)

# %%
# Mutable Defaults
# ----------------
# In `tpcp` we usually deal with the issue of mutable defaults by using the :class:`~tpcp.CloneFactory` (
# :func:`~tpcp.cf`).
# However, when using dataclasses, we can use the (more elegant) `field` annotation to define mutable defaults.


@dataclass
class FilterAlgorithm(Algorithm):
    _action_methods: ClassVar = "filter"

    # Input Parameters
    cutoff_hz: Parameter[float] = 2
    order: Parameter[int] = 5

    # Results
    filtered_signal_: pd.Series = field(init=False)


@dataclass
class HigherLevelFilter(QRSDetector):
    filter_algorithm: Parameter[FilterAlgorithm] = field(default_factory=lambda: FilterAlgorithm(3, 2))


# %%
# We can see that each instance will get a copy of the default value.
v1 = HigherLevelFilter()
v2 = HigherLevelFilter()

nested_object_is_different = v1.filter_algorithm is not v2.filter_algorithm
nested_object_is_different


# %%
# Attrs
# -----
# A popular alternative to dataclasses is `attrs` (_`attrs.org`).
# It has the similar features as `dataclasses`, but has some additional features that can be helpfully.
# It also supports `kw_only` for all Python version (`kw_only` is great! Use it).
#
# You can use it simply be replacing the `dataclass` decorator with the `attrs.define` decorator in most examples above.
# Further, `attrs` has a `field` function, that works like `dataclasses.field`.
# Only the `default_factory` is called `factory`.
#
# Here are all the classes from above using attrs.
from attrs import define, field, Factory


@define
class QRSDetector(Algorithm):
    _action_methods: ClassVar[str] = "detect"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float] = 200.0
    max_heart_rate_bpm: Parameter[float] = 1.0
    min_r_peak_height_over_baseline: Parameter[float] = 0.5

    # Results
    r_peak_positions_: pd.Series = field(init=False)

    # Some internal constants
    _HIGH_PASS_FILTER_ORDER: ClassVar[int] = 4


@define
class FilterAlgorithm(Algorithm):
    _action_methods: ClassVar = "filter"

    # Input Parameters
    cutoff_hz: Parameter[float] = 2
    order: Parameter[int] = 5

    # Results
    filtered_signal_: pd.Series = field(init=False)


@define
class HigherLevelFilter(QRSDetector):
    filter_algorithm: Parameter[FilterAlgorithm] = Factory(lambda: FilterAlgorithm(3, 2))


# %%
# To support subclassing tpcp parameters with existing inits, we provide a `as_attrs` method on the respective classes.


@define(kw_only=True, slots=False)  # Slots sometimes don't play nicely with multiple inherantance
class CustomDataset(Dataset.as_attrs()):  # Note the `as_attrs` call here!
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            list(product(("patient_1", "patient_2", "patient_3"), ("test_1", "test_2"), ("1", "2"))),
            columns=["patient", "test", "extra"],
        )

    custom_param: float  # We don't need a default, as we are using `kw_only` in define


CustomDataset(custom_param=3)
