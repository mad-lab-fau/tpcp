"""

When using `tpcp` you have to write a lot of classes with a lot of parameters.
For each class you need to repeat all paraemter names up to 3 times, even before writing any documentation.

Below you can see the relevant part of the `QRSDetection` algorithm we implemented in another example.
Eventhough it has only 3 parameters, it requires over 20 lines of code to define the basic initialization.
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


# %%
# Luckily, Python has a built-in solution for that, called `dataclasses`.
# With that, we can write the class above much more compact.
from dataclasses import dataclass, field


@dataclass
class QRSDetector:
    _action_methods = "detect"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float] = 200.0
    max_heart_rate_bpm: Parameter[float] = 1.0
    min_r_peak_height_over_baseline: Parameter[float] = 0.5

    # Results
    r_peak_positions_: pd.Series = field(init=False)

    # Some internal constants
    _HIGH_PASS_FILTER_ORDER: int = field(default=4, init=False)


# %%
# We still get all parameters in the init:
QRSDetector(high_pass_filter_cutoff_hz=4, max_heart_rate_bpm=200, min_r_peak_height_over_baseline=1)

# %%
# Unfortunately, the way `dataclasses` work, interferes with some fundamental checks runtime checks we run on
# our classes in tpcp.
# If we want to make this class above a child-class of Algorithm (or any other tpcp class), we have to pass the
# `dataclass` parameter at class initialisation.
# Before, you ask... Yes this is valid Python!
# The parameter will make sure that we change the way certain checks are performed under the assumption,
# the class will be wrapped in a dataclass decorator.
#
# .. warning:: Don't pass the `dataclass` parameter, if you are not using the `dataclass` decorator as well!
#              This will disable some internal checks, which might resurface as hard to debug errors later on.
@dataclass
class QRSDetector(Algorithm, dataclass=True):
    _action_methods = "detect"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float]
    max_heart_rate_bpm: Parameter[float]
    min_r_peak_height_over_baseline: Parameter[float]

    # Results
    r_peak_positions_: pd.Series = field(init=False)

    # Some internal constants
    _HIGH_PASS_FILTER_ORDER: int = field(default=4, init=False)


# %%
# Inheritance
# -----------
# Creating child classes of `dataclasses` is also simple.
# Instead of repeating all parameters, you just need to specify the new once.
# However, you need to make sure that you also apply the `dataclass` decorator to the child class and provide the
# `dataclass` parameter!
@dataclass
class ModifiedQRSDetector(QRSDetector, dataclass=True):
    new_parameter: Parameter[float] = 3


ModifiedQRSDetector(
    high_pass_filter_cutoff_hz=4, max_heart_rate_bpm=200, min_r_peak_height_over_baseline=1, new_parameter=3
)
