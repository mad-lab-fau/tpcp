r"""
.. _pretrained_models:

Pretrained models and predefined parameters
===========================================

All tpcp algorithms and pipelines store their configuration in the parameters of the class.

Each algorithm should already have sensible default parameters, which can be used for many applications.
However, for some algorithms we might know good parameters for specific use cases.
For example, an algorithm could have a parameter set for clean laboratory recordings and another one for noisier
free-living recordings.
Similarly, machine learning based algorithms might have a set of pretrained models that we provide with the package.

For both cases, we need a good way to make these parameters and models available to the user.
Below, we show the recommended way to do this.

The basic idea is to add a static inner class to the algorithm class, which contains the parameters and models.
In the easiest case, these are just dictionaries.
If more complex configuration is needed, we can use the `classproperty` decorator to load these parameters from a file.
In both cases, we wrap the returned dictionaries in `MappingProxyType` to make them readonly, so users do not
accidentally change the predefined parameters using the `update` method of the dict.

The simple case
---------------

For the simple case, we add a static inner class to the algorithm class.
For the example, we assume that the algorithm has two relevant parameters, `max_heart_rate_bpm` and
`min_r_peak_height_over_baseline`, and that we have two sets of predefined parameters for clean and noisy ECG signals.

We will omit the algorithm implementation here, as it is not relevant for the example.
"""

from types import MappingProxyType
from typing import Optional

import pandas as pd

from tpcp import Algorithm, Parameter


class QRSDetector(Algorithm):
    _action_methods = "detect"

    max_heart_rate_bpm: Parameter[float]
    min_r_peak_height_over_baseline: Parameter[float]

    r_peak_positions_: pd.Series

    def __init__(
        self,
        max_heart_rate_bpm: float = 200.0,
        min_r_peak_height_over_baseline: float = 1.0,
    ):
        self.max_heart_rate_bpm = max_heart_rate_bpm
        self.min_r_peak_height_over_baseline = min_r_peak_height_over_baseline

    def detect(self, single_channel_ecg: pd.Series, sampling_rate_hz: float):
        self.r_peak_positions_ = pd.Series([], dtype=int)
        return self

    class PredefinedParameters:
        clean_lab_data = MappingProxyType(
            {
                "max_heart_rate_bpm": 180.0,
                "min_r_peak_height_over_baseline": 1.0,
            }
        )
        noisy_free_living_data = MappingProxyType(
            {
                "max_heart_rate_bpm": 220.0,
                "min_r_peak_height_over_baseline": 2.5,
            }
        )


# %%
# Now we can use the predefined parameters as follows:
clean_lab_params = QRSDetector.PredefinedParameters.clean_lab_data
algo_with_clean_lab_params = QRSDetector(**clean_lab_params)
algo_with_clean_lab_params.get_params()

# %%
# This way, we can easily store different parameters for different use cases.
# Users are still able to overwrite the parameters, if they want to.
# For example, they can use the predefined parameters for some parameters and overwrite others:
noisy_free_living_params = (
    QRSDetector.PredefinedParameters.noisy_free_living_data
)
algo_with_custom_noisy_params = QRSDetector(
    **dict(noisy_free_living_params, min_r_peak_height_over_baseline=3.0)
)
algo_with_custom_noisy_params.get_params()

# %%
# Other possible syntax versions (based on preference):
noisy_free_living_params = (
    QRSDetector.PredefinedParameters.noisy_free_living_data
)
algo_with_custom_noisy_params = QRSDetector(
    **{**noisy_free_living_params, "min_r_peak_height_over_baseline": 3.0}
)
algo_with_custom_noisy_params.get_params()

# %%
# Or:
noisy_free_living_params = (
    QRSDetector.PredefinedParameters.noisy_free_living_data
)
algo_with_custom_noisy_params = QRSDetector(
    **(noisy_free_living_params | dict(min_r_peak_height_over_baseline=3.0))
)
algo_with_custom_noisy_params.get_params()

# %%
# Or by using `set_params`:
algo_with_custom_noisy_params = QRSDetector(**noisy_free_living_params)
algo_with_custom_noisy_params.set_params(min_r_peak_height_over_baseline=3.0)
algo_with_custom_noisy_params.get_params()

# %%
# Depending on the specific case, we can also use one of the predefined parameter sets as the default values for the
# constructor.
# We can use the :func:`~tpcp.misc.set_defaults` function to do this easily.
from tpcp.misc import set_defaults


class QRSDetector(Algorithm):
    _action_methods = "detect"

    max_heart_rate_bpm: Parameter[float]
    min_r_peak_height_over_baseline: Parameter[float]

    r_peak_positions_: pd.Series

    class PredefinedParameters:
        clean_lab_data = MappingProxyType(
            {
                "max_heart_rate_bpm": 180.0,
                "min_r_peak_height_over_baseline": 1.0,
            }
        )
        noisy_free_living_data = MappingProxyType(
            {
                "max_heart_rate_bpm": 220.0,
                "min_r_peak_height_over_baseline": 2.5,
            }
        )

    @set_defaults(**PredefinedParameters.clean_lab_data)
    def __init__(
        self,
        max_heart_rate_bpm: float,
        min_r_peak_height_over_baseline: float,
    ):
        self.max_heart_rate_bpm = max_heart_rate_bpm
        self.min_r_peak_height_over_baseline = min_r_peak_height_over_baseline

    def detect(self, single_channel_ecg: pd.Series, sampling_rate_hz: float):
        self.r_peak_positions_ = pd.Series([], dtype=int)
        return self


algo = QRSDetector()
algo.get_params()

# %%
# Loading parameters from a file (or other source)
# ------------------------------------------------
#
# Sometimes, we have a lot of parameters that we do not want to hardcode in the source code, or we want to include
# objects in the parameters that cannot be easily hardcoded, such as a trained machine learning model.
#
# In this case, we can use the `classproperty` decorator to load the parameters from a file or any other source.
# As our parameter class is "just" a class, we can easily add such a property to it.
#
# Let's assume our algorithm has an additional parameter `model` that takes an optional pretrained ML model.
# We will provide different versions of predefined parameters that use different models.
#
# For the example, we will not actually load a model, but just use a string.
from tpcp.misc import classproperty


class MLQRSDetector(Algorithm):
    _action_methods = "detect"

    max_heart_rate_bpm: Parameter[float]
    min_r_peak_height_over_baseline: Parameter[float]
    model: Parameter[Optional[str]]

    r_peak_positions_: pd.Series

    class PredefinedParameters:
        @classmethod
        def _load_from_file(cls, model_name: str) -> str:
            # Load the model from a file here.
            # We could even add a caching mechanism here, if we want to avoid loading the model multiple times.
            print(f"Loading model {model_name} from file")
            return "model_" + model_name

        @classproperty
        def clean_lab_data(cls):
            model = cls._load_from_file("clean_lab_data")
            return MappingProxyType(
                {
                    "max_heart_rate_bpm": 180.0,
                    "min_r_peak_height_over_baseline": 1.0,
                    "model": model,
                }
            )

        @classproperty
        def noisy_free_living_data(cls):
            model = cls._load_from_file("noisy_free_living_data")
            return MappingProxyType(
                {
                    "max_heart_rate_bpm": 220.0,
                    "min_r_peak_height_over_baseline": 2.5,
                    "model": model,
                }
            )

    def __init__(
        self,
        max_heart_rate_bpm: float = 200.0,
        min_r_peak_height_over_baseline: float = 1.0,
        model: Optional[str] = None,
    ):
        self.max_heart_rate_bpm = max_heart_rate_bpm
        self.min_r_peak_height_over_baseline = min_r_peak_height_over_baseline
        self.model = model

    def detect(self, single_channel_ecg: pd.Series, sampling_rate_hz: float):
        self.r_peak_positions_ = pd.Series([], dtype=int)
        return self


# %%
# Now we can use the predefined parameters as before.
# The file loading is only done when we actually use the parameters, and we avoid loading all models into memory at the
# beginning.
ml_clean_lab_params = MLQRSDetector.PredefinedParameters.clean_lab_data
ml_clean_lab_params

# %%
algo_with_clean_lab_model = MLQRSDetector(**ml_clean_lab_params)
algo_with_clean_lab_model.get_params()
