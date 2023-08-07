r"""
.. _custom_dataset_ecg:

Custom Dataset - A real world example
=====================================

To better understand how you would actually use tpcp datasets, we are going to build a dataset class for an actual
dataset.
We are going to use a subset of the `MIT-BIH Arrhythmia Database <https://physionet.org/content/mitdb/1.0.0/>`_.
The actual content of the data is not relevant, but it has a couple of key characteristics that are typical for such
datasets:

- Data comes in individual files per participant/recording
- There are multiple files (with different formats and structures) for each recording
- Each recording/data point is an entire time series

These characteristics typically make working with such a dataset a little cumbersome.
In the following we want to explore how we can create a tpcp dataset for it, to make future work with the data easier.

If you want to see other real-life implementations of tpcp-datasets you can also check out:

* https://github.com/mad-lab-fau/gaitmap-datasets
* https://github.com/mad-lab-fau/cft-analysis/tree/main/cft_analysis/datasets

If you just want the final implementation, without all the explanation, check :ref:`custom_dataset_final_ecg`.

"""

# %%
# Creating an index
# -----------------
# First we need to figure out what data we have and convert that into an index dataframe.
# To make things a little more complicated we will add a second layer to the data, by assigning each participant
# into one of three "patient groups" arbitrarily.
#
# In the data we have 3 files per participant:
#   1. {patient_id}.pk.gz -> The ECG recording
#   2. {patient_id}_all.csv -> The position of the R-peaks of all heart beats (PVC or normal).
#       All heart beats that show a different condition than PVC are already excluded
#   3. {patient_id}_pvc.csv -> The position of all PVC heart beats in the recording.
#
# Later we need to include the data of all files into the dataset, but to generate out index, it is sufficient to only
# list one of the datatypes.
from pathlib import Path
from typing import List, Optional, Union

from tpcp import Dataset

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path(".").resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"

# Note that we sort the files explicitly, as the file order might depend on the operating system.
# Otherwise, the ordering of our dataset might not be reproducible
participant_ids = [f.name.split("_")[0] for f in sorted(data_path.glob("*_all.csv"))]


# %%
# This information forms one level of our dataset.
# We will add the "patient group" as arbitrary hardcoded second level to our dataset to make things a little more
# interesting.
#
# Afterwards we put everything into an index that we will use for our dataset.
from itertools import cycle

import pandas as pd

patient_group = [g for g, _ in zip(cycle(("group_1", "group_2", "group_3")), participant_ids)]

data_index = pd.DataFrame({"patient_group": patient_group, "participant": participant_ids})
data_index

# %%
# Creating the dataset
# --------------------
# Now that we know how to create our index, we will integrate this logic into our dataset.
# Note, that we do not want to hardcode the dataset path and hence, turn it into a parameter of the dataset.
# The rest of the logic stays the same and goes into the `create_index` method.
#
# .. note:: Note that we sort the files explicitly, as the file order might depend on the operating system.
#           Otherwise, the ordering of our dataset might not be reproducible.


class ECGExampleData(Dataset):
    data_path: Path

    def __init__(
        self,
        data_path: Path,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        participant_ids = [f.name.split("_")[0] for f in sorted(self.data_path.glob("*_all.csv"))]
        patient_group = [g for g, _ in zip(cycle(("group_1", "group_2", "group_3")), participant_ids)]
        df = pd.DataFrame({"patient_group": patient_group, "participant": participant_ids})
        # Some additional checks to avoid common issues
        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df


ECGExampleData(data_path=data_path)

# %%
# Adding data
# -----------
# The implementation above is a fully functional dataset and can be used to split or iterate the index.
# However, to make things really convenient, we want to add data access parameters to the dataset.
# We start with the raw ecg data.
#
# In general, it is completely up to you how you implement this.
# You can use methods on the dataset, properties, or create a set of functions, that get a dataset instance as an input.
# The way below shows how we usually do it.
# The most important thing in all cases is documentation to make sure everyone knows what data they are getting and
# in which format.
#
# As we don't know how people will use the dataset, we will load the data only on demand.
# Further, loading the data (in this case) only makes sense, when there is just a single participant selected in the
# dataset.
#
# We will implement this logic by using a `property` to implement "load-on-demand" and the `dataset.is_single_datapoint`
# method to check that there is really only a single participant selected.


class ECGExampleData(Dataset):
    data_path: Path

    def __init__(
        self,
        data_path: Path,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        """The sampling rate of the raw ECG recording in Hz"""
        return 360.0

    @property
    def data(self) -> pd.DataFrame:
        """The raw ECG data of a participant's recording.

        The dataframe contains a single column called "ecg".
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        # Check that there is only a single participant in the dataset
        if not self.is_single_datapoint():
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        # Reconstruct the ecg file path based on the data index
        p_id = self.index["participant"][0]
        return pd.read_pickle(self.data_path / f"{p_id}.pk.gz")

    def create_index(self) -> pd.DataFrame:
        participant_ids = [f.name.split("_")[0] for f in sorted(self.data_path.glob("*_all.csv"))]
        patient_group = [g for g, _ in zip(cycle(("group_1", "group_2", "group_3")), participant_ids)]
        df = pd.DataFrame({"patient_group": patient_group, "participant": participant_ids})
        # Some additional checks to avoid common issues
        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df


# %%
# With that logic, we can now select a subset of the dataset that contains only a single participant and then access
# the data.
dataset = ECGExampleData(data_path=data_path)
subset = dataset[0]
subset

# %%
subset.data

# %%
# Adding more data
# ----------------
# In the same way, we can add the remaining data.
# The remaining data we have available is both data generated by human labelers on the ECG signal.
# Aka data that you usually would not have available with your ECG recording.
# Hence, we will treat this information as "reference data/labels".
# By convention, we usually use a trailing `_` after the property name to indicate that.
#
# We also add one "derived" property (`labeled_r_peaks_`) that returns the data in a more convenient format for
# certain tasks.
# You could also implement methods or properties that perform certain computations.
# For example if there is a typical pre-processing that should always be applied to the data, it might be good to add
# a property `data_cleaned` (or similar) that runs this processing on demand when the parameter is accessed.


class ECGExampleData(Dataset):
    data_path: Path

    def __init__(
        self,
        data_path: Path,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        """The sampling rate of the raw ECG recording in Hz"""
        return 360.0

    @property
    def data(self) -> pd.DataFrame:
        """The raw ECG data of a participant's recording.

        The dataframe contains a single column called "ecg".
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        # Check that there is only a single participant in the dataset
        self.assert_is_single_datapoint("data")
        # Reconstruct the ecg file path based on the data index
        p_id = self.index["participant"][0]
        return pd.read_pickle(self.data_path / f"{p_id}.pk.gz")

    @property
    def r_peak_positions_(self) -> pd.DataFrame:
        """The sample positions of all R-peaks in the ECG data.

        This includes all R-Peaks (PVC or normal)
        """
        self.assert_is_single_datapoint("r_peaks_")
        p_id = self.index["participant"][0]
        r_peaks = pd.read_csv(self.data_path / f"{p_id}_all.csv", index_col=0)
        r_peaks = r_peaks.rename(columns={"R": "r_peak_position"})
        return r_peaks

    @property
    def pvc_positions_(self) -> pd.DataFrame:
        """The positions of R-peaks belonging to abnormal PVC peaks in the data stream.

        The position is equivalent to a position entry in `self.r_peak_positions_`.
        """
        self.assert_is_single_datapoint("pvc_positions_")
        p_id = self.index["participant"][0]
        pvc_peaks = pd.read_csv(self.data_path / f"{p_id}_pvc.csv", index_col=0)
        pvc_peaks = pvc_peaks.rename(columns={"PVC": "pvc_position"})
        return pvc_peaks

    @property
    def labeled_r_peaks_(self) -> pd.DataFrame:
        """All r-peak positions with an additional column that labels them as normal or PVC."""
        self.assert_is_single_datapoint("labeled_r_peaks_")
        r_peaks = self.r_peak_positions_
        r_peaks["label"] = "normal"
        r_peaks.loc[r_peaks["r_peak_position"].isin(self.pvc_positions_["pvc_position"]), "label"] = "pvc"
        return r_peaks

    def create_index(self) -> pd.DataFrame:
        participant_ids = [f.name.split("_")[0] for f in sorted(self.data_path.glob("*_all.csv"))]
        patient_group = [g for g, _ in zip(cycle(("group_1", "group_2", "group_3")), participant_ids)]
        df = pd.DataFrame({"patient_group": patient_group, "participant": participant_ids})
        # Some additional checks to avoid common issues
        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df


dataset = ECGExampleData(data_path=data_path)
subset = dataset[0]
subset.labeled_r_peaks_

# %%
# Conclusion
# ----------
# While building the dataset is not always easy and requires you to think about how you want to access the data, it
# makes working with the data in the future easy.
# Even without using any other tpcp feature, it provides a clear overview over the data available and abstracts the
# complexity of data loading.
# This can prevent accidental errors and just a much faster and better workflow in case new people want to work with
# a dataset.

# %%
# Advanced Concepts - Caching
# ---------------------------
# Loading/pre-processing the data on demand in the dataset above is a good optimization, if you only need to access the
# data once.
# However, if you need to access the data multiple times you might want to cache the loaded data within a single
# execution of you code, or even cache time-consuming computations across multiple runs/programs.
# Depending on the scenario this can either be achieved by using a in memory cache like `functools.lru_cache` or a disk
# cache like `joblib.Memory`.
#
# Finding the right functions to cache and to do it in a way that balances runtime and other resource usage is tricky.
# So only do that, if you really need it.
# However, when you implement it, the best approach is to extract the part you want to cache into a global function
# **without** side-effects and then wrap this function with your caching method of choice.
#
# Below we will demonstrate how to do that using Pythons `lru_cache` for the `data` property and make caching optional
# using a dataset parameter.
from functools import lru_cache


def load_pandas_pickle_file(file_path):
    return pd.read_pickle(file_path)


cached_load_pandas_pickle_file = lru_cache(10)(load_pandas_pickle_file)


class ECGExampleData(Dataset):
    data_path: Path
    use_lru_cache: bool

    def __init__(
        self,
        data_path: Path,
        *,
        use_lru_cache: bool = True,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        self.use_lru_cache = use_lru_cache
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def sampling_rate_hz(self) -> float:
        """The sampling rate of the raw ECG recording in Hz"""
        return 360.0

    @property
    def data(self) -> pd.DataFrame:
        """The raw ECG data of a participant's recording.

        The dataframe contains a single column called "ecg".
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        # Check that there is only a single participant in the dataset
        self.assert_is_single_datapoint("data")
        # Reconstruct the ecg file path based on the data index
        p_id = self.group_label.participant
        file_path = self.data_path / f"{p_id}.pk.gz"
        # We try to use the cache if enabled.
        if self.use_lru_cache:
            return cached_load_pandas_pickle_file(file_path)
        return load_pandas_pickle_file(file_path)

    @property
    def r_peak_positions_(self) -> pd.DataFrame:
        """The sample positions of all R-peaks in the ECG data.

        This includes all R-Peaks (PVC or normal)
        """
        self.assert_is_single_datapoint("r_peaks_")
        p_id = self.group_label.participant
        r_peaks = pd.read_csv(self.data_path / f"{p_id}_all.csv", index_col=0)
        r_peaks = r_peaks.rename(columns={"R": "r_peak_position"})
        return r_peaks

    @property
    def pvc_positions_(self) -> pd.DataFrame:
        """The positions of R-peaks belonging to abnormal PVC peaks in the data stream.

        The position is equivalent to a position entry in `self.r_peak_positions_`.
        """
        self.assert_is_single_datapoint("pvc_positions_")
        p_id = self.index["participant"][0]
        pvc_peaks = pd.read_csv(self.data_path / f"{p_id}_pvc.csv", index_col=0)
        pvc_peaks = pvc_peaks.rename(columns={"PVC": "pvc_position"})
        return pvc_peaks

    @property
    def labeled_r_peaks_(self) -> pd.DataFrame:
        """All r-peak positions with an additional column that labels them as normal or PVC."""
        self.assert_is_single_datapoint("labeled_r_peaks_")
        r_peaks = self.r_peak_positions_
        r_peaks["label"] = "normal"
        r_peaks.loc[r_peaks["r_peak_position"].isin(self.pvc_positions_["pvc_position"]), "label"] = "pvc"
        return r_peaks

    def create_index(self) -> pd.DataFrame:
        participant_ids = [f.name.split("_")[0] for f in sorted(self.data_path.glob("*_all.csv"))]
        patient_group = [g for g, _ in zip(cycle(("group_1", "group_2", "group_3")), participant_ids)]
        df = pd.DataFrame({"patient_group": patient_group, "participant": participant_ids})
        # Some additional checks to avoid common issues
        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df
