"""
.. _custom_dataset_final_ecg:

The final ECG Example dataset
=============================

This is the final ECG Example dataset, that we developed step by step in the example :ref:`custom_dataset_ecg`.
This file can be used as quick reference or to import the class into other examples without side effects.
"""
from functools import lru_cache
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from tpcp import Dataset


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
        self.assert_is_single(None, "data")
        # Reconstruct the ecg file path based on the data index
        p_id = self.index["participant"][0]
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
        self.assert_is_single(None, "r_peaks_")
        p_id = self.index["participant"][0]
        r_peaks = pd.read_csv(self.data_path / f"{p_id}_all.csv", index_col=0)
        r_peaks = r_peaks.rename(columns={"R": "r_peak_position"})
        return r_peaks

    @property
    def pvc_positions_(self) -> pd.DataFrame:
        """The positions of R-peaks belonging to abnormal PVC peaks in the data stream.

        The position is equivalent to a position entry in `self.r_peak_positions_`.
        """
        self.assert_is_single(None, "pvc_positions_")
        p_id = self.index["participant"][0]
        pvc_peaks = pd.read_csv(self.data_path / f"{p_id}_pvc.csv", index_col=0)
        pvc_peaks = pvc_peaks.rename(columns={"PVC": "pvc_position"})
        return pvc_peaks

    @property
    def labeled_r_peaks_(self) -> pd.DataFrame:
        """All r-peak positions with an additional column that labels them as normal or PVC."""
        self.assert_is_single(None, "labeled_r_peaks_")
        r_peaks = self.r_peak_positions_
        r_peaks["label"] = "normal"
        r_peaks.loc[r_peaks["r_peak_position"].isin(self.pvc_positions_["pvc_position"]), "label"] = "pvc"
        return r_peaks

    def create_index(self) -> pd.DataFrame:
        participant_ids = [f.name.split("_")[0] for f in sorted(self.data_path.glob("*_all.csv"))]
        patient_group = [g for g, _ in zip(cycle(("group_1", "group_2", "group_3")), participant_ids)]
        df = pd.DataFrame({"patient_group": patient_group, "participant": participant_ids})
        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df
