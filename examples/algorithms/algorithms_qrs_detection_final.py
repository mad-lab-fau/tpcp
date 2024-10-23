r"""
.. _custom_algorithms_qrs_detection_final:

The final QRS detection algorithms
==================================

These are the QRS detection algorithms, that we developed step by step :ref:`custom_algorithms_qrs_detection`.
This file can be used as quick reference or to import the class into other examples without side effects.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial import KDTree, minkowski_distance
from sklearn.metrics import roc_curve
from tpcp import (
    Algorithm,
    HyperParameter,
    OptimizableParameter,
    Parameter,
    make_action_safe,
    make_optimize_safe,
)


def match_events_with_reference(
    events: np.ndarray, reference: np.ndarray, tolerance: Union[int, float]
) -> np.ndarray:
    """Find matches in two lists based on the distance between their vectors.

    Parameters
    ----------
    events : array with shape (n, d)
        An n long array of d-dimensional vectors
    reference : array with shape (m, d)
        An m long array of d-dimensional vectors
    tolerance
        Max allowed Chebyshev distance between matches

    Returns
    -------
    A array that marks all matches.
    If one value is NaN, it means that no match was found for this index.

    Notes
    -----
    Only a single match per index is allowed in both directions.
    This means that every index will only occur once in the output arrays.
    If multiple matches are possible based on the tolerance of the Chebyshev distance, the closest match will be
    selected based on the Manhatten distance (aka `np.sum(np.abs(left_match - right_match`).
    Only this match will be returned.
    Note, that in the implementation, we first get the closest match based on the Manhatten distance and check in a
    second step if this closed match is also valid based on the Chebyshev distance.

    """
    if len(events) == 0 or len(reference) == 0:
        return np.array([])

    events = np.atleast_1d(events.squeeze())
    reference = np.atleast_1d(reference.squeeze())
    assert np.ndim(events) == 1, "Events must be a 1D-array"
    assert np.ndim(reference) == 1, "Reference must be a 1D-array"
    events = np.atleast_2d(events).T
    reference = np.atleast_2d(reference).T

    right_tree = KDTree(reference)
    left_tree = KDTree(events)

    # We calculate the closest neighbor based on the Manhatten distance in both directions and then find only the cases
    # were the right side closest neighbor resulted in the same pairing as the left side closest neighbor ensuring
    # that we have true one-to-one-matches
    # p = 1 is used to select the Manhatten distance
    l_nearest_distance, l_nearest_neighbor = right_tree.query(
        events, p=1, workers=-1
    )
    _, r_nearest_neighbor = left_tree.query(reference, p=1, workers=-1)

    # Filter the once that are true one-to-one matches
    l_indices = np.arange(len(events))
    combined_indices = np.vstack([l_indices, l_nearest_neighbor]).T
    boolean_map = r_nearest_neighbor[l_nearest_neighbor] == l_indices
    valid_matches = combined_indices[boolean_map]

    # Check if the remaining matches are inside our Chebyshev tolerance distance.
    # If not, delete them.
    valid_matches_distance = l_nearest_distance[boolean_map]
    index_large_matches = np.where(valid_matches_distance > tolerance)[0]
    if index_large_matches.size > 0:
        # Minkowski with p = np.inf uses the Chebyshev distance
        output = (
            minkowski_distance(
                events[index_large_matches],
                reference[valid_matches[index_large_matches, 1]],
                p=np.inf,
            )
            > tolerance
        )

        valid_matches = np.delete(
            valid_matches, index_large_matches[output], axis=0
        )

    valid_matches = valid_matches
    # Add invalid pairs to the output array
    missing_l_indexes = np.setdiff1d(
        np.arange(len(events)), valid_matches[:, 0]
    )
    missing_l_matches = np.vstack(
        [missing_l_indexes, np.full(len(missing_l_indexes), np.nan)]
    ).T
    missing_r_indexes = np.setdiff1d(
        np.arange(len(reference)), valid_matches[:, 1]
    )
    missing_r_matches = np.vstack(
        [np.full(len(missing_r_indexes), np.nan), missing_r_indexes]
    ).T
    valid_matches = np.vstack(
        [valid_matches, missing_l_matches, missing_r_matches]
    )

    return valid_matches


def precision_recall_f1_score(matches: np.ndarray):
    if len(matches) == 0:
        return 0, 0, 0
    n_tp = np.sum((~np.isnan(matches)).all(axis=-1))
    len_events = np.sum(~np.isnan(matches[:, 0]))
    len_reference = np.sum(~np.isnan(matches[:, 1]))
    precision = n_tp / len_events if len_events > 0 else 0
    recall = n_tp / len_reference if len_reference > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )
    return precision, recall, f1


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
        high_pass_filter_cutoff_hz: float = 1,
    ):
        self.max_heart_rate_bpm = max_heart_rate_bpm
        self.min_r_peak_height_over_baseline = min_r_peak_height_over_baseline
        self.high_pass_filter_cutoff_hz = high_pass_filter_cutoff_hz

    @make_action_safe
    def detect(self, single_channel_ecg: pd.Series, sampling_rate_hz: float):
        ecg = single_channel_ecg.to_numpy().flatten()

        filtered_signal = self._filter(ecg, sampling_rate_hz)
        peak_positions = self._search_strategy(
            filtered_signal, sampling_rate_hz
        )

        self.r_peak_positions_ = pd.Series(peak_positions)
        return self

    def _search_strategy(
        self,
        filtered_signal: np.ndarray,
        sampling_rate_hz: float,
        use_height: bool = True,
    ) -> np.ndarray:
        # Calculate the minimal distance based on the expected heart rate
        min_distance_between_peaks = (
            1 / (self.max_heart_rate_bpm / 60) * sampling_rate_hz
        )

        height = None
        if use_height:
            height = self.min_r_peak_height_over_baseline
        peaks, _ = signal.find_peaks(
            filtered_signal, distance=min_distance_between_peaks, height=height
        )
        return peaks

    def _filter(
        self, ecg_signal: np.ndarray, sampling_rate_hz: float
    ) -> np.ndarray:
        sos = signal.butter(
            btype="high",
            N=self._HIGH_PASS_FILTER_ORDER,
            Wn=self.high_pass_filter_cutoff_hz,
            output="sos",
            fs=sampling_rate_hz,
        )
        return signal.sosfiltfilt(sos, ecg_signal)


class OptimizableQrsDetector(QRSDetector):
    min_r_peak_height_over_baseline: OptimizableParameter[float]
    r_peak_match_tolerance_s: HyperParameter[float]

    def __init__(
        self,
        max_heart_rate_bpm: float = 200.0,
        min_r_peak_height_over_baseline: float = 1.0,
        r_peak_match_tolerance_s: float = 0.01,
        high_pass_filter_cutoff_hz: float = 1,
    ):
        self.r_peak_match_tolerance_s = r_peak_match_tolerance_s
        super().__init__(
            max_heart_rate_bpm=max_heart_rate_bpm,
            min_r_peak_height_over_baseline=min_r_peak_height_over_baseline,
            high_pass_filter_cutoff_hz=high_pass_filter_cutoff_hz,
        )

    @make_optimize_safe
    def self_optimize(
        self,
        ecg_data: list[pd.Series],
        r_peaks: list[pd.Series],
        sampling_rate_hz: float,
    ):
        all_labels = []
        all_peak_heights = []
        for d, p in zip(ecg_data, r_peaks):
            filtered = self._filter(d.to_numpy().flatten(), sampling_rate_hz)
            # Find all potential peaks without the height threshold
            potential_peaks = self._search_strategy(
                filtered, sampling_rate_hz, use_height=False
            )
            # Determine the label for each peak, by matching them with our ground truth
            labels = np.zeros(potential_peaks.shape)
            matches = match_events_with_reference(
                events=np.atleast_2d(potential_peaks).T,
                reference=np.atleast_2d(p.to_numpy().astype(int)).T,
                tolerance=self.r_peak_match_tolerance_s * sampling_rate_hz,
            )
            tp_matches = matches[(~np.isnan(matches)).all(axis=1), 0].astype(
                int
            )
            labels[tp_matches] = 1
            labels = labels.astype(bool)
            all_labels.append(labels)
            all_peak_heights.append(filtered[potential_peaks])
        all_labels = np.hstack(all_labels)
        all_peak_heights = np.hstack(all_peak_heights)
        # We "brute-force" a good cutoff by testing a bunch of thresholds and then calculating the Youden Index for
        # each.
        fpr, tpr, thresholds = roc_curve(all_labels, all_peak_heights)
        youden_index = tpr - fpr
        # The best Youden index gives us a balance between sensitivity and specificity.
        self.min_r_peak_height_over_baseline = thresholds[
            np.argmax(youden_index)
        ]
        return self
