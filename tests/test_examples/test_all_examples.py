# This is needed to avoid plots to open
import matplotlib
from numpy.testing import assert_array_equal, assert_almost_equal

matplotlib.use("Agg")


def test_custom_dataset():
    # There is not really anything specific, we want to test here, so we just run everything and check that there are
    # no errors.
    import examples.datasets._01_datasets_basics  # noqa


def test_real_life_dataset():
    from examples.datasets._02_datasets_real_world_example import ECGExampleData, data_path

    dataset = ECGExampleData(data_path=data_path)
    assert dataset.index.shape == (12, 2)

    # Just test that accessing them does not produce any errors
    subset = dataset[0]
    subset.data
    subset.sampling_rate_hz
    subset.labeled_r_peaks_
    subset.pvc_positions_
    subset.r_peak_positions_


def test_qrs_algorithm():
    from examples.algorithms._01_algorithms_qrs_detection import algorithm

    assert algorithm.min_r_peak_height_over_baseline == 1.1229521656012118
    assert_array_equal(algorithm.r_peak_positions_[:3], [197, 459, 708])


def test_gridsearch():
    from examples.parameter_optimization._01_gridsearch import r_peaks, results

    assert_array_equal(r_peaks[:3], [77, 370, 663])
    assert_almost_equal(results["f1_score"], [0.58380606, 0.57964556, 0.5699779])
