# This is needed to avoid plots to open
import matplotlib
import pandas as pd
from numpy.testing import assert_almost_equal, assert_array_equal

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

    assert algorithm.min_r_peak_height_over_baseline == 0.5434583752370183
    assert_array_equal(algorithm.r_peak_positions_[:3], [197, 459, 708])


def test_gridsearch():
    from examples.parameter_optimization._01_gridsearch import r_peaks, results

    assert_array_equal(r_peaks[:3], [77, 370, 663])
    assert_almost_equal(results["f1_score"], [0.7198637, 0.7169006, 0.7089728])


def test_optimizable_pipeline():
    from examples.parameter_optimization._02_optimizable_pipelines import optimized_results, results

    assert len(results.r_peak_positions_) == 30
    assert len(optimized_results.r_peak_positions_) == 393


def test_gridsearchcv():
    from examples.parameter_optimization._03_gridsearch_cv import r_peaks, results

    assert_array_equal(r_peaks[:3], [77, 370, 663])
    assert_almost_equal(results["mean_test_f1_score"], [0.8640027, 0.861629, 0.8655343])


def test_cross_validate():
    from examples.validation._01_cross_validation import results

    assert_almost_equal(results["test_f1_score"], [0.9770585, 0.7108303, 0.9250665])


def test_optuna():
    from examples.parameter_optimization._04_custom_optuna_optimizer import opti, opti_early_stop

    assert opti.best_params_ == {
        "algorithm__min_r_peak_height_over_baseline": 0.4,
        "algorithm__high_pass_filter_cutoff_hz": 0.4,
    }
    assert opti.best_score_ == 0.858757056619628

    # Check number of pruned trials
    assert pd.DataFrame(opti_early_stop.search_results_)["score"].isna().sum() == 6
    assert opti_early_stop.best_params_ == {
        "algorithm__min_r_peak_height_over_baseline": 0.4,
        "algorithm__high_pass_filter_cutoff_hz": 0.4,
    }
    assert opti_early_stop.best_score_ == 0.858757056619628
