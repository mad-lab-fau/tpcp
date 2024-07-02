# This is needed to avoid plots to open
import matplotlib
import numpy as np
import pandas as pd
import pytest
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
    assert_almost_equal(results["mean__test__f1_score"], [0.8640027, 0.861629, 0.8655343])


def test_validation():
    from examples.validation._01_validation import results

    assert_almost_equal(results["f1_score"], [0.7089727])


def test_cross_validate():
    from examples.validation._02_cross_validation import results

    assert_almost_equal(results["test__f1_score"], [0.9770585, 0.7108303, 0.9250665])


def test_advanced_cross_validate(snapshot):
    from examples.validation._04_advanced_cross_validation import result_df_grouped, result_df_stratified

    snapshot.assert_match(
        result_df_grouped["test__data_labels"].explode().explode().to_frame().rename_axis("fold_id").reset_index()
    )
    snapshot.assert_match(
        result_df_stratified["test__data_labels"].explode().explode().to_frame().rename_axis("fold_id").reset_index()
    )


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


def test_optuna_search():
    from examples.parameter_optimization._05_optuna_search import opti

    assert opti.best_params_ == {
        "algorithm__min_r_peak_height_over_baseline": 0.4,
        "algorithm__high_pass_filter_cutoff_hz": 0.4,
    }
    assert opti.best_score_ == 0.858757056619628


def test_dataclasses():
    from examples.recipies._02_dataclasses import nested_object_is_different

    assert nested_object_is_different is True


def test_custom_scorer():
    from examples.validation._03_custom_scorer import (
        baseline_results_agg,
        complicated_agg,
        complicated_single_no_raw,
        group_weighted_agg,
        median_results_agg,
        multi_agg_agg,
        no_agg_agg,
        partial_median_results_agg,
    )

    assert_almost_equal(baseline_results_agg["f1_score"], 0.7089728)
    assert_almost_equal(median_results_agg["f1_score"], 0.9173713)
    assert_almost_equal(partial_median_results_agg["f1_score"], median_results_agg["f1_score"])
    assert_almost_equal(partial_median_results_agg["precision"], baseline_results_agg["precision"])
    assert_almost_equal(multi_agg_agg["f1_score__mean"], baseline_results_agg["f1_score"])
    assert_almost_equal(multi_agg_agg["f1_score__std"], 0.39387732846763174)

    assert_almost_equal(complicated_agg["f1_score"], baseline_results_agg["f1_score"])
    assert_almost_equal(complicated_agg["per_sample__f1_score"], 0.8172557027823545)

    assert "f1_score" not in no_agg_agg

    assert group_weighted_agg["f1_score__group_mean"] == 0.7089727629059107
    for i in range(1, 4):
        assert f"f1_score__group_{i}" in group_weighted_agg

    assert "per_sample" not in complicated_single_no_raw


def test_composite_objects():
    from examples.recipies._03_composite_objects import workflow_instance

    assert workflow_instance.get_params()["pipelines__pipe1__param"] == 2
    assert workflow_instance.get_params()["pipelines__pipe2__param2"] == 4


def test_optimization_info():
    from examples.recipies._05_optimization_info import optimizer

    assert len(optimizer.optimization_info_["all_thresholds"]) == 935


def test_tensorflow_example():
    pytest.importorskip("tensorflow")

    from examples.integrations._01_tensorflow import cv_results

    # It seems to be impossible to run the tensorflow example in a deterministic way, across different machines and
    # Python versions.
    # We therefore just check if the performance is larger 0.8, which is the case for all runs we have seen so far.
    assert np.all(cv_results["test__per_sample__accuracy"] > 0.8)


def test_caching_example():
    # We just import the example to see if it runs without errors
    import examples.recipies._01_caching  # noqa


def test_typed_iterator_example():
    from examples.recipies._04_typed_iterator import custom_iterator, qrs_iterator

    assert len(qrs_iterator.results_.r_peak_positions) == 17782
    assert sum(qrs_iterator.results_.n_r_peaks.values()) == 17782
    assert len(qrs_iterator.raw_results_) == 12

    assert len(custom_iterator.results_.n_samples) == 2
    assert len(custom_iterator.raw_results_) == 2
