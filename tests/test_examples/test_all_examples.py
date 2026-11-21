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
    assert_almost_equal(results["agg__f1_score"], [0.7198637, 0.7169006, 0.7089728])


def test_optimizable_pipeline():
    from examples.parameter_optimization._02_optimizable_pipelines import optimized_results, results

    assert len(results.r_peak_positions_) == 30
    assert len(optimized_results.r_peak_positions_) == 393


def test_gridsearchcv():
    from examples.parameter_optimization._03_gridsearch_cv import r_peaks, results

    assert_array_equal(r_peaks[:3], [77, 370, 663])
    assert_almost_equal(results["mean__test__agg__f1_score"], [0.8640027, 0.861629, 0.8655343])


def test_pure_parameters():
    from examples.parameter_optimization._06_pure_parameters import results, training_calls

    assert len(training_calls) == 6
    scores = results.loc[
        results["param__algorithm__high_pass_filter_cutoff_hz"] == 0.5,
        "mean__test__agg__score",
    ]
    assert scores.min() < scores.max()


def test_validation():
    from examples.validation._01_validation import results

    assert_almost_equal(results["agg__f1_score"], [0.7089727])


def test_cross_validate():
    from examples.validation._02_cross_validation import results

    assert_almost_equal(results["test__agg__f1_score"], [0.9770585, 0.7108303, 0.9250665])


def test_advanced_cross_validate(snapshot):
    from examples.validation._04_advanced_cross_validation import result_df_grouped, result_df_stratified

    snapshot.assert_match(
        result_df_grouped["test__data_labels"].explode().explode().to_frame().rename_axis("fold_id").reset_index()
    )
    snapshot.assert_match(
        result_df_stratified["test__data_labels"].explode().explode().to_frame().rename_axis("fold_id").reset_index()
    )


def test_train_dataset_transform():
    from examples.validation._05_train_dataset_transform import (
        AugmentedImageDataset,
        ImageDataset,
        augmentation_results,
        augmentation_training_sizes,
        augmented_dataset,
        raw_dataset,
        rotate_training_images,
        smoke_test_results,
        smoke_test_training_sizes,
    )

    assert isinstance(raw_dataset, ImageDataset)
    assert isinstance(augmented_dataset, AugmentedImageDataset)
    assert augmented_dataset.wrapped_dataset is not raw_dataset
    assert_array_equal(raw_dataset[0].image, [[1, 0, 0], [1, 1, 0], [1, 0, 0]])
    rotated_datapoint = augmented_dataset.get_subset(sample_id=0, rotation_deg=90)
    assert rotated_datapoint.label == raw_dataset[0].label
    assert_array_equal(
        rotated_datapoint.image,
        [[0, 0, 0], [0, 1, 0], [1, 1, 1]],
    )

    grouped_dataset = raw_dataset.groupby("sample_id")
    grouped_augmented_dataset = rotate_training_images(grouped_dataset)
    assert len(grouped_augmented_dataset) == 24
    assert grouped_augmented_dataset.groupby_cols == ["sample_id", "rotation_deg"]
    assert grouped_augmented_dataset[0].label == grouped_dataset[0].label
    assert_array_equal(grouped_augmented_dataset[0].image, grouped_dataset[0].image)

    assert augmentation_training_sizes == [16, 16, 16]
    assert smoke_test_training_sizes == [2, 2, 2]
    assert all(len(labels) == 2 for labels in augmentation_results["test__data_labels"])
    assert all(len(labels) == 2 for labels in smoke_test_results["test__data_labels"])


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
        agg_final_agg,
        all_median_results_agg,
        complicated_agg,
        complicated_single_no_raw,
        macro_agg,
        median_results_agg,
        multi_agg_agg,
        no_agg_agg,
    )

    assert "n_labels" not in no_agg_agg

    assert_almost_equal(median_results_agg["f1_score"], 0.7089728)
    assert_almost_equal(median_results_agg["median_f1_score"], 0.9173713)
    assert_almost_equal(all_median_results_agg["f1_score"], median_results_agg["median_f1_score"])
    assert_almost_equal(multi_agg_agg["f1_score__mean"], median_results_agg["f1_score"])
    assert_almost_equal(multi_agg_agg["f1_score__std"], 0.39387732846763174)

    assert macro_agg["f1_score__macro"] == 0.7089727629059107

    assert_almost_equal(complicated_agg["per_sample__f1_score"], 0.8172557027823545)
    assert_almost_equal(complicated_agg["f1_score"], median_results_agg["f1_score"])
    for i in range(1, 4):
        assert f"f1_score__group_{i}" in macro_agg

    assert "per_sample" not in complicated_single_no_raw

    assert complicated_agg == agg_final_agg


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
    assert np.all(cv_results["test__agg__per_sample__accuracy"] > 0.8)


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


def test_pretrained_models_example():
    # Dummy test to make sure that the example is executed as part of the test
    # suite. This does not test any specific functionality.
    from examples.recipies._06_pretrained_models import algo_with_clean_lab_model

    assert algo_with_clean_lab_model.model == "model_clean_lab_data"


def test_warning_error_context_example():
    # Import the gallery example to ensure that it executes without errors.
    import examples.recipies._07_warning_error_context  # noqa: F401


def test_complex_clinical_evaluation():
    """Compare paired outer home tests and participant-isolated inner folds."""
    from examples.validation._08_complex_clinical_evaluation import cv_results, dataset, holdout_results, searches

    assert len(dataset) == 36
    all_b = set(dataset.get_subset(origin="B").group_labels)
    expected_home = set(dataset.get_subset(origin="A", condition="home").group_labels)
    for results, n_folds in ((holdout_results, 1), (cv_results, 3)):
        reference_test_folds = None
        for name, result in results.items():
            hospital_only = name.startswith("hospital_only_")
            assert len(result["test__agg__f1"]) == n_folds
            assert np.all((0 <= result["test__agg__f1"]) & (result["test__agg__f1"] <= 1))
            test_folds = [set(labels) for labels in result["test__data_labels"]]
            if reference_test_folds is None:
                reference_test_folds = test_folds
            assert test_folds == reference_test_folds
            if n_folds == 3:
                tested = [label for labels in result["test__data_labels"] for label in labels]
                assert len(tested) == len(set(tested)) == 12
                assert set(tested) == expected_home
            for train_labels, test_labels, search in zip(
                result["train__data_labels"], result["test__data_labels"], result["optimizer"], strict=True
            ):
                train_ids = {label.participant for label in train_labels}
                test_ids = {label.participant for label in test_labels}
                assert len(train_ids) == 10 and len(test_ids) == 2
                assert train_ids.isdisjoint(test_ids)
                assert all_b.issubset(train_labels)
                assert len(train_labels) == (20 if hospital_only else 28)
                assert {label.condition for label in train_labels} == (
                    {"hospital"} if hospital_only else {"hospital", "home"}
                )
                assert len(test_labels) == 4 and set(test_labels).issubset(expected_home)
                if name.endswith("learnable"):
                    detector = search.optimized_pipeline_.algorithm
                    assert np.all(np.isfinite(detector.template))
                    assert len(detector.template) == 2 * search.best_params_["algorithm__template_radius"] + 1
                    assert len(search.cv_results_["params"]) == 4
                    for split in range(2):
                        for inner_train, inner_test in zip(
                            search.cv_results_[f"split{split}__train__data_labels"],
                            search.cv_results_[f"split{split}__test__data_labels"],
                            strict=True,
                        ):
                            assert set(inner_train).isdisjoint(inner_test)
                            assert set(inner_train + inner_test) == set(train_labels)
                            assert {label.origin for label in inner_train} == {"A", "B"}
                            assert {label.origin for label in inner_test} == {"A", "B"}
                            assert {label.participant for label in inner_train}.isdisjoint(
                                label.participant for label in inner_test
                            )
                            expected_conditions = {"hospital"} if hospital_only else {"hospital", "home"}
                            assert {label.condition for label in inner_train} == expected_conditions
                            assert {label.condition for label in inner_test} == expected_conditions
                else:
                    assert len(search.gs_results_["params"]) == 3
                    assert {p["algorithm__threshold"] for p in search.gs_results_["params"]} == {0.5, 0.9, 1.1}
    assert searches["learnable"].pipeline.algorithm.template is None


@pytest.mark.parametrize(
    ("events", "reference", "tolerance", "expected"),
    [
        ([102, 199, 500], [100, 200, 300], 5, (2 / 3, 2 / 3, 2 / 3)),
        ([99, 100], [100], 5, (0.5, 1.0, 2 / 3)),
        ([205, 95], [100, 200], 5, (1.0, 1.0, 1.0)),
        ([0, 50], [100], 5, (0.0, 0.0, 0.0)),
        ([100], [100, 200], 5, (1.0, 0.5, 2 / 3)),
        ([0, 9], [8, 17], 8, (1.0, 1.0, 1.0)),
        ([], [100], 5, (0.0, 0.0, 0.0)),
        ([100], [], 5, (0.0, 0.0, 0.0)),
        ([], [], 5, (0.0, 0.0, 0.0)),
    ],
)
def test_complex_clinical_event_scoring(events, reference, tolerance, expected):
    """Count events once, with a sample tolerance and defined empty-input scores."""
    from examples.validation._08_complex_clinical_evaluation import event_detection_scores

    result = event_detection_scores(np.array(events), np.array(reference), tolerance)
    assert (result["precision"], result["recall"], result["f1"]) == pytest.approx(expected)


def test_complex_clinical_detector_learning():
    """The learned event template survives cloning and ignores amplitude changes."""
    from examples.validation._08_complex_clinical_evaluation import TemplateMatchingDetector

    samples = np.arange(500)
    signal = pd.DataFrame({"x": np.cos(2 * np.pi * (samples - 50) / 100)})
    events = np.array([50, 150, 250, 350, 450])
    detector = TemplateMatchingDetector().self_optimize([signal], [events])
    assert len(detector.template) == 41
    assert_almost_equal(np.linalg.norm(detector.template), 1)
    assert_array_equal(detector.clone().detect(signal * 3).events_, events)


def test_complex_clinical_detector_interface():
    """Detectors must implement prediction; fitting remains optional."""
    from examples.validation._08_complex_clinical_evaluation import (
        BaseEventDetector,
        EventPipeline,
        PeakDetector,
        dataset,
    )

    with pytest.raises(NotImplementedError):
        BaseEventDetector().detect(dataset[0].data)
    with pytest.raises(NotImplementedError, match="does not support fitting"):
        EventPipeline(PeakDetector()).self_optimize(dataset[:1])
