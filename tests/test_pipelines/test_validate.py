from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from tests.test_pipelines.conftest import (
    DummyDataset,
    DummyGroupedDataset,
    DummyOptimizablePipeline,
    DummyPipeline,
    dummy_single_score_func,
)
from tpcp import Dataset, OptimizableParameter, OptimizablePipeline
from tpcp.exceptions import OptimizationError, TestError
from tpcp.optimize import DummyOptimize, Optimize
from tpcp.validate import DatasetSplitter, cross_validate, validate
from tpcp.validate._scorer import Scorer, _validate_scorer


class CustomOptimizablePipelineWithOptiError(OptimizablePipeline):
    optimized: OptimizableParameter[bool]

    def __init__(self, error_fold, optimized=False):
        self.error_fold = error_fold
        self.optimized = optimized

    def self_optimize(self, dataset: Dataset, **kwargs):
        if (self.error_fold,) not in dataset.group_labels:
            raise ValueError("This is an error")
        return self


class CustomOptimizablePipelineWithRunError(OptimizablePipeline):
    optimized: OptimizableParameter[bool]

    def __init__(self, error_fold, optimized=False):
        self.error_fold = error_fold
        self.optimized = optimized

    def run(self, dataset: Dataset):
        condition = (self.error_fold,) == dataset.group_label
        if condition:
            raise ValueError("This is an error")
        self.optimized = True
        return self


class TestValidate:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"n_jobs": 5, "verbose": 1, "pre_dispatch": 5, "progress_bar": False},
            {"n_jobs": 2, "progress_bar": False},
        ],
    )
    def test_scores(self, kwargs):
        """Test that the scores are calculated correctly and the result dict has the expected structure."""
        ds = DummyDataset()
        pipeline = DummyPipeline()
        results = validate(pipeline, ds, scoring=dummy_single_score_func, **kwargs)

        results_df = pd.DataFrame(results)

        assert len(results_df) == 1
        assert set(results.keys()) == {
            "data_labels",
            "agg__score",
            "single__score",
            "debug__score_time",
        }
        result_row = results_df.iloc[0]  # result_df only has one row
        assert all(len(result_row[key]) == len(ds) for key in ["data_labels", "single__score"])
        assert all(isinstance(result_row[key], float) for key in ["agg__score", "debug__score_time"])

        # The dummy scorer is returning the dataset group label -> The datapoint id is also the result
        all_ids = np.array(ds.group_labels).flatten()
        assert all(np.array(result_row["data_labels"]).flatten() == np.array(result_row["single__score"]))
        assert result_row["agg__score"] == np.mean(all_ids)

    @pytest.mark.parametrize(
        "multiprocess_args",
        [{"n_jobs": 1}, {"verbose": 1}, {"pre_dispatch": 1}, {"progress_bar": False}, {"n_jobs": 1, "verbose": 1}],
    )
    def test_arguments_set_twice_error(self, multiprocess_args):
        """Test that an error is raised when a scorer object is passed for scoring,
        and multiprocessing args are set simultaneously.
        """
        ds = DummyDataset()
        pipeline = DummyPipeline()
        scorer = _validate_scorer(dummy_single_score_func)
        with pytest.raises(ValueError):
            validate(pipeline, ds, scoring=scorer, **multiprocess_args)

    @patch("tpcp.validate._validate._score", autospec=True)
    @pytest.mark.parametrize(
        "multiprocess_args",
        [
            {"n_jobs": 5},
            {"n_jobs": None, "pre_dispatch": 5},
            {"n_jobs": 1, "verbose": 1, "pre_dispatch": 1, "progress_bar": False},
        ],
    )
    def test_multiprocessing_parameters_set_correctly(self, mock_score, multiprocess_args):
        """Check if multiprocessing arguments are passed to scorer correctly."""
        validate(DummyPipeline(), DummyDataset(), scoring=dummy_single_score_func, **multiprocess_args)
        assert mock_score.call_args is not None  # _score function was called
        assert len(mock_score.call_args.args) >= 3  # _score function has 3 positional args
        scorer = mock_score.call_args.args[2]
        assert isinstance(scorer, Scorer)
        assert all(scorer.get_params()[arg] == val for arg, val in multiprocess_args.items())


class TestCrossValidate:
    @pytest.mark.filterwarnings("ignore::tpcp.exceptions.PotentialUserErrorWarning")
    def test_optimize_called(self):
        """Test that optimize of the pipeline is called correctly."""
        ds = DummyDataset()
        pipeline = DummyOptimizablePipeline()

        # We use len(ds) splits, effectively a leave one out CV for testing.
        cv = KFold(n_splits=len(ds))
        train, test = zip(*cv.split(ds))
        with patch.object(DummyOptimizablePipeline, "self_optimize", return_value=pipeline) as mock:
            mock.__name__ = "self_optimize"
            mock.__self__ = "bla"  # We simulate a bound method
            cross_validate(Optimize(pipeline), ds, cv=cv, scoring=lambda x, y: 1)

        assert mock.call_count == len(train)
        for expected, actual in zip(train, mock.call_args_list):
            pd.testing.assert_frame_equal(ds[expected].index, actual[0][0].index)

    def test_run_called(self):
        """Test that optimize of the pipeline is called correctly."""

        def scoring(pipe, ds):
            pipe.run(ds)
            return 1

        ds = DummyDataset()
        pipeline = DummyOptimizablePipeline()

        # We want to have two datapoints in the test set sometimes
        cv = KFold(n_splits=len(ds) // 2)
        train, test = zip(*cv.split(ds))
        with patch.object(DummyOptimizablePipeline, "run", return_value=pipeline) as mock:
            cross_validate(Optimize(pipeline), ds, cv=cv, scoring=scoring)

        test_flat = [t for split in test for t in split]
        assert mock.call_count == len(test_flat)
        for expected, actual in zip(test_flat, mock.call_args_list):
            pd.testing.assert_frame_equal(ds[expected].index, actual[0][0].index)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"n_jobs": 5, "verbose": 1, "pre_dispatch": 5, "progress_bar": False},
            {"n_jobs": 2, "progress_bar": False},
        ],
    )
    def test_single_score(self, kwargs):
        ds = DummyDataset()
        # Then we use len(ds) splits, effectively a leave one out CV for testing.
        cv = KFold(n_splits=len(ds))

        results = cross_validate(
            Optimize(DummyOptimizablePipeline()),
            ds,
            scoring=dummy_single_score_func,
            cv=cv,
            return_train_score=True,
            **kwargs,
        )
        results_df = pd.DataFrame(results)

        assert len(results_df) == 5  # n folds
        assert set(results.keys()) == {
            "train__data_labels",
            "test__data_labels",
            "test__agg__score",
            "test__single__score",
            "train__agg__score",
            "train__single__score",
            "debug__score_time",
            "debug__optimize_time",
        }
        assert all(len(v) == len(ds) - 1 for v in results_df["train__data_labels"])
        assert all(len(v) == len(ds) - 1 for v in results_df["train__single__score"])
        assert all(len(v) == 1 for v in results_df["test__data_labels"])
        assert all(len(v) == 1 for v in results_df["test__single__score"])
        # The dummy scorer is returning the dataset group label -> The datapoint id is also the result
        for i, r in results_df.iterrows():
            all_ids = np.array(ds.group_labels).flatten()
            assert r["test__data_labels"] == [(i,)]
            assert r["test__data_labels"][0][0] == r["test__single__score"][0]
            assert r["test__agg__score"] == i
            all_ids = all_ids[all_ids != i]
            assert all(np.array(r["train__data_labels"]).flatten() == all_ids)
            assert all(np.array(r["train__data_labels"]).flatten() == np.array(r["train__single__score"]))
            assert r["train__agg__score"] == np.mean(all_ids)

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"return_optimizer": True}, ("optimizer",)),
            ({"return_train_score": True}, ("train__agg__score", "train__single__score")),
        ],
    )
    def test_return_elements(self, kwargs, expected):
        results = cross_validate(Optimize(DummyOptimizablePipeline()), DummyDataset(), scoring=dummy_single_score_func)
        results_additionally = cross_validate(
            Optimize(DummyOptimizablePipeline()), DummyDataset(), scoring=dummy_single_score_func, **kwargs
        )

        assert set(results_additionally.keys()) - set(results.keys()) == set(expected)

    def test_returned_optimizer_per_fold_independent(self):
        """Double check that the optimizer is cloned correctly."""
        optimizer = Optimize(DummyOptimizablePipeline())
        results = cross_validate(
            Optimize(DummyOptimizablePipeline()), DummyDataset(), scoring=dummy_single_score_func, return_optimizer=True
        )
        optimizers = results["optimizer"]
        for o in optimizers:
            assert o is not optimizer

    @pytest.mark.parametrize("error_fold", [0, 2])
    def test_cross_validate_opti_error(self, error_fold):
        with pytest.raises(OptimizationError) as e:
            cross_validate(
                Optimize(CustomOptimizablePipelineWithOptiError(error_fold=error_fold)),
                DummyDataset(),
                scoring=dummy_single_score_func,
                cv=5,
            )

        assert f"This error occurred in fold {error_fold}" in str(e.value)

    @pytest.mark.parametrize("error_fold", [0, 2])
    def test_cross_validate_test_error(self, error_fold):
        def simple_scorer(pipeline, data_point):
            pipeline.run(data_point)
            return data_point.group_labels[0]

        with pytest.raises(TestError) as e:
            cross_validate(
                DummyOptimize(CustomOptimizablePipelineWithRunError(error_fold=error_fold)),
                DummyDataset(),
                scoring=simple_scorer,
                cv=5,
            )

        assert f"This error occurred in fold {error_fold}" in str(e.value)

    @pytest.mark.parametrize("return_train_score", [True, False])
    def test_cross_validate_train_error(self, return_train_score):
        """Test that a different error message is used, if the error occurs during evaluating the train set.

        We only trigger that for fold 0. Otherwise it would be to annoying to write a testcase for.
        """

        def simple_scorer(pipeline, data_point):
            pipeline.run(data_point)
            return data_point.group_labels[0]

        with pytest.raises(TestError) as e:
            cross_validate(
                # We need to select any fold other than 0 as error fold to get the error triggered during training
                DummyOptimize(CustomOptimizablePipelineWithRunError(error_fold=1)),
                DummyDataset(),
                scoring=simple_scorer,
                return_train_score=return_train_score,
                cv=5,
            )

        if return_train_score:
            assert "This error occurred in fold 0" in str(e.value)
            assert "train-set" in str(e.value)
        else:
            assert "This error occurred in fold 1" in str(e.value)
            assert "test-set" in str(e.value)

    def test_cross_validate_optimizer_are_cloned(self):
        results = cross_validate(
            Optimize(DummyOptimizablePipeline()),
            DummyDataset(),
            scoring=dummy_single_score_func,
            return_optimizer=True,
            cv=5,
        )

        assert len(results["optimizer"]) == 5
        assert len({id(o) for o in results["optimizer"]}) == 5


class TestTpcpSplitter:
    def test_normal_k_fold(self):
        ds = DummyGroupedDataset()
        splitter = DatasetSplitter(base_splitter=KFold(n_splits=5))
        # This should be identical to just calling the splitter directly
        splits_expected = list(KFold(n_splits=5).split(ds))

        splits = list(splitter.split(ds))

        for (train_expected, test_expected), (train, test) in zip(splits_expected, splits):
            assert train_expected.tolist() == train.tolist()
            assert test_expected.tolist() == test.tolist()

    def test_normal_k_fold_with_groupby_ignored(self):
        ds = DummyGroupedDataset()
        splitter = DatasetSplitter(base_splitter=KFold(n_splits=5), groupby="v1")
        # This should be identical to just calling the splitter directly
        splits_expected = list(KFold(n_splits=5).split(ds))

        splits = list(splitter.split(ds))

        for (train_expected, test_expected), (train, test) in zip(splits_expected, splits):
            assert train_expected.tolist() == train.tolist()
            assert test_expected.tolist() == test.tolist()

    def test_normal_group_k_fold(self):
        ds = DummyGroupedDataset()
        splitter = DatasetSplitter(base_splitter=GroupKFold(n_splits=3), groupby="v1")
        # This should be identical to just calling the splitter directly
        splits_expected = list(GroupKFold(n_splits=3).split(ds, groups=ds.create_string_group_labels("v1")))

        splits = list(splitter.split(ds))

        for (train_expected, test_expected), (train, test) in zip(splits_expected, splits):
            assert train_expected.tolist() == train.tolist()
            assert test_expected.tolist() == test.tolist()

    def test_normal_stratified_k_fold(self):
        ds = DummyGroupedDataset()
        splitter = DatasetSplitter(base_splitter=StratifiedKFold(n_splits=3), stratify="v1")
        # This should be identical to just calling the splitter directly
        splits_expected = list(StratifiedKFold(n_splits=3).split(ds, y=ds.create_string_group_labels("v1")))

        splits = list(splitter.split(ds))

        for (train_expected, test_expected), (train, test) in zip(splits_expected, splits):
            assert train_expected.tolist() == train.tolist()
            assert test_expected.tolist() == test.tolist()
