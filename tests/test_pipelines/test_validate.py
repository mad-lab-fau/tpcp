from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from tests.test_pipelines.conftest import (
    DummyDataset,
    DummyGroupedDataset,
    DummyOptimizablePipeline,
    dummy_single_score_func,
)
from tpcp import Dataset, OptimizableParameter, OptimizablePipeline
from tpcp.exceptions import OptimizationError, TestError
from tpcp.optimize import DummyOptimize, Optimize
from tpcp.validate import cross_validate


class CustomOptimizablePipelineWithOptiError(OptimizablePipeline):
    optimized: OptimizableParameter[bool]

    def __init__(self, error_fold, optimized=False):
        self.error_fold = error_fold
        self.optimized = optimized

    def self_optimize(self, dataset: Dataset, **kwargs):
        if self.error_fold not in dataset.groups:
            raise ValueError("This is an error")
        return self


class CustomOptimizablePipelineWithRunError(OptimizablePipeline):
    optimized: OptimizableParameter[bool]

    def __init__(self, error_fold, optimized=False):
        self.error_fold = error_fold
        self.optimized = optimized

    def run(self, dataset: Dataset):
        condition = self.error_fold == dataset.group
        if condition:
            raise ValueError("This is an error")
        self.optimized = True
        return self


class TestCrossValidate:
    @pytest.mark.filterwarnings("ignore::tpcp.exceptions.PotentialUserErrorWarning")
    def test_optimize_called(self):
        """Test that optimize of the pipeline is called correctly."""
        ds = DummyDataset()
        pipeline = DummyOptimizablePipeline()

        # The we use len(ds) splits, effectively a leave one our CV for testing.
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

    def test_single_score(self):
        ds = DummyDataset()
        # The we use len(ds) splits, effectively a leave one our CV for testing.
        cv = KFold(n_splits=len(ds))

        results = cross_validate(
            Optimize(DummyOptimizablePipeline()), ds, scoring=dummy_single_score_func, cv=cv, return_train_score=True
        )
        results_df = pd.DataFrame(results)

        assert len(results_df) == 5  # n folds
        assert set(results.keys()) == {
            "train_data_labels",
            "test_data_labels",
            "test_score",
            "test_single_score",
            "train_score",
            "train_single_score",
            "score_time",
            "optimize_time",
        }
        assert all(len(v) == len(ds) - 1 for v in results_df["train_data_labels"])
        assert all(len(v) == len(ds) - 1 for v in results_df["train_single_score"])
        assert all(len(v) == 1 for v in results_df["test_data_labels"])
        assert all(len(v) == 1 for v in results_df["test_single_score"])
        # The dummy scorer is returning the dataset group id -> The datapoint id is also the result
        for i, r in results_df.iterrows():
            all_ids = ds.groups
            assert r["test_data_labels"] == [i]
            assert r["test_data_labels"] == r["test_single_score"]
            assert r["test_score"] == i
            all_ids.remove(i)
            assert r["train_data_labels"] == all_ids
            assert all(np.array(r["train_data_labels"]) == np.array(r["train_single_score"]))
            assert r["train_score"] == np.mean(all_ids)

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        (
            ({"return_optimizer": True}, ("optimizer",)),
            ({"return_train_score": True}, ("train_score", "train_single_score")),
        ),
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

    @pytest.mark.parametrize("propagate", (True, False))
    def test_propagate_groups(self, propagate):
        pipeline = DummyOptimizablePipeline()
        dataset = DummyGroupedDataset()
        groups = dataset.create_group_labels("v1")
        # With 3 splits, each group get its own split -> so basically only "a", only "b", and only "c"
        cv = GroupKFold(n_splits=3)

        dummy_results = Optimize(pipeline).optimize(dataset)
        with patch.object(Optimize, "optimize", return_value=dummy_results) as mock:
            cross_validate(
                Optimize(pipeline), dataset, cv=cv, scoring=lambda x, y: 1, groups=groups, propagate_groups=propagate
            )

        assert mock.call_count == 3
        for call, label in zip(mock.call_args_list, "cba"):
            train_labels = "abc".replace(label, "")
            if propagate:
                assert set(np.unique(call[1]["groups"])) == set(train_labels)
            else:
                assert "groups" not in call[1]

    @pytest.mark.parametrize("propagate", (True, False))
    def test_propagate_mock_labels(self, propagate):
        pipeline = DummyOptimizablePipeline()
        dataset = DummyGroupedDataset()
        groups = dataset.create_group_labels("v1")
        # With 5 folds, we expect exactly on "a", one "b", and one "c" in each fold
        cv = StratifiedKFold(n_splits=5)

        dummy_results = Optimize(pipeline).optimize(dataset)
        with patch.object(Optimize, "optimize", return_value=dummy_results) as mock:
            cross_validate(
                Optimize(pipeline),
                dataset,
                cv=cv,
                scoring=lambda x, y: 1,
                mock_labels=groups,
                propagate_mock_labels=propagate,
                propagate_groups=False,
            )

        assert mock.call_count == 5
        for call in mock.call_args_list:
            if propagate:
                assert len(np.unique(call[1]["mock_labels"])) == 3
                assert set(np.unique(call[1]["mock_labels"])) == set("abc")
            else:
                assert "mock_labels" not in call[1]

    @pytest.mark.parametrize("error_fold", (0, 2))
    def test_cross_validate_opti_error(self, error_fold):

        with pytest.raises(OptimizationError) as e:
            cross_validate(
                Optimize(CustomOptimizablePipelineWithOptiError(error_fold=error_fold)),
                DummyDataset(),
                scoring=dummy_single_score_func,
                cv=5,
            )

        assert f"This error occurred in fold {error_fold}" in str(e.value)

    @pytest.mark.parametrize("error_fold", (0, 2))
    def test_cross_validate_test_error(self, error_fold):
        def simple_scorer(pipeline, data_point):
            pipeline.run(data_point)
            return data_point.groups[0]

        with pytest.raises(TestError) as e:
            cross_validate(
                DummyOptimize(CustomOptimizablePipelineWithRunError(error_fold=error_fold)),
                DummyDataset(),
                scoring=simple_scorer,
                cv=5,
            )

        assert f"This error occurred in fold {error_fold}" in str(e.value)
