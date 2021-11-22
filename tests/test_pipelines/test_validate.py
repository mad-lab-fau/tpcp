from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from tests.test_pipelines.conftest import DummyDataset, DummyOptimizablePipeline, dummy_single_score_func
from tpcp.optimize import Optimize
from tpcp.validation import cross_validate


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
            assert all(r["train_data_labels"] == r["train_single_score"])
            assert r["train_score"] == np.mean(all_ids)

    @pytest.mark.parametrize(
        "kwargs,expected",
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
        """Double check that the optimizer is cloned correctly"""
        optimizer = Optimize(DummyOptimizablePipeline())
        results = cross_validate(
            Optimize(DummyOptimizablePipeline()), DummyDataset(), scoring=dummy_single_score_func, return_optimizer=True
        )
        optimizers = results["optimizer"]
        for o in optimizers:
            assert o is not optimizer
