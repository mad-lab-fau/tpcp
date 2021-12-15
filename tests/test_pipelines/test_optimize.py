from tempfile import TemporaryDirectory
from typing import Union
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid, PredefinedSplit

from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_pipelines.conftest import (
    DummyDataset,
    DummyOptimizablePipeline,
    DummyPipeline,
    MutableCustomClass,
    MutableParaPipeline,
    create_dummy_multi_score_func,
    create_dummy_score_func,
    dummy_multi_score_func,
    dummy_single_score_func,
)
from tpcp import clone, make_optimize_safe
from tpcp._optimize import BaseOptimize
from tpcp._utils._score import _optimize_and_score
from tpcp.exceptions import PotentialUserErrorWarning
from tpcp.optimize import DummyOptimize, GridSearch, GridSearchCV, Optimize
from tpcp.validate import Scorer


class TestMetaFunctionalityGridSearch(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = GridSearch

    @pytest.fixture()
    def after_action_instance(self) -> GridSearch:
        gs = GridSearch(DummyOptimizablePipeline(), ParameterGrid({"para_1": [1]}), scoring=dummy_single_score_func)
        gs.optimize(DummyDataset())
        return gs

    def test_empty_init(self):
        pytest.skip()


class TestMetaFunctionalityGridSearchCV(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = GridSearchCV

    @pytest.fixture()
    def after_action_instance(self) -> GridSearchCV:
        gs = GridSearchCV(
            DummyOptimizablePipeline(), ParameterGrid({"para_1": [1]}), cv=2, scoring=dummy_single_score_func
        )
        gs.optimize(DummyDataset())
        return gs

    def test_empty_init(self):
        pytest.skip()


class TestMetaFunctionalityOptimize(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = Optimize

    @pytest.fixture()
    def after_action_instance(self) -> Optimize:
        gs = self.algorithm_class(DummyOptimizablePipeline())
        gs.optimize(DummyDataset())
        return gs

    def test_empty_init(self):
        pytest.skip()


class TestMetaFunctionalityDummyOptimize(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = DummyOptimize

    @pytest.fixture()
    def after_action_instance(self) -> DummyOptimize:
        gs = self.algorithm_class(DummyPipeline())
        gs.optimize(DummyDataset())
        return gs

    def test_empty_init(self):
        pytest.skip()


class TestGridSearchCommon:
    optimizer: Union[GridSearch, GridSearchCV]

    @pytest.fixture(
        autouse=True,
        ids=["GridSearch", "GridSearchCV"],
        params=(
            GridSearch(DummyOptimizablePipeline(), parameter_grid=ParameterGrid({"para_1": [1, 2]})),
            GridSearchCV(DummyOptimizablePipeline(), ParameterGrid({"para_1": [1, 2]}), cv=2),
        ),
    )
    def gridsearch(self, request):
        self.optimizer = request.param.clone()

    @pytest.mark.parametrize("return_optimized", (True, False, "some_str"))
    def test_return_optimized_single(self, return_optimized):
        gs = self.optimizer
        gs.set_params(
            return_optimized=return_optimized,
            parameter_grid=ParameterGrid({"para_1": [1, 2]}),
            scoring=create_dummy_score_func("para_1"),
        )
        warning = None
        if isinstance(return_optimized, str):
            warning = UserWarning

        with pytest.warns(warning) as w:
            gs.optimize(DummyDataset())

        if isinstance(return_optimized, str):
            assert "return_optimize" in str(w[0])
        else:
            assert len(w) == 0

        if return_optimized:  # True or str
            assert gs.best_params_ == {"para_1": 2}
            assert gs.best_index_ == 1
            assert gs.best_score_ == 2
            assert isinstance(gs.optimized_pipeline_, DummyOptimizablePipeline)
            assert gs.optimized_pipeline_.para_1 == gs.best_params_["para_1"]
        else:
            assert not hasattr(gs, "best_params_")
            assert not hasattr(gs, "best_index_")
            assert not hasattr(gs, "best_score_")
            assert not hasattr(gs, "optimized_pipeline_")

    @pytest.mark.parametrize("return_optimized", (False, "score_1", "score_2"))
    def test_return_optimized_multi(self, return_optimized):
        gs = self.optimizer
        gs.set_params(
            return_optimized=return_optimized,
            parameter_grid=ParameterGrid({"para_1": [1, 2], "para_2": [4, 3]}),
            scoring=create_dummy_multi_score_func(("para_1", "para_2")),
        )
        gs.optimize(DummyDataset())

        if return_optimized in ("score_1", "score_2"):
            assert (
                gs.best_params_
                == {"score_1": {"para_1": 2, "para_2": 4}, "score_2": {"para_1": 1, "para_2": 4}}[return_optimized]
            )
            assert gs.best_index_ == {"score_1": 2, "score_2": 0}[return_optimized]
            assert gs.best_score_ == {"score_1": 2, "score_2": 4}[return_optimized]
            assert isinstance(gs.optimized_pipeline_, DummyOptimizablePipeline)
            assert gs.optimized_pipeline_.para_1 == gs.best_params_["para_1"]
        else:
            assert not hasattr(gs, "best_params_")
            assert not hasattr(gs, "best_index_")
            assert not hasattr(gs, "best_score_")
            assert not hasattr(gs, "optimized_pipeline_")

    @pytest.mark.parametrize("return_optimized", (True, "some_str"))
    def test_return_optimized_multi_exception(self, return_optimized):
        gs = self.optimizer
        gs.set_params(
            return_optimized=return_optimized,
            parameter_grid=ParameterGrid({"para_1": [1, 2]}),
            scoring=dummy_multi_score_func,
        )

        with pytest.raises(ValueError):
            gs.optimize(DummyDataset())

    @pytest.mark.parametrize("best_value", (1, 2))
    def test_rank(self, best_value):
        def dummy_best_scorer(best):
            def scoring(pipe, ds):
                if pipe.para_1 == best:
                    return 1
                return 0

            return scoring

        paras = [1, 2]
        gs = self.optimizer
        gs.set_params(
            parameter_grid=ParameterGrid({"para_1": paras}),
            scoring=dummy_best_scorer(best_value),
            return_optimized=True,
        )
        gs.optimize(DummyDataset())

        assert gs.best_score_ == 1
        assert gs.best_index_ == paras.index(best_value)
        assert gs.best_params_ == {"para_1": best_value}
        expected_ranking = [2, 2]
        expected_ranking[paras.index(best_value)] = 1
        if isinstance(self.optimizer, GridSearch):
            results = gs.gs_results_["rank_score"]
        else:
            results = gs.cv_results_["rank_test_score"]
        assert list(results) == expected_ranking


class TestGridSearch:
    def test_single_score(self):
        gs = GridSearch(DummyOptimizablePipeline(), ParameterGrid({"para_1": [1, 2]}), scoring=dummy_single_score_func)
        gs.optimize(DummyDataset())
        results = gs.gs_results_
        results_df = pd.DataFrame(results)

        assert len(results_df) == 2  # Parameters
        assert all(
            s in results for s in ["data_labels", "score", "rank_score", "single_score", "params", "param_para_1"]
        )
        assert all(len(v) == 5 for v in results_df["single_score"])  # 5 data points
        assert all(len(v) == 5 for v in results_df["data_labels"])  # 5 data points
        assert list(results["param_para_1"]) == [1, 2]
        assert list(results["params"]) == [{"para_1": 1}, {"para_1": 2}]
        # In this case the dummy scorer returns the same mean value (2) for each para.
        # Therefore, the ranking should be the same.
        assert list(results["rank_score"]) == [1, 1]
        assert list(results["score"]) == [2, 2]

        assert gs.multimetric_ is False

    def test_multi_score(self):
        gs = GridSearch(
            DummyOptimizablePipeline(),
            ParameterGrid({"para_1": [1, 2]}),
            scoring=dummy_multi_score_func,
            return_optimized=False,
        )
        gs.optimize(DummyDataset())
        results = gs.gs_results_
        results_df = pd.DataFrame(results)

        assert len(results_df) == 2  # Parameters
        assert all(
            s in results
            for s in [
                "data_labels",
                "score_1",
                "rank_score_1",
                "single_score_1",
                "score_2",
                "rank_score_2",
                "single_score_2",
                "params",
                "param_para_1",
            ]
        )
        assert all(len(v) == 5 for c in ["single_score_1", "single_score_2"] for v in results_df[c])  # 5 data points
        assert all(len(v) == 5 for v in results_df["data_labels"])  # 5 data points
        assert list(results["param_para_1"]) == [1, 2]
        assert list(results["params"]) == [{"para_1": 1}, {"para_1": 2}]
        # In this case the dummy scorer returns the same mean value (2) for each para.
        # Therefore, the ranking should be the same.
        assert list(results["rank_score_1"]) == [1, 1]
        assert list(results["rank_score_2"]) == [1, 1]
        assert list(results["score_1"]) == [2, 2]
        assert list(results["score_2"]) == [3, 3]

        assert gs.multimetric_ is True


class TestGridSearchCV:
    def test_single_score(self):
        """Test scoring when only a single performance parameter."""
        # Fixed cv iterator
        cv = PredefinedSplit(test_fold=[0, 0, 1, 1, 1])  # Test Fold 0 has len==2 and 1 has len == 3
        ds = DummyDataset()
        gs = GridSearchCV(
            DummyOptimizablePipeline(), ParameterGrid({"para_1": [1, 2]}), scoring=dummy_single_score_func, cv=cv
        )
        gs.optimize(ds)
        results = gs.cv_results_
        results_df = pd.DataFrame(results)

        assert len(results_df) == 2  # Parameters
        assert set(results_df.columns) == {
            "mean_optimize_time",
            "std_optimize_time",
            "mean_score_time",
            "std_score_time",
            "split0_test_data_labels",
            "split0_train_data_labels",
            "split1_test_data_labels",
            "split1_train_data_labels",
            "param_para_1",
            "params",
            "split0_test_score",
            "split1_test_score",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "split0_test_single_score",
            "split1_test_single_score",
        }

        assert all(len(v) == 2 for v in results_df["split0_test_single_score"])
        assert all(len(v) == 2 for v in results_df["split0_test_data_labels"])
        assert all(len(v) == 3 for v in results_df["split1_test_single_score"])
        assert all(len(v) == 3 for v in results_df["split1_test_data_labels"])
        assert list(results["param_para_1"]) == [1, 2]
        assert list(results["params"]) == [{"para_1": 1}, {"para_1": 2}]
        # fold 1 performance datapoints = [0, 1], fold 2 = [2, 3, 4].
        # The dummy scorer returns average of data points.
        # This is independent of the para.
        # Therefore, rank and score identical.
        folds = cv.split(ds)
        assert all(results["split0_test_score"] == np.mean(next(folds)[1]))
        assert all(results["split1_test_score"] == np.mean(next(folds)[1]))
        assert all(results["mean_test_score"] == np.mean([results["split0_test_score"], results["split1_test_score"]]))
        assert all(results["std_test_score"] == np.std([results["split0_test_score"], results["split1_test_score"]]))
        assert all(results["rank_test_score"] == 1)
        assert gs.multimetric_ is False

    def test_multi_score(self):
        """Test scoring when only a multiple performance parameter."""
        # Fixed cv iterator
        cv = PredefinedSplit(test_fold=[0, 0, 1, 1, 1])  # Test Fold 0 has len==2 and 1 has len == 3
        gs = GridSearchCV(
            DummyOptimizablePipeline(),
            ParameterGrid({"para_1": [1, 2]}),
            scoring=dummy_multi_score_func,
            return_optimized=False,
            cv=cv,
        )
        gs.optimize(DummyDataset())
        results = gs.cv_results_
        results_df = pd.DataFrame(results)

        assert len(results_df) == 2  # Parameters
        assert set(results.keys()) == {
            "mean_optimize_time",
            "std_optimize_time",
            "mean_score_time",
            "std_score_time",
            "split0_test_data_labels",
            "split0_train_data_labels",
            "split1_test_data_labels",
            "split1_train_data_labels",
            "param_para_1",
            "params",
            "split0_test_score_1",
            "split1_test_score_1",
            "mean_test_score_1",
            "std_test_score_1",
            "rank_test_score_1",
            "split0_test_single_score_1",
            "split1_test_single_score_1",
            "split0_test_score_2",
            "split1_test_score_2",
            "mean_test_score_2",
            "std_test_score_2",
            "rank_test_score_2",
            "split0_test_single_score_2",
            "split1_test_single_score_2",
        }

        assert all(
            len(v) == 2 for c in ["split0_test_single_score_2", "split0_test_single_score_1"] for v in results_df[c]
        )
        assert all(
            len(v) == 3 for c in ["split1_test_single_score_2", "split1_test_single_score_1"] for v in results_df[c]
        )
        assert all(len(v) == 2 for v in results_df["split0_test_data_labels"])
        assert all(len(v) == 3 for v in results_df["split1_test_data_labels"])
        assert list(results["param_para_1"]) == [1, 2]
        assert list(results["params"]) == [{"para_1": 1}, {"para_1": 2}]
        # In this case the dummy scorer returns the same mean value (2) for each para.
        # Therefore, the ranking should be the same.
        assert list(results["rank_test_score_1"]) == [1, 1]
        assert list(results["rank_test_score_2"]) == [1, 1]
        folds = list(cv.split(DummyDataset()))
        assert all(results["split0_test_score_1"] == np.mean(folds[0][1]))
        assert all(results["split0_test_score_2"] == np.mean(folds[0][1])) + 1
        assert all(results["split1_test_score_1"] == np.mean(folds[1][1]))
        assert all(results["split1_test_score_2"] == np.mean(folds[1][1])) + 1
        assert all(
            results["mean_test_score_1"] == np.mean([results["split0_test_score_1"], results["split1_test_score_1"]])
        )
        assert all(
            results["std_test_score_1"] == np.std([results["split0_test_score_1"], results["split1_test_score_1"]])
        )
        assert all(
            results["mean_test_score_2"] == np.mean([results["split0_test_score_2"], results["split1_test_score_2"]])
        )
        assert all(
            results["std_test_score_2"] == np.std([results["split0_test_score_2"], results["split1_test_score_2"]])
        )

        assert gs.multimetric_ is True

    def test_return_train_values(self):
        # Fixed cv iterator
        cv = PredefinedSplit(test_fold=[0, 0, 1, 1, 1])  # Test Fold 0 has len==2 and 1 has len == 3
        gs = GridSearchCV(
            DummyOptimizablePipeline(),
            ParameterGrid({"para_1": [1, 2]}),
            scoring=dummy_multi_score_func,
            return_optimized=False,
            return_train_score=True,
            cv=cv,
        )
        gs.optimize(DummyDataset())
        results = gs.cv_results_

        assert set(results.keys()).issuperset(
            {
                "split0_train_data_labels",
                "split1_train_data_labels",
                "split0_train_score_1",
                "split1_train_score_1",
                "mean_train_score_1",
                "std_train_score_1",
                "split0_train_single_score_1",
                "split1_train_single_score_1",
                "split0_train_score_2",
                "split1_train_score_2",
                "mean_train_score_2",
                "std_train_score_2",
                "split0_train_single_score_2",
                "split1_train_single_score_2",
            }
        )

    def test_final_optimized_trained_on_all_data(self):
        optimized_pipe = DummyOptimizablePipeline()
        optimized_pipe.optimized = True
        ds = DummyDataset()

        with patch.object(DummyOptimizablePipeline, "self_optimize", return_value=optimized_pipe) as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipeline.self_optimize = make_optimize_safe(DummyOptimizablePipeline.self_optimize)
            GridSearchCV(
                DummyOptimizablePipeline(),
                ParameterGrid({"para_1": [1, 2, 3]}),
                scoring=dummy_single_score_func,
                cv=2,
                return_optimized=True,
            ).optimize(ds)

        assert mock.call_count == 7  # 3 paras * 2 folds + final optimize
        # Final optimize was called with all the data.
        assert len(mock.call_args_list[-1][0][1]) == 5

    @pytest.mark.parametrize("pure_paras,call_count", ((False, 6 * 2), (True, 2 * 2), (["para_1"], 2 * 2)))
    def test_pure_parameters(self, pure_paras, call_count):
        optimized_pipe = DummyOptimizablePipeline()
        optimized_pipe.optimized = True
        ds = DummyDataset()

        with patch.object(DummyOptimizablePipeline, "self_optimize", return_value=optimized_pipe) as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipeline.self_optimize = make_optimize_safe(DummyOptimizablePipeline.self_optimize)
            GridSearchCV(
                DummyOptimizablePipeline(),
                ParameterGrid({"para_1": [1, 2, 3], "para_2": [0, 1]}),
                scoring=dummy_single_score_func,
                cv=2,
                return_optimized=False,
                pure_parameters=pure_paras,
            ).optimize(ds)

        assert mock.call_count == call_count

    def test_pure_parameters_cache(self):
        """Test that pure parameter cache is deleted after run."""
        # We just run our test twice. If the cache is not delted, the second run should fail.
        self.test_pure_parameters(True, 4)
        self.test_pure_parameters(True, 4)

    def test_pure_parameter_modified_error(self):
        optimized_pipe = DummyOptimizablePipeline()
        optimized_pipe.optimized = True
        # Modify pure para
        optimized_pipe.para_1 = "something"
        ds = DummyDataset()

        with patch.object(DummyOptimizablePipeline, "self_optimize", return_value=optimized_pipe) as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipeline.self_optimize = make_optimize_safe(DummyOptimizablePipeline.self_optimize)
            with pytest.raises(ValueError) as e:
                GridSearchCV(
                    DummyOptimizablePipeline(),
                    ParameterGrid({"para_1": [1, 2, 3], "para_2": [0, 1]}),
                    scoring=dummy_single_score_func,
                    cv=2,
                    pure_parameters=["para_1"],
                    return_optimized=False,
                ).optimize(ds)

        assert "Optimizing the pipeline modified a parameter marked as `pure`." in str(e)

    def test_parameters_set_correctly(self):
        with TemporaryDirectory() as tmp:
            # We run that multiple times to trigger the cache
            for _ in range(2):
                result = _optimize_and_score(
                    Optimize(DummyOptimizablePipeline()),
                    DummyDataset(),
                    Scorer(dummy_single_score_func),
                    np.array([0]),
                    np.array([1]),
                    pure_parameters={"pipeline__para_1": "some_value"},
                    hyperparameters={"pipeline__para_2": "some_other_value"},
                    return_optimizer=True,
                    memory=joblib.Memory(tmp),
                )
                assert result["optimizer"].optimized_pipeline_.para_1 == "some_value"
                assert result["optimizer"].optimized_pipeline_.para_2 == "some_other_value"
                assert result["optimizer"].optimized_pipeline_.optimized == "some_other_value"


class TestOptimize:
    def test_self_optimized_called(self):
        optimized_pipe = DummyOptimizablePipeline()
        optimized_pipe.optimized = True

        ds = DummyDataset()
        kwargs = {"some_kwargs": "some value"}
        with patch.object(DummyOptimizablePipeline, "self_optimize", return_value=optimized_pipe) as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipeline.self_optimize = make_optimize_safe(DummyOptimizablePipeline.self_optimize)
            pipe = DummyOptimizablePipeline()
            # We make the mock a bound method for this test
            mock.__self__ = pipe
            result = Optimize(pipe).optimize(ds, **kwargs)

        mock.assert_called_once()
        mock.assert_called_with(ds, **kwargs)

        assert result.optimized_pipeline_.get_params() == optimized_pipe.get_params()
        # The id must been different, indicating that `optimize` correctly called clone on the output
        assert id(result.optimized_pipeline_) != id(optimized_pipe)

    def test_mutable_inputs(self):
        """Test how mutable inputs are handled.

        Sometimes we have mutableobjects (e.g. dicts or sklearn classifier) as parameters.
        These might be modified during training.
        However, in no case should this modify the original pipeline when using Optimize.
        """
        ds = DummyDataset()
        mutable_instance = MutableCustomClass()
        p = MutableParaPipeline(para_mutable=mutable_instance)
        opt = Optimize(p).optimize(ds)

        assert joblib.hash(mutable_instance) == joblib.hash(p.para_mutable)
        mutable_instance = clone(mutable_instance)
        mutable_instance.test = True
        assert joblib.hash(mutable_instance) == joblib.hash(opt.optimized_pipeline_.para_mutable)
        assert joblib.hash(p.para_mutable) != joblib.hash(opt.optimized_pipeline_.para_mutable)


class TestDummyOptimize:
    def test_optimize_does_not_modify(self):
        pipeline = DummyPipeline()
        optimizer = DummyOptimize(pipeline)

        optimized = optimizer.optimize(dataset=None).optimized_pipeline_
        assert optimized is not pipeline
        assert joblib.hash(pipeline) == joblib.hash(optimized)

    def test_warning(self):
        with pytest.warns(PotentialUserErrorWarning) as w:
            DummyOptimize(DummyOptimizablePipeline()).optimize(dataset=None)

        assert len(w.list) == 1

        with pytest.warns(None) as w:
            DummyOptimize(DummyPipeline()).optimize(dataset=None)

        assert len(w.list) == 0


class TestOptimizeBase:
    optimizer: BaseOptimize

    @pytest.fixture(
        autouse=True,
        params=(
            Optimize(DummyOptimizablePipeline()),
            GridSearch(DummyOptimizablePipeline(), ParameterGrid({"para_1": [1]})),
        ),
    )
    def optimizer_instance(self, request):
        self.optimizer = request.param

    def test_run(self):
        ds = DummyDataset()[0]
        return_val = "return_val"
        self.optimizer.optimized_pipeline_ = DummyOptimizablePipeline()
        with patch.object(DummyOptimizablePipeline, "run", return_value=return_val) as mock_method:
            out = getattr(self.optimizer, "run")(ds)

        assert mock_method.called_with(ds)
        assert return_val == out
