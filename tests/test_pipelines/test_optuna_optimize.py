from typing import Callable, List, Optional, Sequence, Union
from unittest.mock import Mock, patch

import numpy as np
import pytest
from optuna import Study, Trial, create_study
from optuna.samplers import GridSampler, RandomSampler
from optuna.trial import FrozenTrial

from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_pipelines.conftest import (
    DummyDataset,
    DummyOptimizablePipeline,
    DummyPipeline,
    dummy_multi_score_func,
    dummy_single_score_func,
)
from tpcp import make_optimize_safe
from tpcp._dataset import DatasetT
from tpcp._pipeline import OptimizablePipeline, PipelineT
from tpcp.optimize.optuna import CustomOptunaOptimize, OptunaSearch
from tpcp.validate import Scorer


class DummyOptunaOptimizer(CustomOptunaOptimize[PipelineT, DatasetT]):
    def __init__(
        self,
        pipeline: PipelineT,
        create_study: Callable[[], Study],
        scoring: Callable,
        create_search_space: Callable,
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
        return_optimized: bool = True,
        n_jobs: int = 1,
        eval_str_paras: Sequence[str] = (),
        mock_objective=None,
    ) -> None:
        self.scoring = scoring
        self.create_search_space = create_search_space
        self.mock_objective = mock_objective
        super().__init__(
            pipeline,
            create_study,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
            return_optimized=return_optimized,
            n_jobs=n_jobs,
            eval_str_paras=eval_str_paras,
        )

    def create_objective(self) -> Callable[[Trial, PipelineT, DatasetT], Union[float, Sequence[float]]]:
        def objective(trial: Trial, pipeline: PipelineT, dataset: DatasetT) -> float:
            self.create_search_space(trial)
            pipeline = pipeline.set_params(**self.sanitize_params(trial.params))

            scorer = Scorer(self.scoring)

            average_score, single_scores = scorer(pipeline, dataset)
            trial.set_user_attr("single_scores", single_scores)
            return average_score

        if self.mock_objective:
            return self.mock_objective
        return objective


def dummy_search_space(trial: Trial):
    trial.suggest_categorical("para_1", [1])


def _get_study():
    # We define it globally so that we can pickle it
    return create_study(sampler=RandomSampler(42))


class TestMetaFunctionalityOptuna(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = DummyOptunaOptimizer
    _ignored_names = ("create_search_space", "scoring", "mock_objective")

    @pytest.fixture()
    def after_action_instance(self) -> DummyOptunaOptimizer:

        gs = DummyOptunaOptimizer(
            DummyOptimizablePipeline(),
            _get_study,
            scoring=dummy_single_score_func,
            create_search_space=dummy_search_space,
            n_trials=1,
        )
        gs.optimize(DummyDataset())
        return gs

    def test_empty_init(self):
        pytest.skip()


class TestCustomOptunaOptimize:
    def test_invalid_study_stop(self):
        with pytest.raises(ValueError):
            DummyOptunaOptimizer(
                DummyOptimizablePipeline(),
                _get_study,
                scoring=dummy_single_score_func,
                create_search_space=dummy_search_space,
                n_trials=None,  # These should not be both None
                timeout=None,
            ).optimize(DummyDataset())

    def test_objective_called(self):
        mock_objective = Mock(return_value=3)

        n_trials = 5
        dataset = DummyDataset()
        pipe = DummyOptimizablePipeline()

        DummyOptunaOptimizer(
            pipe,
            _get_study,
            scoring=dummy_single_score_func,
            create_search_space=dummy_search_space,
            n_trials=n_trials,
            timeout=None,
            mock_objective=mock_objective,
        ).optimize(dataset)

        assert mock_objective.call_count == n_trials
        assert mock_objective.call_args[0][2] is dataset
        # Should be a clone of the pipe, but not the pipe
        assert mock_objective.call_args[0][1].get_params() == pipe.get_params()
        assert mock_objective.call_args[0][1] is not pipe

        assert isinstance(mock_objective.call_args[0][0], Trial)

    @pytest.mark.parametrize("return_optimize", (True, False))
    def test_return_optimized(self, return_optimize):
        mock_objective = Mock(return_value=3)

        opti = DummyOptunaOptimizer(
            DummyOptimizablePipeline(),
            _get_study,
            scoring=dummy_single_score_func,
            create_search_space=dummy_search_space,
            n_trials=1,
            timeout=None,
            mock_objective=mock_objective,
            return_optimized=return_optimize,
        ).optimize(DummyDataset())

        assert hasattr(opti, "optimized_pipeline_") == return_optimize

    def test_return_optimized_calls_optimize(self):
        mock_objective = Mock(return_value=3)
        optimized_pipe = DummyOptimizablePipeline()
        dataset = DummyDataset()

        with patch.object(DummyOptimizablePipeline, "self_optimize", return_value=optimized_pipe) as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipeline.self_optimize = make_optimize_safe(DummyOptimizablePipeline.self_optimize)

            DummyOptunaOptimizer(
                DummyOptimizablePipeline(),
                _get_study,
                scoring=dummy_single_score_func,
                create_search_space=dummy_search_space,
                n_trials=1,
                timeout=None,
                mock_objective=mock_objective,
                return_optimized=True,
            ).optimize(dataset)

        assert mock.call_args[0][1] is dataset

    @pytest.mark.parametrize("pipe", (DummyOptimizablePipeline(), DummyPipeline()))
    @pytest.mark.parametrize("n_jobs", (1, 2))
    def test_correct_paras_selected(self, pipe, n_jobs, tmp_path):
        # Should select 1, as it has the highest score
        scores = {0: 1, 1: 2, 2: 0}

        def create_search_space(trial):
            trial.suggest_categorical("para_1", list(scores.keys()))

        def scoring(pipe, _):
            return scores[pipe.para_1]

        storage = f"sqlite:///{tmp_path}/test.db" if n_jobs > 1 else None

        opti = DummyOptunaOptimizer(
            pipe,
            lambda: create_study(
                sampler=GridSampler({"para_1": list(scores.keys())}), direction="maximize", storage=storage
            ),
            scoring=scoring,
            create_search_space=create_search_space,
            # In case of the multiprocessing backend, we need to set n_trials to a value > len(scores), as we can not
            # guarantee that the different trails will not choose the same parameters due to timing issues.
            # See docu for `GridSampler` in optuna.
            n_trials=6,
            n_jobs=n_jobs,
            timeout=None,
            return_optimized=True,
        ).optimize(DummyDataset())

        assert opti.best_params_ == {"para_1": 1}
        assert opti.best_score_ == 2
        assert opti.optimized_pipeline_.para_1 == 1

        r = opti.search_results_

        assert set(r["param_para_1"]) == set(scores.keys())
        assert {tuple(p.items()) for p in r["params"]} == {(("para_1", k),) for k in scores}

        if isinstance(pipe, OptimizablePipeline):
            # That is expected when self_optimize was called correctly.
            assert opti.optimized_pipeline_.optimized == opti.optimized_pipeline_.para_2

    def test_literal_string_eval(self):
        # Note, this does not actually test, if the string is correctly evaluated before being passed to the pipeline.
        def search_space(trial):
            # This para should remain a string
            trial.suggest_categorical("para_1", ["('a', 'b')"])
            # This one will be evaluated
            trial.suggest_categorical("para_2", ["('a', 'b')"])

        optuna_search = DummyOptunaOptimizer(
            DummyOptimizablePipeline(),
            _get_study,
            create_search_space=search_space,
            scoring=dummy_single_score_func,
            n_trials=1,
            eval_str_paras=["para_2"],
        )

        optuna_search.optimize(DummyDataset())

        assert optuna_search.optimized_pipeline_.get_params()["para_1"] == "('a', 'b')"
        assert optuna_search.optimized_pipeline_.get_params()["para_2"] == ("a", "b")

        assert optuna_search.best_params_["para_1"] == "('a', 'b')"
        assert optuna_search.best_params_["para_2"] == ("a", "b")


class TestMetaFunctionalityOptunaSearch(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = OptunaSearch

    @pytest.fixture()
    def after_action_instance(self) -> OptunaSearch:

        gs = OptunaSearch(
            DummyOptimizablePipeline(),
            _get_study,
            dummy_search_space,
            scoring=dummy_single_score_func,
            n_trials=1,
        )
        gs.optimize(DummyDataset())
        return gs

    def test_empty_init(self):
        pytest.skip()


class TestOptunaSearch:
    def test_single_score(self):
        optuna_search = OptunaSearch(
            DummyOptimizablePipeline(),
            _get_study,
            dummy_search_space,
            scoring=dummy_single_score_func,
            n_trials=1,
        )

        optuna_search.optimize(DummyDataset())
        results = optuna_search.search_results_

        assert "param_para_1" in results
        assert "score" in results
        assert results["score"][0] == np.mean(range(5))
        assert "single_score" in results
        assert results["single_score"][0] == list(range(5))
        assert "data_labels" in results

        assert not any(c.startswith("user_attrs_") for c in results)

        assert optuna_search.optimized_pipeline_.get_params()["para_1"] == optuna_search.best_params_["para_1"]

    @pytest.mark.parametrize("score_name", ("score_2", "score_1"))
    def test_search_result_columns_multi_score(self, score_name):
        optuna_search = OptunaSearch(
            DummyOptimizablePipeline(),
            _get_study,
            dummy_search_space,
            scoring=dummy_multi_score_func,
            score_name=score_name,
            n_trials=1,
        )

        optuna_search.optimize(DummyDataset())
        results = optuna_search.search_results_

        assert "param_para_1" in results
        assert "score_1" in results
        assert results["score_1"][0] == np.mean(range(5))
        assert "score_2" in results
        assert results["score_2"][0] == np.mean(range(5)) + 1
        assert "single_score_1" in results
        assert results["single_score_1"][0] == list(range(5))
        assert "single_score_2" in results
        assert results["single_score_2"][0] == list(range(1, 6))
        assert "data_labels" in results

        assert not any(c.startswith("user_attrs_") for c in results)

        assert optuna_search.optimized_pipeline_.get_params()["para_1"] == optuna_search.best_params_["para_1"]

        # The dummy scorer returns the datapoint id for score 1 and the datapoint id + 1 for score 2.
        assert optuna_search.best_score_ == np.mean(range(5)) + int(score_name == "score_2")

    @pytest.mark.parametrize("score_name", ("score_3", False, None))
    def test_multi_metric_wrong_score_name(self, score_name):
        with pytest.raises(ValueError):
            OptunaSearch(
                DummyOptimizablePipeline(),
                _get_study,
                dummy_search_space,
                scoring=dummy_multi_score_func,
                score_name=score_name,
                n_trials=1,
            ).optimize(DummyDataset())

    def test_warns_single_score_score_name(self):
        with pytest.warns(UserWarning):
            OptunaSearch(
                DummyOptimizablePipeline(),
                _get_study,
                dummy_search_space,
                scoring=dummy_single_score_func,
                score_name="score_1",
                n_trials=1,
            ).optimize(DummyDataset())

    def test_literal_string_eval(self):
        # Note, this does not actually test, if the string is correctly evaluated before being passed to the pipeline.
        def search_space(trial):
            # This para should remain a string
            trial.suggest_categorical("para_1", ["('a', 'b')"])
            # This one will be evaluated
            trial.suggest_categorical("para_2", ["('a', 'b')"])

        optuna_search = OptunaSearch(
            DummyOptimizablePipeline(),
            _get_study,
            search_space,
            scoring=dummy_single_score_func,
            n_trials=1,
            eval_str_paras=["para_2"],
        )

        optuna_search.optimize(DummyDataset())

        assert optuna_search.optimized_pipeline_.get_params()["para_1"] == "('a', 'b')"
        assert optuna_search.optimized_pipeline_.get_params()["para_2"] == ("a", "b")

        assert optuna_search.best_params_["para_1"] == "('a', 'b')"
        assert optuna_search.best_params_["para_2"] == ("a", "b")
