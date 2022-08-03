from typing import Callable, List, Optional, Sequence, Union
from unittest.mock import Mock, patch

import pytest
from optuna import Study, Trial, create_study
from optuna.samplers import GridSampler, RandomSampler
from optuna.structs import FrozenTrial

from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_pipelines.conftest import DummyDataset, DummyOptimizablePipeline, DummyPipeline, dummy_single_score_func
from tpcp import make_optimize_safe
from tpcp._dataset import DatasetT
from tpcp._pipeline import OptimizablePipeline, PipelineT
from tpcp.optimize.optuna import CustomOptunaOptimize
from tpcp.validate import Scorer


class DummyOptunaOptimizer(CustomOptunaOptimize[PipelineT, DatasetT]):
    def __init__(
        self,
        pipeline: PipelineT,
        study: Study,
        scoring: Callable,
        create_search_space: Callable,
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
        return_optimized: bool = True,
        mock_objective=None,
    ) -> None:
        self.scoring = scoring
        self.create_search_space = create_search_space
        self.mock_objective = mock_objective
        super().__init__(
            pipeline,
            study,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
            return_optimized=return_optimized,
        )

    def create_objective(self) -> Callable[[Trial, PipelineT, DatasetT], Union[float, Sequence[float]]]:
        def objective(trial: Trial, pipeline: PipelineT, dataset: DatasetT) -> float:
            self.create_search_space(trial)
            pipeline = pipeline.set_params(**trial.params)

            scorer = Scorer(self.scoring)

            average_score, single_scores = scorer(pipeline, dataset)
            trial.set_user_attr("single_scores", single_scores)
            return average_score

        if self.mock_objective:
            return self.mock_objective
        return objective


def dummy_search_space(trial: Trial):
    trial.suggest_categorical("para_1", [1])


class TestMetaFunctionalityGridSearch(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = DummyOptunaOptimizer
    _ignored_names = ("create_search_space", "scoring", "mock_objective")

    @pytest.fixture()
    def after_action_instance(self) -> DummyOptunaOptimizer:

        study = create_study(sampler=RandomSampler(42))

        gs = DummyOptunaOptimizer(
            DummyOptimizablePipeline(),
            study,
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
                study=create_study(sampler=RandomSampler(42)),
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
            study=create_study(sampler=RandomSampler(42)),
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
            study=create_study(sampler=RandomSampler(42)),
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
                study=create_study(sampler=RandomSampler(42)),
                scoring=dummy_single_score_func,
                create_search_space=dummy_search_space,
                n_trials=1,
                timeout=None,
                mock_objective=mock_objective,
                return_optimized=True,
            ).optimize(dataset)

        assert mock.call_args[0][1] is dataset

    @pytest.mark.parametrize("pipe", (DummyOptimizablePipeline(), DummyPipeline()))
    def test_correct_paras_selected(self, pipe):
        # Should select 1, as it has the highest score
        scores = {0: 1, 1: 2, 2: 0}

        def create_search_space(trial):
            trial.suggest_categorical("para_1", list(scores.keys()))

        def scoring(pipe, _):
            return scores[pipe.para_1]

        opti = DummyOptunaOptimizer(
            pipe,
            study=create_study(sampler=GridSampler({"para_1": list(scores.keys())}), direction="maximize"),
            scoring=scoring,
            create_search_space=create_search_space,
            n_trials=3,
            timeout=None,
            return_optimized=True,
        ).optimize(DummyDataset())

        assert opti.best_params_ == {"para_1": 1}
        assert opti.best_score_ == 2
        assert opti.optimized_pipeline_.para_1 == 1

        r = opti.search_results_

        assert set(r["param_para_1"]) == set(scores.keys())
        assert set(tuple(p.items()) for p in r["params"]) == set((("para_1", k),) for k in scores.keys())

        if isinstance(pipe, OptimizablePipeline):
            # That is expected when self_optimize was called correctly.
            assert opti.optimized_pipeline_.optimized == opti.optimized_pipeline_.para_2
