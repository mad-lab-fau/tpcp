from typing import Callable, List, Optional, Sequence, Union

import pytest
from optuna import Study, Trial, create_study
from optuna.samplers import GridSampler, RandomSampler
from optuna.structs import FrozenTrial

from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_pipelines.conftest import DummyDataset, DummyOptimizablePipeline, dummy_single_score_func
from tpcp._dataset import Dataset_
from tpcp._pipeline import Pipeline_
from tpcp.optimize.optuna import CustomOptunaOptimize
from tpcp.validate import Scorer


class DummyOptunaOptimizer(CustomOptunaOptimize[Pipeline_, Dataset_]):
    def __init__(
        self,
        pipeline: Pipeline_,
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
    ) -> None:
        self.scoring = scoring
        self.create_search_space = create_search_space
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

    def create_objective(self) -> Callable[[Trial, Pipeline_, Dataset_], Union[float, Sequence[float]]]:
        def objective(trial: Trial, pipeline: Pipeline_, dataset: Dataset_) -> float:
            self.create_search_space(trial)
            pipeline = pipeline.set_params(**trial.params)

            scorer = Scorer(self.scoring)

            average_score, single_scores = scorer(pipeline, dataset, error_score="raise")

            return average_score

        return objective


def dummy_search_space(trial: Trial):
    trial.suggest_categorical("para_1", [1])


class TestMetaFunctionalityGridSearch(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = DummyOptunaOptimizer
    _ignored_names = ("create_search_space", "scoring")

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
