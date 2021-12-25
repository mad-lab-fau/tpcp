import pandas as pd
from optuna import Trial

from tests.test_pipelines.conftest import DummyOptimizablePipeline, dummy_single_score_func, DummyDataset
from tpcp.optimize._optuna_optimize import OptunaSearch


class TestGridSearch:
    def test_single_score(self):
        def search_space(trial: Trial):
            return {"para_1": trial.suggest_categorical("para_1", [1, 2])}

        gs = OptunaSearch(
            DummyOptimizablePipeline(), search_space, scoring=dummy_single_score_func, n_trials=5, n_jobs=2
        )
        gs.optimize(DummyDataset())
        results = gs.gs_results_
        results_df = pd.DataFrame(results)

        # assert len(results_df) == 2  # Parameters
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
