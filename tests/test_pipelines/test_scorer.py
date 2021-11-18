from unittest.mock import Mock

import numpy as np
import pytest

from tests.test_pipelines.conftest import (
    DummyDataset,
    DummyOptimizablePipeline,
    dummy_error_score_func,
    dummy_error_score_func_multi,
    dummy_multi_score_func,
    dummy_single_score_func,
)
from tpcp.validation import Scorer
from tpcp.validation._scorer import _passthrough_scoring, _validate_scorer


class TestScorerCalls:
    scorer: Scorer

    @pytest.fixture(autouse=True)
    def create_scorer(self):
        self.scorer = Scorer(lambda x: x)

    def test_score_func_called(self):
        """Test that the score func is called once per dataset"""
        mock_score_func = Mock(return_value=1)
        scorer = Scorer(mock_score_func)
        pipe = DummyOptimizablePipeline()
        scorer(pipeline=pipe, data=DummyDataset(), error_score=np.nan)

        assert mock_score_func.call_count == len(DummyDataset())
        for call, d in zip(mock_score_func.call_args_list, DummyDataset()):
            assert call[0][1].groups == d.groups
            assert isinstance(call[0][0], DummyOptimizablePipeline)
            # Check that the pipeline was cloned before calling
            assert id(pipe) != id(call[0][0])


class TestScorer:
    def test_score_return_val_single_score(self):
        scorer = Scorer(dummy_single_score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data, np.nan)
        assert len(single) == len(data)
        # Our Dummy scorer, returns the groupname of the dataset
        assert all(single == data.groups)
        assert agg == np.mean(data.groups)

    def test_score_return_val_multi_score(self):
        scorer = Scorer(dummy_multi_score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data, np.nan)
        assert isinstance(single, dict)
        for k, v in single.items():
            assert len(v) == len(data)
            # Our Dummy scorer, returns the groupname of the dataset
            if k == "score_2":
                assert all(v == np.array(data.groups) + 1)
            else:
                assert all(v == data.groups)
        assert isinstance(agg, dict)
        for k, v in agg.items():
            if k == "score_2":
                assert v == np.mean(data.groups) + 1
            else:
                assert v == np.mean(data.groups)

    @pytest.mark.parametrize("err_val", (np.nan, 1))
    def test_scoring_return_err_val(self, err_val):
        scorer = Scorer(dummy_error_score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        with pytest.warns(UserWarning) as ws:
            agg, single = scorer(pipe, data, err_val)

        assert len(ws) == 3
        for w, n in zip(ws, [0, 2, 4]):
            assert str(n) in str(w)
            assert str(err_val) in str(w)

        expected = np.array([err_val, 1, err_val, 3, err_val])

        assert len(single) == len(data)
        nan_vals = np.isnan(single)
        assert all(nan_vals == np.isnan(expected))
        assert all(single[~nan_vals] == expected[~nan_vals])

        # agg should become nan if a single value is nan
        if sum(nan_vals) > 0:
            assert np.isnan(agg)
        else:
            assert agg == np.mean(expected)

    @pytest.mark.parametrize("err_val", (np.nan, 1))
    @pytest.mark.filterwarnings("ignore::tpcp.exceptions.ScorerFailed")
    def test_scoring_return_err_val_multi(self, err_val):
        scorer = Scorer(dummy_error_score_func_multi)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data, err_val)

        expected = np.array([err_val, 1, err_val, 3, err_val])

        for v in single.values():
            assert len(v) == len(data)
            nan_vals = np.isnan(v)
            assert all(nan_vals == np.isnan(expected))
            assert all(v[~nan_vals] == expected[~nan_vals])

        for v in agg.values():
            # agg should become nan if a single value is nan
            if sum(nan_vals) > 0:
                assert np.isnan(v)
            else:
                assert v == np.mean(expected)

    def test_err_val_raises(self):
        scorer = Scorer(dummy_error_score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        with pytest.raises(ValueError) as e:
            scorer(pipe, data, "raise")

        assert str(e.value) == "Dummy Error for 0"

    @pytest.mark.parametrize("error_score", ("raise", 0))
    @pytest.mark.parametrize("bad_scorer", (lambda x, y: "test", lambda x, y: {"val": "test"}))
    def test_bad_scorer(self, error_score, bad_scorer):
        """Check that we catch cases where the scoring func returns invalid values independent of the error_score val"""
        scorer = Scorer(bad_scorer)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        with pytest.raises(ValueError) as e:
            scorer(pipe, data, error_score)
        assert "The scoring function must return" in str(e.value)


def _dummy_func(x):
    return x


def _dummy_func_2(x):
    return x


class TestScorerUtils:
    @pytest.mark.parametrize(
        "scoring, expected",
        (
            (None, Scorer(_passthrough_scoring)),
            (_dummy_func, Scorer(_dummy_func)),
            (Scorer(_dummy_func_2), Scorer(_dummy_func_2)),
        ),
    )
    def test_validate_scorer(self, scoring, expected):
        pipe = DummyOptimizablePipeline()
        pipe.score = lambda x: x
        out = _validate_scorer(scoring, pipe)
        assert isinstance(out, type(expected))
        assert out._score_func == expected._score_func

    def test_score_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _validate_scorer(None, DummyOptimizablePipeline())

    def test_invalid_input(self):
        pipe = DummyOptimizablePipeline()
        pipe.score = lambda x: x
        with pytest.raises(ValueError) as e:
            _validate_scorer("something invalid", pipe)

        assert "A valid scorer must either be a instance of" in str(e)
