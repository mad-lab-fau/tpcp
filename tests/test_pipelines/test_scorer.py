from typing import Sequence
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
from tpcp.exceptions import ValidationError
from tpcp.validate import Scorer
from tpcp.validate._scorer import NoAgg, _passthrough_scoring, _validate_scorer


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
        scorer(pipeline=pipe, dataset=DummyDataset())

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
        agg, single = scorer(pipe, data)
        assert len(single) == len(data)
        # Our Dummy scorer, returns the groupname of the dataset
        assert all(np.array(single) == data.groups)
        assert agg == np.mean(data.groups)

    def test_score_return_val_multi_score(self):
        scorer = Scorer(dummy_multi_score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data)
        assert isinstance(single, dict)
        for k, v in single.items():
            assert len(v) == len(data)
            # Our Dummy scorer, returns the groupname of the dataset
            if k == "score_2":
                assert all(np.array(v) == np.array(data.groups) + 1)
            else:
                assert all(np.array(v) == data.groups)
        assert isinstance(agg, dict)
        for k, v in agg.items():
            if k == "score_2":
                assert v == np.mean(data.groups) + 1
            else:
                assert v == np.mean(data.groups)

    def test_score_return_val_multi_score_no_agg(self):
        def multi_score_func(pipeline, data_point):
            return {"score_1": data_point.groups[0], "no_agg_score": NoAgg(str(data_point.groups))}

        scorer = Scorer(multi_score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data)
        assert isinstance(single, dict)
        for k, v in single.items():
            assert len(v) == len(data)
            # Our Dummy scorer, returns the groupname of the dataset as string in the no-agg case
            if k == "no_agg_score":
                assert all(np.array(v) == [str(d.groups) for d in data])
            else:
                assert all(np.array(v) == data.groups)
        assert isinstance(agg, dict)
        assert "score_1" in agg
        assert "no_agg_score" not in agg
        assert agg["score_1"] == np.mean(data.groups)

    @pytest.mark.parametrize(
        "bad_scorer", (lambda x, y: "test", lambda x, y: {"val": "test"}, lambda x, y: NoAgg(None))
    )
    def test_bad_scorer(self, bad_scorer):
        """Check that we catch cases where the scoring func returns invalid values independent of the error_score val"""
        scorer = Scorer(bad_scorer)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        with pytest.raises(ValidationError) as e:
            scorer(pipe, data)
        assert "The scoring function must have one" in str(e.value)

    def test_kwargs_passed(self):
        kwargs = {"a": 3, "b": "test"}
        scorer = Scorer(lambda x: x, **kwargs)
        assert kwargs == scorer.kwargs

    def test_callback_called(self):
        mock_score_func = Mock(return_value=1)
        mock_callback = Mock()
        scorer = Scorer(mock_score_func, single_score_callback=mock_callback)

        pipe = DummyOptimizablePipeline()
        scorer(pipeline=pipe, dataset=DummyDataset())

        assert mock_callback.call_count == len(DummyDataset())
        for call, i in zip(mock_callback.call_args_list, range(len(DummyDataset()))):
            assert call[0] == tuple()
            kwargs = call[1]
            assert kwargs.pop("scorer") == scorer
            assert kwargs.pop("scores") == (1,) * (i + 1)
            assert kwargs.pop("pipeline") == pipe
            assert kwargs.pop("dataset").groups == DummyDataset().groups
            assert kwargs.pop("step") == i
            assert kwargs == {}

    def test_documented_callback_signature_valid(self):
        def callback(*, step: int, scores: Sequence[float], **_):
            assert isinstance(step, int)
            assert len(scores) == step + 1

        mock_score_func = Mock(return_value=1)
        scorer = Scorer(mock_score_func, single_score_callback=callback)

        pipe = DummyOptimizablePipeline()
        scorer(pipeline=pipe, dataset=DummyDataset())

    def test_no_agg_scoring(self):
        pass


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
