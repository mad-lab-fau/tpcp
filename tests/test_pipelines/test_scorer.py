from itertools import cycle
from typing import Dict, Sequence
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest

from tests.test_pipelines.conftest import (
    DummyDataset,
    DummyOptimizablePipeline,
    dummy_multi_score_func,
    dummy_single_score_func,
)
from tpcp.exceptions import ScorerFailed, ValidationError
from tpcp.validate import Scorer
from tpcp.validate._scorer import Aggregator, NoAgg, _passthrough_scoring, _validate_scorer


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

    @pytest.mark.parametrize("bad_scorer", (lambda x, y: "test", lambda x, y: {"val": "test"}))
    def test_bad_scorer(self, bad_scorer):
        """Check that we catch cases where the scoring func returns invalid values independent of the error_score val"""
        scorer = Scorer(bad_scorer)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        with pytest.raises(ValidationError) as e:
            scorer(pipe, data)
        assert "MeanAggregator can only be used with float values" in str(e.value)

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

    @pytest.mark.parametrize("reversed", (True, False))
    def test_non_homogeneus_return(self, reversed):
        non_homogeneous_return = [3, {"val": 3}]
        if reversed:
            non_homogeneous_return.reverse()

        return_vals = cycle(non_homogeneous_return)

        with pytest.raises(ValidationError) as e:
            scorer = Scorer(lambda x, y: next(return_vals))
            pipe = DummyOptimizablePipeline()
            data = DummyDataset()
            _ = scorer(pipe, data)

        assert "The returned score values are not homogeneous." in str(e)

    def test_inconsistent_keys(self):
        return_vals = cycle([{"val1": 3, "val2": 3}, {"val1": 3, "val3": 3}])

        with pytest.raises(ValidationError) as e:
            scorer = Scorer(lambda x, y: next(return_vals))
            pipe = DummyOptimizablePipeline()
            data = DummyDataset()
            _ = scorer(pipe, data)

        assert "same keys for each datapoint." in str(e)
        assert "missing the key 'val2'" in str(e)
        assert "are: ('val1', 'val2')" in str(e)

    def test_score_func_raises_exception(self):
        def score_func(x, y):
            raise Exception("test")

        scorer = Scorer(score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        with pytest.raises(ScorerFailed) as e:
            scorer(pipe, data)

        assert 'Exception("test")' in str(e)


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


class TestCustomAggregator:
    class MultiAgg(Aggregator):
        @classmethod
        def aggregate(cls, /, values: Sequence[float], **_) -> Dict[str, float]:
            return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    @pytest.mark.parametrize(
        "score_type, scorefunc", [("single", lambda x, y: 3), ("multi", lambda x, y: {"value": 3})]
    )
    def test_multi_agg(self, score_type, scorefunc):
        scorer = Scorer(scorefunc, default_aggregator=self.MultiAgg)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        agg, single = scorer(pipe, data)
        assert isinstance(agg, dict)
        if score_type == "multi":
            assert "value__mean" in agg
            assert "value__std" in agg
        else:
            assert "mean" in agg
            assert "std" in agg

    class DummyAgg(Aggregator):
        @classmethod
        def aggregate(cls, /, values: Sequence[float], **_) -> float:
            return 0

    @pytest.mark.parametrize(
        "scorer_return", (1, {"val": 1}, {"val": 1, "val2": 2}, {"val": 1, "val2": 2, "val3": NoAgg(None)})
    )
    @mock.patch("tests.test_pipelines.test_scorer.TestCustomAggregator.DummyAgg.aggregate", return_value=1)
    def test_default_agg_method(self, mock_aggregate, scorer_return):
        scorer = Scorer(lambda x, y: scorer_return, default_aggregator=self.DummyAgg)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        _ = scorer(pipe, data)

        expected_called = (
            len([s for s in scorer_return.values() if not isinstance(s, NoAgg)])
            if isinstance(scorer_return, dict)
            else 1
        )

        assert mock_aggregate.call_count == expected_called

    @pytest.mark.parametrize(
        "scorer_return,expected_val",
        (
            (1, 0),
            ({"val": 1}, 0),
            ({"val": 1, "val2": 2}, 0),
            ({"val": 1, "val2": 2, "val3": NoAgg(None)}, 0),
            (DummyAgg(1), 1),
            ({"val": DummyAgg(1)}, 1),
            ({"val": DummyAgg(1), "val2": DummyAgg(2)}, 2),
            ({"val": DummyAgg(1), "val2": 2, "val3": NoAgg(3)}, 1),
        ),
    )
    @mock.patch("tests.test_pipelines.test_scorer.TestCustomAggregator.DummyAgg.aggregate", return_value=1)
    def test_selective_agg(self, mock_aggregate, scorer_return, expected_val):
        scorer = Scorer(lambda x, y: scorer_return)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        _ = scorer(pipe, data)
        assert mock_aggregate.call_count == expected_val

    def test_inconsistent_return_single(self):
        return_vals = cycle([self.DummyAgg(1), 3])

        with pytest.raises(ValidationError) as e:
            scorer = Scorer(lambda x, y: next(return_vals))
            pipe = DummyOptimizablePipeline()
            data = DummyDataset()
            _ = scorer(pipe, data)

        assert "The score values are not consistent." in str(e)

    def test_inconsistent_return_multi(self):
        return_vals = cycle(
            [{"val1": self.DummyAgg(1), "val2": 3}, {"val1": self.DummyAgg(1), "val2": self.DummyAgg(5)}]
        )

        with pytest.raises(ValidationError) as e:
            scorer = Scorer(lambda x, y: next(return_vals))
            pipe = DummyOptimizablePipeline()
            data = DummyDataset()
            _ = scorer(pipe, data)

        assert "val2" in str(e)

    def test_no_agg_single_raises(self):
        with pytest.raises(ValidationError) as e:
            scorer = Scorer(lambda x, y: NoAgg(None))
            pipe = DummyOptimizablePipeline()
            data = DummyDataset()
            _ = scorer(pipe, data)

        assert "Scorer returned a NoAgg object. " in str(e)

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

    class InvalidMultiAgg(Aggregator):
        @classmethod
        def aggregate(cls, **_):
            return {"val": "invalid"}

    class InvalidSingleAgg(Aggregator):
        @classmethod
        def aggregate(cls, **_):
            return "invalid"

    @pytest.mark.parametrize("score_func", (lambda x, y: 1, lambda x, y: {"val": 1}))
    @pytest.mark.parametrize("aggregator", [InvalidMultiAgg, InvalidSingleAgg])
    def test_invalid_aggregator_return_type(self, aggregator, score_func):
        scorer = Scorer(score_func, default_aggregator=aggregator)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        with pytest.raises(ValidationError) as e:
            _ = scorer(pipe, data)
        assert "number" in str(e)

    class TestAgg1(Aggregator):
        @classmethod
        def aggregate(cls, **_):
            return 1

    class TestAgg2(Aggregator):
        @classmethod
        def aggregate(cls, **_):
            return 2

    def test_all_aggregators_called_correctly(self):
        def score_func(p, d):
            return {"agg1": self.TestAgg1(None), "agg2": self.TestAgg2(None), "default_agg": 3, "no_agg": NoAgg(4)}

        scorer = Scorer(score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        agg_score, _ = scorer(pipe, data)

        assert agg_score["agg1"] == 1
        assert agg_score["agg2"] == 2
        assert agg_score["default_agg"] == 3
        assert "no_agg" not in agg_score

    class DatapointAgg(Aggregator):
        @classmethod
        def aggregate(cls, /, values, datapoints):
            return 1

    @pytest.mark.parametrize("score_func_type", ("single", "multi"))
    @mock.patch("tests.test_pipelines.test_scorer.TestCustomAggregator.DatapointAgg.aggregate", return_value=1)
    def test_datapoints_forwarded_to_agg(self, mock_method, score_func_type):
        if score_func_type == "single":
            score_func = lambda x, y: self.DatapointAgg(y)
        else:
            score_func = lambda x, y: {"val": self.DatapointAgg(y)}
        scorer = Scorer(score_func)
        pipe = DummyOptimizablePipeline()
        data = DummyDataset()
        _ = scorer(pipe, data)
        assert mock_method.called_with(values=[d for d in data], datapoints=[d for d in data])
