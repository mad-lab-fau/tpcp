"""Test basic pipeline functionality."""
from unittest.mock import patch

import pytest

from tests.test_pipelines.conftest import DummyDataset, DummyOptimizablePipeline
from tpcp import Dataset, Pipeline, cf
from tpcp._parameter import para


class PipelineInputModify(Pipeline):
    def __init__(self, test="a value"):
        self.test = test

    def run(self, datapoint: Dataset):
        self.test = "another value"
        return self


class PipelineInputModifyNested(Pipeline):
    def __init__(self, pipe=cf(PipelineInputModify())):
        self.pipe = pipe

    def run(self, datapoint: Dataset):
        self.pipe.run(datapoint)
        return self


class PipelineNoOutput(Pipeline):
    def __init__(self, test="a value"):
        self.test = test

    def run(self, datapoint: Dataset):
        self.not_a_output_paras = "something"
        return self


# TODO: Test with algorithms and not just pipeline
# TODO: Test double apply
class TestSafeRun:
    @pytest.mark.parametrize("pipe", (PipelineInputModify, PipelineInputModifyNested))
    def test_modify_input_paras_simple(self, pipe):
        with pytest.raises(ValueError) as e:
            pipe().safe_run(DummyDataset()[0])

        assert f"Running `safe_run` of {pipe.__name__} did modify the parameters of the algorithm" in str(e)

    def test_no_self_return(self):
        pipe = DummyOptimizablePipeline()
        pipe.run = lambda d: "some Value"
        with pytest.raises(ValueError) as e:
            pipe.safe_run(DummyDataset()[0])

        assert "The `safe_run` method of DummyOptimizablePipeline must return `self`" in str(e)

    def test_no_output(self):
        with pytest.raises(ValueError) as e:
            PipelineNoOutput().safe_run(DummyDataset()[0])

        assert "Running the `safe_run` method of PipelineNoOutput " in str(e)

    def test_output(self):
        pipe = DummyOptimizablePipeline()
        pipe.result_ = "some result"
        ds = DummyDataset()[0]
        with patch.object(DummyOptimizablePipeline, "run", return_value=pipe) as mock:
            result = DummyOptimizablePipeline().safe_run(ds)

        mock.assert_called_with(ds)
        assert id(result) == id(pipe)
