from unittest.mock import patch

import pytest

from tests.test_pipelines.conftest import DummyDataset, DummyOptimizablePipeline
from tpcp import (
    make_optimize_safe,
    OptimizablePipeline,
    OptimizableParameter,
    PureParameter,
    HyperParameter,
    Dataset,
    make_action_safe,
    Pipeline,
    cf,
)
from tpcp.exceptions import PotentialUserErrorWarning


class PipelineInputModify(Pipeline):
    def __init__(self, test="a value"):
        self.test = test

    def run(self, datapoint: Dataset):
        self.test = "another value"
        return self


class PipelineInputModifyNested(Pipeline):
    def __init__(self, pipe: Pipeline = cf(PipelineInputModify())):
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


class DummyOptimizablePipelineUnsafe(OptimizablePipeline):
    optimized: OptimizableParameter[bool]
    para_1: PureParameter
    para_2: HyperParameter

    def __init__(self, para_1=None, para_2=None, optimized=False):
        self.para_1 = para_1
        self.para_2 = para_2
        self.optimized = optimized

    def self_optimize(self, dataset: Dataset, **kwargs):
        self.optimized = self.para_2
        return self


class DummyActionPipelineUnsafe(Pipeline):
    def __init__(self, para_1=None, para_2=None, optimized=False):
        self.para_1 = para_1
        self.para_2 = para_2
        self.optimized = optimized

    def run(self, ds):
        self.result_ = ds
        return self


class TestSafeAction:
    def test_double_wrapping_protection(self):
        before_wrapped = DummyActionPipelineUnsafe.run
        before_wrap_id = id(before_wrapped)
        wrapped = make_action_safe(before_wrapped)
        wrapped_id = id(wrapped)
        second_wrapped = make_action_safe(wrapped)
        second_wrapped_id = id(second_wrapped)
        assert before_wrap_id != wrapped_id
        assert second_wrapped_id == wrapped_id

    @pytest.mark.parametrize("name,warn", (("run", False), ("other_name", True)))
    def test_wrapper_checks_name(self, name, warn):
        test_func = DummyActionPipelineUnsafe.run
        test_func.__name__ = name
        ds = DummyDataset()

        warning = PotentialUserErrorWarning if warn else None
        with pytest.warns(warning) as w:
            make_action_safe(test_func)(DummyActionPipelineUnsafe(), ds)

        if warn:
            assert "The `make_action_safe` decorator should only be applied to an action methods" in str(w[0])

    @pytest.mark.parametrize("pipe", (PipelineInputModify, PipelineInputModifyNested))
    def test_modify_input_paras_simple(self, pipe):
        with pytest.raises(ValueError) as e:
            make_action_safe(pipe.run)(pipe(), DummyDataset()[0])

        assert f"Running `safe_run` of {pipe.__name__} did modify the parameters of the algorithm" in str(e)

    def test_no_self_return(self):
        pipe = DummyActionPipelineUnsafe
        pipe.run = lambda p, d: "some Value"
        with pytest.raises(ValueError) as e:
            make_action_safe(pipe.run)(pipe(), DummyDataset()[0])

        assert "method of DummyActionPipelineUnsafe must return `self`" in str(e)

    def test_no_output(self):
        with pytest.raises(ValueError) as e:
            make_action_safe(PipelineNoOutput.run)(PipelineNoOutput(), DummyDataset()[0])

        assert "Running the `run` method of PipelineNoOutput did not set any results on the output" in str(e)

    def test_output(self):
        pipe = DummyActionPipelineUnsafe()
        pipe.result_ = "some result"
        ds = DummyDataset()[0]
        with patch.object(DummyActionPipelineUnsafe, "run", return_value=pipe) as mock:
            instance = DummyActionPipelineUnsafe()
            mock.__name__ = "run"
            result = make_action_safe(DummyActionPipelineUnsafe.run)(instance, ds)

        mock.assert_called_with(instance, ds)
        assert id(result) == id(pipe)


class TestSafeOptimize:
    def test_double_wrapping_protection(self):
        before_wrapped = DummyOptimizablePipelineUnsafe.self_optimize
        before_wrap_id = id(before_wrapped)
        wrapped = make_optimize_safe(before_wrapped)
        wrapped_id = id(wrapped)
        second_wrapped = make_optimize_safe(wrapped)
        second_wrapped_id = id(second_wrapped)
        assert before_wrap_id != wrapped_id
        assert second_wrapped_id == wrapped_id

    @pytest.mark.parametrize("name,warn", (("self_optimize", False), ("other_name", True)))
    def test_wrapper_checks_name(self, name, warn):
        test_func = DummyOptimizablePipelineUnsafe.self_optimize
        test_func.__name__ = name
        ds = DummyDataset()

        warning = PotentialUserErrorWarning if warn else None
        with pytest.warns(warning) as w:
            make_optimize_safe(test_func)(DummyOptimizablePipelineUnsafe(), ds)

        if warn:
            assert "The `make_optimize_safe` decorator is only meant for the `self_optimize`" in str(w[0])

    @pytest.mark.parametrize("output,warn", (({}, True), (dict(optimized=True), False)))
    def test_optimize_warns(self, output, warn):
        optimized_pipe = DummyOptimizablePipelineUnsafe()
        for k, v in output.items():
            setattr(optimized_pipe, k, v)
        ds = DummyDataset()
        with patch.object(DummyOptimizablePipelineUnsafe, "self_optimize", return_value=optimized_pipe) as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipelineUnsafe.self_optimize = make_optimize_safe(
                DummyOptimizablePipelineUnsafe.self_optimize
            )
            warning = PotentialUserErrorWarning if warn else None
            with pytest.warns(warning) as w:
                DummyOptimizablePipelineUnsafe().self_optimize(ds)

            if len(w) > 0:
                assert "Optimizing the algorithm doesn't seem to have changed" in str(w[0])

    @pytest.mark.parametrize("output", (dict(some_random_para_="val"), dict(optimized=True, some_random_para_="val")))
    def test_other_para_modified_error(self, output):
        optimized_pipe = DummyOptimizablePipelineUnsafe()
        for k, v in output.items():
            setattr(optimized_pipe, k, v)
        ds = DummyDataset()
        with patch.object(DummyOptimizablePipelineUnsafe, "self_optimize", return_value=optimized_pipe) as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipelineUnsafe.self_optimize = make_optimize_safe(
                DummyOptimizablePipelineUnsafe.self_optimize
            )
            with pytest.raises(RuntimeError) as e:
                DummyOptimizablePipelineUnsafe().self_optimize(ds)

        assert "Optimizing seems to have changed class attributes that are not parameters" in str(e.value)

    def test_optimize_error(self):
        ds = DummyDataset()
        # return anything that is not of the optimizer class
        with patch.object(DummyOptimizablePipelineUnsafe, "self_optimize", return_value="some_value") as mock:
            mock.__name__ = "self_optimize"
            DummyOptimizablePipelineUnsafe.self_optimize = make_optimize_safe(
                DummyOptimizablePipelineUnsafe.self_optimize
            )
            with pytest.raises(ValueError) as e:
                DummyOptimizablePipelineUnsafe().self_optimize(ds)

        assert "Calling `self_optimize` did not return an instance" in str(e.value)
