from typing import Dict
from unittest import mock
from unittest.mock import patch

import pytest

from tests.test_pipelines.conftest import DummyDataset
from tpcp import (
    Dataset,
    HyperParameter,
    OptimizableParameter,
    OptimizablePipeline,
    Parameter,
    Pipeline,
    PureParameter,
    cf,
    make_action_safe,
    make_optimize_safe,
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

    @pytest.mark.parametrize(("name", "warn"), (("run", False), ("other_name", True)))
    def test_wrapper_checks_name(self, name, warn):
        test_func = DummyActionPipelineUnsafe.run
        test_func.__name__ = name
        ds = DummyDataset()

        warning = PotentialUserErrorWarning if warn else None
        with pytest.warns(warning) as w:
            make_action_safe(test_func)(DummyActionPipelineUnsafe(), ds)

        if warn:
            assert "The `make_action_safe` decorator should only be applied to an action method" in str(w[0])

    @pytest.mark.parametrize("pipe", (PipelineInputModify, PipelineInputModifyNested))
    def test_modify_input_paras_simple(self, pipe):
        with pytest.raises(ValueError) as e:
            make_action_safe(pipe.run)(pipe(), DummyDataset()[0])

        assert f"Running `run` of {pipe.__name__} did modify the parameters of the algorithm" in str(e)

    def test_no_self_return(self):
        pipe = DummyActionPipelineUnsafe
        pipe.run = lambda p, d: "some Value"
        with pytest.raises(TypeError) as e:
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

    @pytest.mark.parametrize(("name", "warn"), (("self_optimize", False), ("other_name", True)))
    def test_wrapper_checks_name(self, name, warn):
        test_func = DummyOptimizablePipelineUnsafe.self_optimize
        test_func.__name__ = name
        ds = DummyDataset()

        warning = PotentialUserErrorWarning if warn else None
        with pytest.warns(warning) as w:
            make_optimize_safe(test_func)(DummyOptimizablePipelineUnsafe(), ds)

        if warn:
            assert "The `make_optimize_safe` decorator is only meant for the `self_optimize`" in str(w[0])

    @pytest.mark.parametrize(("output", "warn"), (({}, True), ({"optimized": True}, False)))
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

    @pytest.mark.parametrize("output", ({"some_random_para_": "val"}, {"optimized": True, "some_random_para_": "val"}))
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
            with pytest.raises(TypeError) as e:
                DummyOptimizablePipelineUnsafe().self_optimize(ds)

        assert "Calling `self_optimize`/`self_optimize_with_info` did not return an instance" in str(e.value)

    class PipelineNoOptiParas(OptimizablePipeline):
        para: Parameter[int]

        def __init__(self, para=3):
            self.para = para

    def test_no_opti_para(self):
        with pytest.raises(ValueError) as e:
            make_optimize_safe(self.PipelineNoOptiParas.self_optimize)(self.PipelineNoOptiParas(), DummyDataset())
        assert f"No parameter of {self.PipelineNoOptiParas.__name__} was marked as optimizable" in str(e)

    class PipelineModifyOtherParas(OptimizablePipeline):
        para: Parameter[int]
        opti_para: OptimizableParameter[int]

        def __init__(self, para=3, opti_para: int = 0):
            self.para = para
            self.opti_para = opti_para

        def self_optimize(self, dataset: Dataset, **kwargs):
            self.opti_para += 1
            self.para += 1
            return self

    class PipelineModifyOtherParasMutable(OptimizablePipeline):
        para: Parameter[Dict]
        opti_para: OptimizableParameter[int]

        def __init__(self, para=cf({}), opti_para: int = 0):
            self.para = para
            self.opti_para = opti_para

        def self_optimize(self, dataset: Dataset, **kwargs):
            self.opti_para += 1
            self.para["a"] = 3
            return self

    @pytest.mark.parametrize("klass", (PipelineModifyOtherParas, PipelineModifyOtherParasMutable))
    def test_non_opti_para_changed(self, klass):
        with pytest.raises(RuntimeError) as e:
            make_optimize_safe(klass.self_optimize)(klass(), DummyDataset())

        assert "optimizable: ['para']" in str(e.value)

    class PipelineModifyNestedParas(OptimizablePipeline):
        para: Parameter[int]
        opti_para: OptimizableParameter[int]
        nested: Parameter[DummyOptimizablePipelineUnsafe]

        def __init__(
            self, para=3, opti_para: int = 0, nested: DummyActionPipelineUnsafe = cf(DummyActionPipelineUnsafe())
        ):
            self.para = para
            self.opti_para = opti_para
            self.nested = nested

        def self_optimize(self, dataset: Dataset, **kwargs):
            self.opti_para += 1
            self.nested.para_1 = "bla"
            return self

    def test_nested_opti_para(self):
        with pytest.raises(RuntimeError) as e:
            make_optimize_safe(self.PipelineModifyNestedParas.self_optimize)(
                self.PipelineModifyNestedParas(), DummyDataset()
            )

        assert "optimizable: ['nested', 'nested__para_1']" in str(e.value)

    class PipelineModifyNestedParasSpecific(OptimizablePipeline):
        para: Parameter[int]
        opti_para: OptimizableParameter[int]
        nested__para_1: OptimizableParameter

        def __init__(self, para=3, opti_para: int = 0, nested: Pipeline = cf(DummyActionPipelineUnsafe())):
            self.para = para
            self.opti_para = opti_para
            self.nested = nested

        def self_optimize(self, dataset: Dataset, **kwargs):
            self.opti_para += 1
            self.nested.para_1 = "bla"
            self.nested.para_2 = "shouldn't change"
            return self

    def test_nested_opti_para_specific(self):
        with pytest.raises(RuntimeError) as e:
            make_optimize_safe(self.PipelineModifyNestedParasSpecific.self_optimize)(
                self.PipelineModifyNestedParasSpecific(), DummyDataset()
            )

        assert "optimizable: ['nested__para_2']" in str(e.value)

    class PipelineModifyNestedParasDeleted(OptimizablePipeline):
        para: Parameter[int]
        opti_para: OptimizableParameter[int]
        nested__para_1: OptimizableParameter

        def __init__(self, para=3, opti_para: int = 0, nested: Pipeline = cf(DummyActionPipelineUnsafe())):
            self.para = para
            self.opti_para = opti_para
            self.nested = nested

        def self_optimize(self, dataset: Dataset, **kwargs):
            self.opti_para += 1
            # We completely switch the nested class
            self.nested = PipelineInputModify()
            return self

    def test_nested_opti_para_deleted(self):
        with pytest.raises(RuntimeError) as e:
            # We just run it and expect no error
            make_optimize_safe(self.PipelineModifyNestedParasDeleted.self_optimize)(
                self.PipelineModifyNestedParasDeleted(), DummyDataset()
            )

        # The fact that "nested (added)" is shown in the error is not entirely correct, but the logic to find the
        # changed paras, is already complicated enough and the error messsage is verbose enough, so we will just
        # leave it.
        assert (
            "['nested__optimized (removed)', 'nested__para_2 (removed)', 'nested (added)', 'nested__test (added)']"
            in str(e.value)
        )

    class PipelineModifyNestedParasWorks(OptimizablePipeline):
        para: Parameter[int]
        opti_para: OptimizableParameter[int]
        nested__para_1: OptimizableParameter

        def __init__(self, para=3, opti_para: int = 0, nested: Pipeline = cf(DummyActionPipelineUnsafe())):
            self.para = para
            self.opti_para = opti_para
            self.nested = nested

        def self_optimize(self, dataset: Dataset, **kwargs):
            self.opti_para += 1
            self.nested.para_1 = "bla"
            return self

    def test_nested_opti_para_works(self):
        # We just run it and expect no error
        make_optimize_safe(self.PipelineModifyNestedParasWorks.self_optimize)(
            self.PipelineModifyNestedParasWorks(), DummyDataset()
        )


class TestPipelineSafeRun:
    @pytest.mark.parametrize("wrapped", (True, False))
    def test_checks_only_performed_once(self, wrapped):
        # We mock _check_safe_run from _algorithm_utils.py to test if it is really only called once
        with mock.patch("tpcp._algorithm_utils._check_safe_run") as mock_check:
            # We need to mock both imported places
            with mock.patch("tpcp._pipeline._check_safe_run") as mock_check_2:
                decorator = make_action_safe if wrapped else lambda x: x

                class PipelineWithManualWrappedRun(Pipeline):
                    @decorator
                    def run(self, datapoint: Dataset):
                        return self

                PipelineWithManualWrappedRun().safe_run(DummyDataset())

                if wrapped:
                    assert mock_check.call_count == 1
                    assert mock_check_2.call_count == 0
                else:
                    assert mock_check.call_count == 0
                    assert mock_check_2.call_count == 1
