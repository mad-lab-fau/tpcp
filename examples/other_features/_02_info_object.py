import dataclasses

from typing_extensions import Self

from tpcp import Pipeline, OptimizablePipeline, Dataset, Algorithm, make_action_safe, OptiPara, make_optimize_safe
from tpcp.optimize import Optimize


@dataclasses.dataclass
class MyNestedAlgo(Algorithm):
    _action_methods = ("test",)
    para1: int

    @make_action_safe
    def test(self):
        self.info.add("self", self.get_params())
        self.my_result_ = 2
        return self


@dataclasses.dataclass
class MyPipeline(OptimizablePipeline):
    test_para: OptiPara[int]
    nested: MyNestedAlgo

    @make_optimize_safe
    def self_optimize(self, dataset: Dataset, **kwargs) -> Self:

        nested_clone = self.nested.clone().test()

        self.info.update(params=self.get_params(), test="bla").inherit("nested", nested_clone).add(
            "final value", "bla"
        ).add(self.test_para, "test_para")
        return self


print(Optimize(MyPipeline(3, MyNestedAlgo(2))).optimize(1).optimize_info_)
