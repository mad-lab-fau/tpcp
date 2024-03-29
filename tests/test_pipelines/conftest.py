from itertools import product

import pandas as pd

from tpcp import HyperParameter, OptimizableParameter, PureParameter, cf, make_optimize_safe
from tpcp._dataset import Dataset
from tpcp._pipeline import OptimizablePipeline, Pipeline


class DummyPipeline(Pipeline):
    def __init__(self, para_1=None, para_2=None, optimized=False):
        self.para_1 = para_1
        self.para_2 = para_2
        self.optimized = optimized


class DummyOptimizablePipeline(OptimizablePipeline):
    optimized: OptimizableParameter[bool]
    para_1: PureParameter
    para_2: HyperParameter

    def __init__(self, para_1=None, para_2=None, optimized=False):
        self.para_1 = para_1
        self.para_2 = para_2
        self.optimized = optimized

    @make_optimize_safe
    def self_optimize(self, dataset: Dataset, **kwargs):
        self.optimized = self.para_2
        return self


class DummyOptimizablePipelineWithInfo(OptimizablePipeline):
    optimized: OptimizableParameter[bool]
    para_1: PureParameter
    para_2: HyperParameter

    def __init__(self, para_1=None, para_2=None, optimized=False):
        self.para_1 = para_1
        self.para_2 = para_2
        self.optimized = optimized

    @make_optimize_safe
    def self_optimize_with_info(self, dataset: Dataset, **kwargs):
        self.optimized = self.para_2
        return self, "info"


class MutableCustomClass:
    test: str


class MutableParaPipeline(OptimizablePipeline):
    optimized: OptimizableParameter[bool]
    para_mutable: OptimizableParameter[bool]

    def __init__(self, para_normal=3, para_mutable: dict = cf(MutableCustomClass()), optimized=False):
        self.para_normal = para_normal
        self.para_mutable = para_mutable
        self.optimized = optimized

    @make_optimize_safe
    def self_optimize(self, dataset: Dataset, **kwargs):
        self.optimized = True
        self.para_mutable.test = True
        return self


class DummyDataset(Dataset):
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"value": list(range(5))})


class DummyGroupedDataset(Dataset):
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame(list(product("abc", range(5))), columns=["v1", "v2"])


def dummy_single_score_func(pipeline, data_point):
    return data_point.group_labels[0][0]


def create_dummy_score_func(name):
    return lambda x, y: getattr(x, name)


def create_dummy_multi_score_func(names):
    return lambda x, y: {"score_1": getattr(x, names[0]), "score_2": getattr(x, names[1])}


def dummy_multi_score_func(pipeline, data_point):
    return {"score_1": data_point.group_labels[0][0], "score_2": data_point.group_labels[0][0] + 1}


def dummy_error_score_func(pipeline, data_point):
    if data_point.group_labels[0][0] in [0, 2, 4]:
        raise ValueError(f"Dummy Error for {data_point.group_labels[0]}")
    return data_point.group_labels[0][0]


def dummy_error_score_func_multi(pipeline, data_point):
    tmp = dummy_error_score_func(pipeline, data_point)
    return {"score_1": tmp, "score_2": tmp}
