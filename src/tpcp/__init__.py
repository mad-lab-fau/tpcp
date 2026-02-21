"""tpcp - Tiny Pipelines for Complicated Problems."""

from tpcp._algorithm import Algorithm
from tpcp._algorithm_utils import (
    get_action_method,
    get_action_methods_names,
    get_action_params,
    get_results,
    is_action_applied,
    make_action_safe,
    make_optimize_safe,
)
from tpcp._base import NOTHING, BaseFactory, BaseTpcpObject, CloneFactory, cf, clone, get_param_names
from tpcp._dataset import Dataset
from tpcp._parameters import (
    HyperPara,
    HyperParameter,
    OptimizableParameter,
    OptiPara,
    Para,
    Parameter,
    PurePara,
    PureParameter,
)
from tpcp._pipeline import OptimizablePipeline, Pipeline

__version__ = "2.1.1"


__all__ = [
    "NOTHING",
    "Algorithm",
    "BaseFactory",
    "BaseTpcpObject",
    "CloneFactory",
    "Dataset",
    "HyperPara",
    "HyperParameter",
    "OptiPara",
    "OptimizableParameter",
    "OptimizablePipeline",
    "Para",
    "Parameter",
    "Pipeline",
    "PurePara",
    "PureParameter",
    "cf",
    "clone",
    "get_action_method",
    "get_action_methods_names",
    "get_action_params",
    "get_param_names",
    "get_results",
    "is_action_applied",
    "make_action_safe",
    "make_optimize_safe",
]
