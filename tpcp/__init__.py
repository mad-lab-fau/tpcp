"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._algorithm import Algorithm, OptimizableAlgorithm
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

__version__ = "0.6.0"


__all__ = [
    "make_action_safe",
    "make_optimize_safe",
    "clone",
    "cf",
    "CloneFactory",
    "BaseFactory",
    "HyperParameter",
    "HyperPara",
    "PureParameter",
    "PurePara",
    "OptimizableParameter",
    "OptiPara",
    "Parameter",
    "Para",
    "Algorithm",
    "OptimizableAlgorithm",
    "BaseTpcpObject",
    "Dataset",
    "Pipeline",
    "OptimizablePipeline",
    "get_param_names",
    "get_action_params",
    "get_results",
    "get_action_method",
    "get_action_methods_names",
    "is_action_applied",
    "NOTHING",
]
