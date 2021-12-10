"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._algorithm import Algorithm, BaseTpcpObject, OptimizableAlgorithm
from tpcp._algorithm_utils import get_action_params, get_results, make_action_safe, make_optimize_safe
from tpcp._base import BaseFactory, CloneFactory, cf, clone, get_param_names
from tpcp._dataset import Dataset
from tpcp._parameters import (
    HyperParameter,
    PureParameter,
    HyperPara,
    PurePara,
    OptimizableParameter,
    OptiPara,
    Parameter,
    Para,
)
from tpcp._pipeline import OptimizablePipeline, Pipeline

__version__ = "0.3.1"


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
]
