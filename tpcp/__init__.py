"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._algorithm import Algorithm, BaseTpcpObject, OptimizableAlgorithm
from tpcp._algorithm_utils import get_action_params, get_results, make_action_safe, make_optimize_safe
from tpcp._base import CloneFactory, cf, clone, get_param_names, BaseFactory
from tpcp._dataset import Dataset
from tpcp._parameter import (
    hyper_parameter,
    hyperpara,
    optimizable_parameter,
    optipara,
    para,
    parameter,
    pure_parameter,
    purepara,
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
    "Algorithm",
    "OptimizableAlgorithm",
    "BaseTpcpObject",
    "Dataset",
    "Pipeline",
    "OptimizablePipeline",
    "get_param_names",
    "get_action_params",
    "get_results"
]
