"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._algorithm import Algorithm, BaseTpcpObject, OptimizableAlgorithm
from tpcp._algorithm_utils import make_action_safe, make_optimize_safe, get_action_params, get_results
from tpcp._base import clone, get_param_names, CloneFactory, cf
from tpcp._dataset import Dataset
from tpcp._parameter import (
    parameter,
    para,
    hyper_parameter,
    hyperpara,
    optimizable_parameter,
    optipara,
    pure_parameter,
    purepara,
)
from tpcp._pipeline import OptimizablePipeline, Pipeline

__version__ = "0.3.1"


__all__ = [
    "make_action_safe",
    "make_optimize_safe",
    "clone",
    "parameter",
    "para",
    "hyper_parameter",
    "hyperpara",
    "optimizable_parameter",
    "optipara",
    "pure_parameter",
    "purepara",
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
