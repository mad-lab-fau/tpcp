"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._dataset import Dataset
from tpcp._pipelines import OptimizablePipeline, SimplePipeline
from tpcp._utils._general import clone, default, safe_action, safe_optimize
from tpcp.base import BaseAlgorithm, BaseTpcpObject, OptimizableAlgorithm

__version__ = "0.3.1"


mdf = default

__all__ = [
    "default",
    "mdf",
    "clone",
    "safe_action",
    "safe_optimize",
    "BaseAlgorithm",
    "OptimizableAlgorithm",
    "BaseTpcpObject",
    "Dataset",
    "SimplePipeline",
    "OptimizablePipeline",
]
