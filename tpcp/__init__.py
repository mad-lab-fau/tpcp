"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._dataset import Dataset
from tpcp._pipelines import OptimizablePipeline, SimplePipeline
from tpcp._utils._general import clone, default, safe_action
from tpcp.base import BaseAlgorithm, BaseTpcpObject, OptimizableAlgorithm
from tpcp.base import BaseAlgorithm, OptimizableAlgorithm

__version__ = "0.3.1"


mdf = default

__all__ = [
    "default",
    "mdf",
    "clone",
    "safe_action",
    "BaseAlgorithm",
    "OptimizableAlgorithm",
    "BaseTpcpObject",
    "Dataset",
    "SimplePipeline",
    "OptimizablePipeline",
]
