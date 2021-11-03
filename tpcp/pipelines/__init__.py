"""A collection of higher level algorithms to run, optimize and validate algorithms."""
from tpcp.pipelines._optimize import BaseOptimize, GridSearch, GridSearchCV, Optimize
from tpcp.pipelines._pipelines import OptimizablePipeline, SimplePipeline
from tpcp.pipelines._scorer import Scorer
from tpcp.pipelines._validation import cross_validate

__all__ = [
    "SimplePipeline",
    "OptimizablePipeline",
    "Scorer",
    "BaseOptimize",
    "GridSearch",
    "GridSearchCV",
    "Optimize",
    "cross_validate",
]
