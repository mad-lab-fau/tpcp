"""Public type variables and scoring aliases for custom tpcp code.

The type variables preserve concrete subclasses in generic annotations. The scoring aliases
describe supported scoring callables and aggregator return values. Definitions live beside the
interfaces that use them and are re-exported here for application code.
"""

from tpcp._algorithm import AlgorithmT
from tpcp._base import BaseTpcpObjectT
from tpcp._dataset import DatasetT
from tpcp._pipeline import OptimizablePipelineT, PipelineT
from tpcp.validate._scorer import (
    AggReturnType,
    ScoreFunc,
    ScoreFuncMultiple,
    ScoreFuncSingle,
)

__all__ = [
    "AggReturnType",
    "AlgorithmT",
    "BaseTpcpObjectT",
    "DatasetT",
    "OptimizablePipelineT",
    "PipelineT",
    "ScoreFunc",
    "ScoreFuncMultiple",
    "ScoreFuncSingle",
]
