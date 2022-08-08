"""Type only classes that should be used when trying to type tpcp derived classes.

Note, this module only imports them, but they are defined close to places where they are used.
"""


from tpcp._algorithm import AlgorithmT
from tpcp._base import BaseTpcpObjectObjT
from tpcp._dataset import DatasetT
from tpcp._pipeline import OptimizablePipelineT, PipelineT
from tpcp.validate._scorer import (
    AggReturnType,
    MultiScoreTypeT,
    ScoreFunc,
    ScoreFuncMultiple,
    ScoreFuncSingle,
    ScoreTypeT,
    SingleScoreTypeT,
)

__all__ = [
    "BaseTpcpObjectObjT",
    "PipelineT",
    "OptimizablePipelineT",
    "DatasetT",
    "AlgorithmT",
    "ScoreTypeT",
    "SingleScoreTypeT",
    "MultiScoreTypeT",
    "AggReturnType",
    "ScoreFuncSingle",
    "ScoreFuncMultiple",
    "ScoreFunc",
]
