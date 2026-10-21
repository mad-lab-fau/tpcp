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
from tpcp._config import config_context, get_config, set_config
from tpcp._dataset import Dataset, DatasetWrapperMixin
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

__version__ = "3.5.0"


__all__ = [
    "NOTHING",
    "Algorithm",
    "BaseFactory",
    "BaseTpcpObject",
    "CloneFactory",
    "Dataset",
    "DatasetWrapperMixin",
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
    "config_context",
    "get_action_method",
    "get_action_methods_names",
    "get_action_params",
    "get_config",
    "get_param_names",
    "get_results",
    "is_action_applied",
    "make_action_safe",
    "make_optimize_safe",
    "set_config",
]
