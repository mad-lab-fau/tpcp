"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._utils._general import clone, default
from tpcp.base import BaseAlgorithm, BaseOptimize

__version__ = "0.2.0-alpha.3"

mdf = default
__all__ = ["BaseAlgorithm", "BaseOptimize", "default", "mdf", "clone"]
