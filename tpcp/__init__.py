"""tpcp - Tiny Pipelines for Complicated Problems."""
from tpcp._utils._general import default, clone
from tpcp.base import BaseAlgorithm, BaseOptimize

__version__ = "0.2.0-alpha.0"

df = default
__all__ = ["BaseAlgorithm", "BaseOptimize", "default", "df", "clone"]
