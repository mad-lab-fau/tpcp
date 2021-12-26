"""Type annotations to indicate the use case of specific parameters."""
from enum import Enum, auto
from inspect import cleandoc
from typing import TypeVar

from typing_extensions import Annotated

T = TypeVar("T")


class _ParaTypes(Enum):
    HYPER = auto()
    SIMPLE = auto()
    PURE = auto()
    OPTI = auto()


Parameter = Annotated[T, _ParaTypes.SIMPLE]
Parameter.__doc__ = cleandoc(
    """Mark class attribute as a simple parameter for an algorithm or pipeline.

Generally this is not required, as all parameters listed in the init and not annotated by any other fields types,
are considered plain parameters.
However, if you want to be explicit you can use this type annotation.
"""
)
Para = Parameter
HyperParameter = Annotated[T, _ParaTypes.HYPER]
HyperParameter.__doc__ = cleandoc(
    """Mark class attribute as a hyper-parameter for an algorithm or pipeline.

Compared to normal parameters (:func:`~tpcp.parameter`), hyper-parameters must only be specified for optimizable
Algorithms or Pipelines.
Hyper-Parameter are expected to change the outcome of the `self_optimize` method, but not change themself during the
optimization procedure.
This information can be used for internal checks and performance optimizations.
"""
)
HyperPara = HyperParameter
PureParameter = Annotated[T, _ParaTypes.PURE]
PureParameter.__doc__ = cleandoc(
    """Mark a class attribute as pure parameter for an algorithm or pipeline.

Compared to normal parameters (:func:`~tpcp.parameter`), pure parameters must only be specified for optimizable
Algorithms or Pipelines.
Pure parameters are expected to **not** influence the outcome of self optimize.
This information can be used for internal checks and performance optimizations.
These are most typically used in pipelines with multiple steps, that have an initial ML part that can be optimized
independently and a second non-ML part, that still has some parameters.
The knowledge of what parameter do not influence the outcome of optimization can be used to dramatically reduce the
complexity of black box based parameter optimizations (learn more TODO).
However, using `pure_parameter` incorrectly can lead to hard to detect issues.
If you are unsure, just mark a parameter as `parameter` as `pure_parameter`.
This has no negative side effect, besides disabling potential performance optimizations.
"""
)
PurePara = PureParameter
OptimizableParameter = Annotated[T, _ParaTypes.OPTI]
OptimizableParameter.__doc__ = cleandoc(
    """Mark class attribute as an optimizable parameter for an algorithm or pipeline.

Compared to normal parameters (:func:`~tpcp.parameter`), optimizable parameters must only be specified for optimizable
Algorithms or Pipelines.
Optimizable parameters are expected to be modified when calling `self_optimize`.
This information can be used for internal checks and performance optimizations.
Further, this means there default values will likely be overwritten.
In most cases you should still specify sensible default values so that the algorithm or pipeline can be used without
explicit optimization.
The default values can further be used as a starting point for optimization.
"""
)
OptiPara = OptimizableParameter
