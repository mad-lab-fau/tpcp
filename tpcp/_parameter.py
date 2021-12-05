"""Parameter helper functions to build Algorithm and Pipeline classes."""
from functools import partial
from inspect import cleandoc
from typing import Any, Optional, Callable, List, Dict, Union

import attr

PARAMETER_FIELD_TYPE = "__TPCP_PARAMETER_FIELD_TYPE"


def _is_tpcp_parameter_field(field: attr.Attribute, field_type: Optional[str] = None):
    tmp = PARAMETER_FIELD_TYPE in field.metadata
    if field_type is None:
        return tmp
    return tmp and field.metadata[PARAMETER_FIELD_TYPE] == field_type


def _parameter(
    default: Any = attr.NOTHING,
    validator: Optional[Union[Callable, List[Callable]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    factory: Optional[Callable] = None,
    kw_only: bool = True,
    para_name: Optional[str] = None,
):
    metadata = {} if not metadata else metadata
    if para_name is not None:
        metadata[PARAMETER_FIELD_TYPE] = para_name
    return attr.field(
        default=default,
        validator=validator,
        repr=True,
        init=True,
        metadata=metadata,
        factory=factory,
        kw_only=kw_only,
    )


parameter = partial(_parameter, para_name="simple")
parameter.__name__ = "parameter"
parameter.__doc__ = cleandoc(
    """Mark class attribute as a simple parameter for an algorithm or pipeline.

All attributes marked this way, will be included in the automatically generated `__init__`.

"""
)
para = parameter

hyper_parameter = partial(_parameter, para_name="hyper")
hyper_parameter.__name__ = "hyper_parameter"
hyper_parameter.__doc__ = cleandoc(
    """Mark class attribute as a hyper-parameter for an algorithm or pipeline.

Compared to normal parameters (:func:`~tpcp.parameter`), hyper-parameters must only be specified for optimizable
Algorithms or Pipelines.
Like simple parameters these parameters are included in the `__init__`.

Hyper-Parameter are expected to change the outcome of the `self_optimize` method, but not change themself during the
optimization procedure.
This information can be used for internal checks and performance optimizations.

"""
)
hyperpara = hyper_parameter

optimizable_parameter = partial(_parameter, para_name="optimizable")
optimizable_parameter.__name__ = "optimizable_parameter"
optimizable_parameter.__doc__ = cleandoc(
    """Mark class attribute as an optimizable parameter for an algorithm or 
pipeline.

Compared to normal parameters (:func:`~tpcp.parameter`), optimizable parameters must only be specified for optimizable
Algorithms or Pipelines.
Like simple parameters these parameters are included in the `__init__`.

Optimizable parameters are expected to be modified when calling `self_optimize`.
This information can be used for internal checks and performance optimizations.
Further, this means there default values will likely be overwritten.
In most cases you should still specify sensible default values so that the algorithm or pipeline can be used without
explicit optimization.
The default values can further be used as a starting point for optimization.

"""
)
optipara = optimizable_parameter

pure_parameter = partial(_parameter, para_name="pure")
pure_parameter.__name__ = "pure_parameter"
pure_parameter.__doc__ = cleandoc(
    """Mark a class attribute as pure parameter for an algorithm or pipeline.

Compared to normal parameters (:func:`~tpcp.parameter`), pure parameters must only be specified for optimizable
Algorithms or Pipelines.
Like simple parameters these parameters are included in the `__init__`.

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
purepara = pure_parameter


