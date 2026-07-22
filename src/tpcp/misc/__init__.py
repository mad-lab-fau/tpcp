"""Some general utilities for tpcp."""

from tpcp._hash import custom_hash
from tpcp.misc._class_utils import classproperty, set_defaults
from tpcp.misc._typed_iterator import BaseTypedIterator, TypedIterator, TypedIteratorResultTuple
from tpcp.misc._warning_error_context import (
    WarningErrorContext,
    WarningErrorContextRecord,
    iter_with_warning_error_context,
    warning_error_context,
)

__all__ = [
    "BaseTypedIterator",
    "TypedIterator",
    "TypedIteratorResultTuple",
    "WarningErrorContext",
    "WarningErrorContextRecord",
    "classproperty",
    "custom_hash",
    "iter_with_warning_error_context",
    "set_defaults",
    "warning_error_context",
]
