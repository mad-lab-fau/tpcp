"""Some general utilities for tpcp."""
from tpcp._hash import custom_hash
from tpcp.misc._class_utils import classproperty, set_defaults
from tpcp.misc._typed_iterator import BaseTypedIterator, TypedIterator, TypedIteratorResultTuple

__all__ = [
    "TypedIterator",
    "BaseTypedIterator",
    "classproperty",
    "set_defaults",
    "custom_hash",
    "TypedIteratorResultTuple",
]
