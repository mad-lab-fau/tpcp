"""A set of helper functions to correctly handle global variables in parallel processes.

Both functions are required as long as https://github.com/joblib/joblib/issues/1071 is not resolved.
"""
from tpcp._utils._multiprocess import delayed, register_global_parallel_callback

__all__ = ["delayed", "register_global_parallel_callback"]
