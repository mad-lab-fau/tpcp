"""A test module required by the test_global_parallel_callback.py test."""

_GLOBAL_CONFIG = None


def set_config(value):
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = value


def config():
    return _GLOBAL_CONFIG
