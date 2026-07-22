"""A test module required by the test_global_parallel_callback.py test."""

from contextlib import contextmanager

_GLOBAL_CONFIG = None
_SIDE_CHANNEL = []


def set_config(value):
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = value


def config():
    return _GLOBAL_CONFIG


@contextmanager
def restore_side_channel(prefix):
    _SIDE_CHANNEL.clear()
    try:
        yield lambda: tuple((prefix, value) for value in _SIDE_CHANNEL)
    finally:
        _SIDE_CHANNEL.clear()


def add_to_side_channel(value):
    _SIDE_CHANNEL.append(value)
    return value
