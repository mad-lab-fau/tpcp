import doctest

import joblib

from tests._config_test import config, set_config
from tpcp import parallel
from tpcp.parallel import delayed, register_global_parallel_callback


def test_simple_callback():
    # We reset everything to avoid side effects
    parallel._PARALLEL_CONTEXT_CALLBACKS = []

    set_config("set")

    def callback():
        def setter(config):
            set_config(config)

        return config(), setter

    def func():
        return config()

    # First we test with normal joblib parallel
    # This is expected to not work
    assert joblib.Parallel(n_jobs=2)(joblib.delayed(func)() for _ in range(2)) == [None, None]

    # Now we test with our custom parallel
    # This is expected to work
    register_global_parallel_callback(callback)

    assert joblib.Parallel(n_jobs=2)(delayed(func)() for _ in range(2)) == ["set", "set"]

    # We reset everything to avoid side effects
    parallel._PARALLEL_CONTEXT_CALLBACKS = []


def test_doctest():
    from tpcp import parallel

    # We reset everything to avoid side effects
    parallel._PARALLEL_CONTEXT_CALLBACKS = []

    doctest_results = doctest.testmod(m=parallel)
    assert doctest_results.failed == 0

    # We reset everything to avoid side effects
    parallel._PARALLEL_CONTEXT_CALLBACKS = []
