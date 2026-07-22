import doctest
import time
import warnings

import joblib
import pytest

from tests._config_test import config, set_config
from tpcp import parallel
from tpcp.misc import warning_error_context
from tpcp.parallel import delayed, register_global_parallel_callback


@pytest.fixture(autouse=True)
def reset_parallel_context():
    parallel._PARALLEL_CONTEXT_CALLBACKS = {}
    yield
    parallel._PARALLEL_CONTEXT_CALLBACKS = {}


def test_simple_callback():
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


def test_warning_context_is_restored_in_worker_process():
    provider_state = {"calls": 0}

    def context_provider():
        provider_state["calls"] += 1
        return {"provider_call": provider_state["calls"]}

    def warn_in_worker():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("first", UserWarning, stacklevel=1)
            warnings.warn("second", UserWarning, stacklevel=1)
        return [str(warning.message) for warning in caught]

    with warning_error_context("parent", {"fixed": "value"}, context_provider=context_provider):
        result = joblib.Parallel(n_jobs=2)(delayed(warn_in_worker)() for _ in range(1))

    expected_context = "parent: fixed='value', provider_call=1"
    assert result == [
        [
            f"first\n[{expected_context}] first",
            f"second\n[{expected_context}] second",
        ]
    ]


def test_warning_context_record_only_setting_is_restored_in_worker_process():
    def warn_in_worker():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("worker warning", UserWarning, stacklevel=1)
        return len(caught)

    with warning_error_context("parent", record_only=True):
        result = joblib.Parallel(n_jobs=2)(delayed(warn_in_worker)() for _ in range(1))

    assert result == [0]


def test_warning_context_annotates_worker_process_error():
    def fail_in_worker():
        raise RuntimeError("worker failed")

    with (
        warning_error_context("parent", {"item": 3}),
        pytest.raises(RuntimeError, match="worker failed") as error,
    ):
        joblib.Parallel(n_jobs=2)(delayed(fail_in_worker)() for _ in range(1))

    assert error.value.__notes__ == ["Context: parent: item=3"]


def test_warning_filters_are_restored_in_worker_process():
    def warn_in_worker(delay=0):
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("worker warning", UserWarning, stacklevel=1)
        time.sleep(delay)
        return len(caught)

    with joblib.Parallel(n_jobs=2, batch_size=1) as parallel:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = parallel(delayed(warn_in_worker)(0.1) for _ in range(4))
        restored_result = parallel(joblib.delayed(warn_in_worker)() for _ in range(4))

    assert result == [0, 0, 0, 0]
    assert restored_result == [1, 1, 1, 1]


def test_warning_filters_are_not_overridden_in_worker_threads():
    def warn_in_worker():
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("worker warning", UserWarning, stacklevel=1)
        return len(caught)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task = delayed(warn_in_worker)()

    with warnings.catch_warnings():
        warnings.simplefilter("always", UserWarning)
        result = joblib.Parallel(n_jobs=2, backend="threading")([task])

    assert result == [1]


def test_warning_filter_changes_from_callback_setter_do_not_leak():
    def callback():
        def setter(_):
            warnings.simplefilter("ignore", UserWarning)

        return None, setter

    def warn_in_worker():
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("worker warning", UserWarning, stacklevel=1)
        return len(caught)

    register_global_parallel_callback(callback)
    with joblib.Parallel(n_jobs=2) as parallel_runner:
        filtered_result = parallel_runner(delayed(warn_in_worker)() for _ in range(2))
        restored_result = parallel_runner(joblib.delayed(warn_in_worker)() for _ in range(2))

    assert filtered_result == [0, 0]
    assert restored_result == [1, 1]


def test_doctest():
    from tpcp import parallel

    doctest_results = doctest.testmod(m=parallel)
    assert doctest_results.failed == 0
