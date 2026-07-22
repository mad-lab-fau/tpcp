import doctest
import time
import warnings

import joblib
import pytest

from tests._config_test import add_to_side_channel, config, restore_side_channel, set_config
from tpcp import parallel
from tpcp.misc import print_with_context, warning_error_context
from tpcp.parallel import (
    Parallel,
    delayed,
    register_global_parallel_callback,
    register_parallel_side_channel,
    remove_parallel_side_channel,
)


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


@pytest.mark.parametrize("parallel_class", [joblib.Parallel, Parallel])
def test_warning_context_annotates_worker_process_error(parallel_class):
    def fail_in_worker():
        raise RuntimeError("worker failed")

    with (
        warning_error_context("parent", {"item": 3}),
        pytest.raises(RuntimeError, match="worker failed") as error,
    ):
        parallel_class(n_jobs=2)(delayed(fail_in_worker)() for _ in range(1))

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


@pytest.mark.parametrize(("n_jobs", "backend"), [(1, None), (2, None), (2, "threading")])
def test_tpcp_parallel_recovers_worker_context_records(n_jobs, backend):
    def report_from_worker():
        warnings.warn("worker warning", UserWarning, stacklevel=1)
        print_with_context("worker print")
        return "worker result"

    with warning_error_context("parent", {"run": 1}, record_only=True) as context:
        result = Parallel(n_jobs=n_jobs, backend=backend)(delayed(report_from_worker)() for _ in range(1))

    assert result == ["worker result"]
    assert [(record.type, record.context, str(record.message)) for record in context.records] == [
        ("warning", "parent: run=1", "worker warning"),
        ("print", "parent: run=1", "worker print"),
    ]
    assert isinstance(context.records[0].message, UserWarning)


def test_reused_delayed_callable_captures_context_when_each_task_is_created():
    def report_from_worker(label):
        print_with_context(label)
        return label

    task = delayed(report_from_worker)
    with warning_error_context("first", record_only=True) as first_context:
        first_task = task("first task")
    with warning_error_context("second", record_only=True) as second_context:
        second_task = task("second task")

    assert Parallel(n_jobs=2)([first_task, second_task]) == ["first task", "second task"]
    assert [(record.context, record.message) for record in first_context.records] == [("first", "first task")]
    assert [(record.context, record.message) for record in second_context.records] == [("second", "second task")]


def test_tpcp_parallel_recovers_records_as_generator_is_consumed():
    def report_from_worker(i):
        warnings.warn(f"worker warning {i}", UserWarning, stacklevel=1)
        return i

    with warning_error_context("parent", record_only=True) as context:
        results = Parallel(n_jobs=2, return_as="generator")(delayed(report_from_worker)(i) for i in range(5))
        assert context.records == []
        assert list(results) == list(range(5))

    assert [str(record.message) for record in context.records] == [f"worker warning {i}" for i in range(5)]


def test_joblib_parallel_does_not_expose_or_recover_tpcp_side_channel_data():
    def report_from_worker():
        warnings.warn("worker warning", UserWarning, stacklevel=1)
        return "worker result"

    with warning_error_context("parent", record_only=True) as context:
        result = joblib.Parallel(n_jobs=2)(delayed(report_from_worker)() for _ in range(1))

    assert result == ["worker result"]
    assert context.records == []


@pytest.mark.parametrize(("n_jobs", "backend"), [(1, None), (2, None), (2, "threading")])
def test_custom_parallel_side_channel_restores_worker_state_and_collects_results(n_jobs, backend):
    collected = []
    name = register_parallel_side_channel(
        lambda: "captured state",
        restore_side_channel,
        collected.extend,
    )
    try:
        results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(add_to_side_channel)(i) for i in range(3))
    finally:
        remove_parallel_side_channel(name)

    assert results == [0, 1, 2]
    assert collected == [("captured state", 0), ("captured state", 1), ("captured state", 2)]


@pytest.mark.parametrize(
    "register",
    [
        lambda: register_global_parallel_callback(lambda: (None, lambda _: None), name="__tpcp_internal__.custom"),
        lambda: register_parallel_side_channel(
            lambda: None,
            restore_side_channel,
            lambda _: None,
            name="__tpcp_internal__.custom",
        ),
        lambda: parallel.remove_global_parallel_callback("__tpcp_internal__.custom"),
        lambda: remove_parallel_side_channel("__tpcp_internal__.custom"),
    ],
)
def test_tpcp_parallel_prefix_is_reserved(register):
    with pytest.raises(ValueError, match="reserved for TPCP"):
        register()


def test_doctest():
    from tpcp import parallel

    doctest_results = doctest.testmod(m=parallel)
    assert doctest_results.failed == 0
