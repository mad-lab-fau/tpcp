import gc
from threading import Lock

import pytest
from joblib import Parallel as JoblibParallel
from joblib.externals.loky import ProcessPoolExecutor

from tpcp import Algorithm, Pipeline, config_context, get_config, make_action_safe, set_config
from tpcp.parallel import Parallel, delayed


class IncorrectAction(Algorithm):
    _action_methods = "run"

    def __init__(self, value=1):
        self.value = value

    @make_action_safe
    def run(self, fault):
        if fault == "parameters":
            self.value += 1
        if fault == "return":
            return None
        if fault != "results":
            self.result_ = self.value
        return self


def test_config_context_restores_nested_settings_after_exception():
    assert get_config()["action_safety_checks"] is True
    with config_context(action_safety_checks=False):
        assert get_config()["action_safety_checks"] is False
        with pytest.raises(RuntimeError, match="inside scope"):
            with config_context(action_safety_checks=True):
                assert get_config()["action_safety_checks"] is True
                raise RuntimeError("inside scope")
        assert get_config()["action_safety_checks"] is False
    assert get_config()["action_safety_checks"] is True


def test_set_config_is_persistent_within_current_context():
    with config_context():
        set_config(action_safety_checks=False)
        assert get_config()["action_safety_checks"] is False
        set_config()
        assert get_config()["action_safety_checks"] is False
    assert get_config()["action_safety_checks"] is True


@pytest.mark.parametrize(
    ("fault", "error"), [("parameters", ValueError), ("return", TypeError), ("results", ValueError)]
)
def test_action_safety_can_be_disabled_and_reenabled(fault, error):
    with config_context(action_safety_checks=False):
        instance = IncorrectAction()
        result = instance.run(fault)
        assert result is (None if fault == "return" else instance)
        with config_context(action_safety_checks=True), pytest.raises(error):
            IncorrectAction().run(fault)
    with pytest.raises(error):
        IncorrectAction().run(fault)


def test_safe_run_can_be_disabled_for_undecorated_pipeline():
    class IncorrectPipeline(Pipeline):
        def run(self, datapoint):
            return None

    with config_context(action_safety_checks=False):
        assert IncorrectPipeline().safe_run(None) is None
    with pytest.raises(TypeError, match="must return"):
        IncorrectPipeline().safe_run(None)


def test_disabled_safety_does_not_serialize_parameters():
    with config_context(action_safety_checks=False):
        assert IncorrectAction(value=Lock()).run("return") is None


def _unchecked_action_in_worker():
    return get_config()["action_safety_checks"], IncorrectAction().run("return")


@pytest.mark.parametrize(
    ("runner", "n_jobs", "backend"),
    [(Parallel, 1, None), (Parallel, 2, None), (Parallel, 2, "threading"), (JoblibParallel, 2, None)],
)
def test_action_setting_is_captured_for_parallel_tasks(runner, n_jobs, backend):
    with config_context(action_safety_checks=False):
        tasks = [delayed(_unchecked_action_in_worker)() for _ in range(2)]
    assert runner(n_jobs=n_jobs, backend=backend)(tasks) == [(False, None), (False, None)]
    assert get_config()["action_safety_checks"] is True


@pytest.mark.parametrize(("n_jobs", "backend"), [(1, None), (2, None), (2, "threading")])
def test_config_scope_can_span_multiple_lazy_task_yields(n_jobs, backend):
    def tasks():
        with config_context(action_safety_checks=False):
            for _ in range(3):
                yield delayed(_unchecked_action_in_worker)()

    assert Parallel(n_jobs=n_jobs, backend=backend)(tasks()) == [(False, None)] * 3
    assert get_config()["action_safety_checks"] is True


def _fail_in_worker():
    raise RuntimeError("task failure")


def test_lazy_producer_scope_is_closed_in_its_context_after_task_failure():
    cleanup_settings = []

    def tasks():
        with config_context(action_safety_checks=False):
            try:
                for _ in range(3):
                    yield delayed(_fail_in_worker)()
            finally:
                cleanup_settings.append(get_config()["action_safety_checks"])

    with pytest.raises(RuntimeError, match="task failure"):
        Parallel(n_jobs=1)(tasks())
    gc.collect()
    assert cleanup_settings == [False]
    assert get_config()["action_safety_checks"] is True


def _mutate_worker_setting(fail):
    set_config(action_safety_checks=False)
    if fail:
        raise RuntimeError("worker failed")
    return get_config()["action_safety_checks"]


@pytest.mark.parametrize("fail", [False, True])
def test_worker_configuration_is_restored_after_each_task(fail):
    task, args, kwargs = delayed(_mutate_worker_setting)(fail)
    with ProcessPoolExecutor(max_workers=1) as executor:
        assert executor.submit(get_config).result()["action_safety_checks"] is True
        result = executor.submit(task, *args, **kwargs)
        if fail:
            with pytest.raises(RuntimeError, match="worker failed"):
                result.result()
        else:
            assert result.result() is False
        # The same worker also receives an ordinary task with no TPCP restoration wrapper.
        assert executor.submit(get_config).result()["action_safety_checks"] is True
