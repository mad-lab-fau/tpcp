import warnings
from functools import partial
from typing import Callable, Literal

import joblib
import numpy as np
import pandas as pd
import pytest
from joblib import Memory
from joblib.externals.loky import get_reusable_executor

from tests._example_pipelines import CacheWarning, ExampleClassOtherModule
from tpcp import Algorithm
from tpcp.caching import _is_cached, global_disk_cache, global_ram_cache, hybrid_cache, remove_any_cache


class ExampleClass(Algorithm):
    _action_methods = ["action"]

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def action(self, x):
        self.result_1_ = x + self.a + self.b
        self.result_2_ = x + self.a - self.b

        # We use a warning here to make it detectable that the function was called.
        warnings.warn("This function was called without caching.", CacheWarning)
        return self


def example_func(a, b):
    warnings.warn("This function was called without caching.", CacheWarning)

    return a + b


class ExampleClassMultiAction(ExampleClass):
    _action_methods = ["action", "action_2"]

    def action_2(self, x):
        self.result_1_ = (x + self.a + self.b) * 2
        self.result_2_ = (x + self.a - self.b) * 2

        # We use a warning here to make it detectable that the function was called.
        warnings.warn("This function was called without caching.", CacheWarning)
        return self


@pytest.fixture(params=(({}, ExampleClass), ({"action_method_name": "action_2"}, ExampleClassMultiAction)))
def example_class(request):
    yield request.param
    remove_any_cache(request.param[1])


@pytest.fixture
def simple_example_class(request):
    yield ExampleClassOtherModule
    remove_any_cache(ExampleClassOtherModule)


@pytest.fixture
def joblib_cache():
    memory = joblib.Memory(location=".cache", verbose=0)
    yield memory
    memory.clear()


@pytest.fixture
def joblib_cache_verbose():
    memory = joblib.Memory(location=".cache", verbose=10)
    yield memory
    memory.clear()


@pytest.fixture
def hybrid_cache_clear():
    yield None
    hybrid_cache.__cache_registry__.clear()


class TestGlobalCache:
    cache_method: Callable[[type[Algorithm]], type[Algorithm]]
    cache_method_name: Literal["disk", "ram"]

    @pytest.fixture(autouse=True, params=["disk", "ram"])
    def get_cache_method(self, request, joblib_cache):
        if request.param == "disk":
            self.cache_method = partial(global_disk_cache, joblib_cache)

        else:
            self.cache_method = partial(global_ram_cache, None)
        self.cache_method_name = request.param

    def test_caching_twice_same_instance(self, example_class):
        config, example_class = example_class
        action_name = config.get("action_method_name", "action")
        multiplier = 2 if action_name == "action_2" else 1
        self.cache_method(**config)(example_class)
        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            getattr(example, action_name)(3)

        assert example.result_1_ == 6 * multiplier

        with pytest.warns(CacheWarning):
            getattr(example, action_name)(2)
        assert example.result_1_ == 5 * multiplier

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            getattr(example, action_name)(3)
        assert example.result_1_ == 6 * multiplier
        assert not w

    def test_numeric_dataframe_layouts_remain_distinct(self, example_class):
        import numpy as np
        import pandas as pd

        config, algorithm = example_class
        self.cache_method(**config)(algorithm)
        action = getattr(algorithm(1, 2), config.get("action_method_name", "action"))
        values = np.arange(12).reshape(4, 3)
        with pytest.warns(CacheWarning):
            action(pd.DataFrame(values, copy=False))
        with pytest.warns(CacheWarning):
            action(pd.DataFrame(np.asfortranarray(values), copy=False))

    def test_caching_twice_new_instance(self, example_class):
        config, example_class = example_class
        action_name = config.get("action_method_name", "action")
        multiplier = 2 if action_name == "action_2" else 1
        self.cache_method(**config)(example_class)
        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            getattr(example, action_name)(3)
        assert example.result_1_ == 6 * multiplier

        with pytest.warns(CacheWarning):
            getattr(example, action_name)(2)
        assert example.result_1_ == 5 * multiplier

        example = example_class(1, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            getattr(example, action_name)(3)
        assert example.result_1_ == 6 * multiplier
        assert not w

    def test_cache_invalidated_on_para_change(self, example_class):
        config, example_class = example_class
        action_name = config.get("action_method_name", "action")
        multiplier = 2 if action_name == "action_2" else 1

        self.cache_method(**config)(example_class)
        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            getattr(example, action_name)(3)
        assert example.result_1_ == 6 * multiplier

        example.set_params(a=4)

        with pytest.warns(CacheWarning):
            getattr(example, action_name)(3)
        assert example.result_1_ == 9 * multiplier

    def test_cache_only(self, example_class):
        config, example_class = example_class
        action_name = config.get("action_method_name", "action")
        multiplier = 2 if action_name == "action_2" else 1
        self.cache_method(cache_only=["result_1_"], **config)(example_class)

        # We expect the uncached and the cached version to both not have result_2_ available.

        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            getattr(example, action_name)(2)
        assert example.result_1_ == 5 * multiplier
        assert not hasattr(example, "result_2_")

        example = example.clone()

        # Now in the cached version
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            getattr(example, action_name)(2)
        assert not w
        assert example.result_1_ == 5 * multiplier
        assert not hasattr(example, "result_2_")

    def test_double_cache_warning(self, example_class):
        config, example_class = example_class
        action_name = config.get("action_method_name", "action")
        self.cache_method(**config)(example_class)
        with pytest.warns(
            UserWarning, match=f"The action method {action_name} of {example_class.__name__} is already cached"
        ):
            self.cache_method(**config)(example_class)

    @pytest.mark.parametrize("restore_in_parallel_process", [True, False])
    def test_cache_correctly_restored_in_parallel_process(self, simple_example_class, restore_in_parallel_process):
        from joblib import Parallel

        from tpcp.parallel import delayed

        self.cache_method(restore_in_parallel_process=restore_in_parallel_process)(simple_example_class)

        # Hot cache (only matters for disk)
        simple_example_class(1, 2).action(1)

        def worker_func(pipe):
            if restore_in_parallel_process is False:
                assert _is_cached(simple_example_class, "action") is False
            if self.cache_method_name == "disk" and restore_in_parallel_process is True:
                # Disk cache can work across processes. This means, already on the first call in the new process,
                # we should get the cached result.
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    pipe.action(1)
                assert not w
            else:
                # For RAM cache the cache is reset in the new process, so the first call is expected to be uncached.
                with pytest.warns(CacheWarning):
                    pipe.action(1)

            if restore_in_parallel_process is True:
                # Id we set the restore option to True, the second call should be correctly cached
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    pipe.action(1)
                assert not w
            else:
                with pytest.warns(CacheWarning):
                    pipe.action(1)

        Parallel(n_jobs=2)(delayed(worker_func)(simple_example_class(1, 2)) for _ in range(2))
        # This is important! Otherwise, the different parameterized versions of the test reuse the same processes.
        # Hence, the global caching will already be reactivated in the new process.
        get_reusable_executor().shutdown(wait=True, kill_workers=True)


class TestFurtherCachingStuff:
    def test_double_cache_error_disk_first(self, joblib_cache, simple_example_class):
        global_disk_cache(joblib_cache)(simple_example_class)
        with pytest.raises(ValueError):
            global_ram_cache()(simple_example_class)

    def test_double_cache_error_ram_first(self, joblib_cache, simple_example_class):
        global_ram_cache(None)(simple_example_class)
        with pytest.raises(ValueError):
            global_disk_cache(joblib_cache)(simple_example_class)


class TestHybridCache:
    def test_staggered_cache_all_disabled(self):
        cached_func = hybrid_cache(joblib.Memory(None), False)(example_func)

        with pytest.warns(CacheWarning):
            r = cached_func(1, 2)

        assert r == 3

    def test_staggered_cache_returns_from_registry(self, hybrid_cache_clear):
        cached_func_1 = hybrid_cache(joblib.Memory(None), False)(example_func)
        cached_func_2 = hybrid_cache(joblib.Memory(None), False)(example_func)

        assert cached_func_1 is cached_func_2

    def test_joblib_only(self, joblib_cache, hybrid_cache_clear):
        cached_func = hybrid_cache(joblib_cache, False)(example_func)

        with pytest.warns(CacheWarning):
            r = cached_func(1, 2)

        assert r == 3

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = cached_func(1, 2)

        assert r == 3
        assert not w

    def test_lru_only(self, hybrid_cache_clear):
        cached_func = hybrid_cache(Memory(None), 2)(example_func)

        with pytest.warns(CacheWarning):
            r = cached_func(1, 2)

        assert r == 3

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = cached_func(1, 2)

        assert r == 3
        assert not w

    def test_staggered_cache(self, joblib_cache_verbose, hybrid_cache_clear, capfd):
        cached_func = hybrid_cache(joblib_cache_verbose, 2)(example_func)

        with pytest.warns(CacheWarning):
            r = cached_func(1, 2)

        out = capfd.readouterr()
        clean_out = out.out.replace("\n", "")
        # This should have triggered the joblib cache
        assert "[Memory] Calling tests.test_caching.example_func" in clean_out

        assert r == 3

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = cached_func(1, 2)

        # This should not hit the joblib cache, as the lru cache should have been used
        out = capfd.readouterr()
        clean_out = out.out.replace("\n", "")

        assert clean_out == ""

        assert r == 3
        assert not w

    def test_joblib_cache_survives_clear(self, joblib_cache_verbose, hybrid_cache_clear, capfd):
        cached_func = hybrid_cache(joblib_cache_verbose, 2)(example_func)

        with pytest.warns(CacheWarning):
            r = cached_func(1, 2)

        out = capfd.readouterr()
        clean_out = out.out.replace("\n", "")
        # This should have triggered the joblib cache
        assert "[Memory] Calling tests.test_caching.example_func" in clean_out

        assert r == 3

        hybrid_cache.__cache_registry__.clear()

        cached_func_new = hybrid_cache(joblib_cache_verbose, 2)(example_func)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = cached_func_new(1, 2)

        # This time this should hit the joblib cache, as the lru cache should have been cleared
        out = capfd.readouterr()
        clean_out = out.out.replace("\n", "")

        assert "Loading example_func from" in clean_out

        assert r == 3
        assert not w

        # And now the lru cache should be used again
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = cached_func_new(1, 2)

        # This time this should hit the joblib cache, as the lru cache should have been cleared
        out = capfd.readouterr()
        clean_out = out.out.replace("\n", "")

        assert clean_out == ""

        assert r == 3
        assert not w


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("lru_size", [False, 2])
def test_fast_hybrid_cache_hits_and_detects_mutation(tmp_path, hybrid_cache_clear, disk, lru_size):
    calls = []

    def total(data):
        calls.append(1)
        return data.to_numpy().sum()

    memory = Memory(tmp_path if disk else None, verbose=0)
    cached = hybrid_cache(memory, lru_size, fast_inaccurate_hashing=True)(total)
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    frame = pd.DataFrame(values, copy=False)
    equivalent = pd.DataFrame(np.asfortranarray(values), copy=False)
    assert cached(frame) == 66
    assert cached(equivalent) == 66
    assert len(calls) == (1 if disk or lru_size else 2)
    frame.iloc[-1, -1] = 20
    assert cached(frame) == 75
    assert len(calls) == (2 if disk or lru_size else 3)


@pytest.mark.parametrize("fast", [False, True])
def test_hybrid_dataframe_layout_policy(tmp_path, hybrid_cache_clear, fast):
    calls = []

    def total(data):
        calls.append(1)
        return data.to_numpy().sum()

    cached = hybrid_cache(Memory(None), 2, fast_inaccurate_hashing=fast)(total)
    values = np.arange(12).reshape(4, 3)
    assert cached(pd.DataFrame(values, copy=False)) == 66
    assert cached(pd.DataFrame(np.asfortranarray(values), copy=False)) == 66
    assert len(calls) == (1 if fast else 2)


def test_fast_hybrid_cache_isolated_from_default_and_joblib(tmp_path, hybrid_cache_clear):
    calls = []

    def total(data):
        calls.append(1)
        return data.sum()

    memory = Memory(tmp_path, verbose=0)
    legacy = hybrid_cache(memory, 2)(total)
    fast = hybrid_cache(memory, 2, fast_inaccurate_hashing=True)(total)
    assert fast is hybrid_cache(memory, 2, fast_inaccurate_hashing=True)(total)
    assert legacy is hybrid_cache(memory, 2)(total)
    assert fast is not legacy
    values = np.arange(4)
    assert legacy(values) == fast(values) == 6
    assert len(calls) == 2
    # An ordinary joblib cache still shares the default disk entries.
    assert memory.cache(total)(values) == 6
    assert len(calls) == 2
    assert legacy(values) == fast(values) == 6
    assert len(calls) == 2


def test_fast_disk_cache_survives_ram_eviction_and_registry_clear(tmp_path, hybrid_cache_clear):
    calls = []

    def total(data, offset=0):
        calls.append(1)
        return data.sum() + offset

    memory = Memory(tmp_path, verbose=0)
    cached = hybrid_cache(memory, 1, fast_inaccurate_hashing=True)(total)
    assert cached(np.arange(4)) == 6
    assert cached(np.arange(4), offset=10) == 16
    assert cached(np.arange(4)) == 6
    assert len(calls) == 2
    hybrid_cache.__cache_registry__.clear()
    cached = hybrid_cache(memory, 1, fast_inaccurate_hashing=True)(total)
    # Disk argument binding treats keyword/positional/default arguments alike.
    assert cached(data=np.arange(4), offset=0) == 6
    assert len(calls) == 2


def test_fast_ram_keys_snapshot_mutable_arguments(hybrid_cache_clear):
    calls = []

    def total(data):
        calls.append(1)
        return data.sum()

    cached = hybrid_cache(Memory(None), 3, fast_inaccurate_hashing=True)(total)
    values = np.array([1, 2])
    assert cached(values) == 3
    values[0] = 10
    assert cached(values) == 12
    assert cached(np.array([1, 2])) == 3
    assert len(calls) == 2


@pytest.mark.parametrize("fast", [False, True])
def test_hybrid_disk_cache_invalidates_changed_function(tmp_path, hybrid_cache_clear, fast):
    def total(data):
        return data.sum()

    def twice_total(data):
        return data.sum() * 2

    cached = hybrid_cache(Memory(tmp_path, verbose=0), fast_inaccurate_hashing=fast)(total)
    assert cached(np.arange(4)) == 6
    total.__code__ = twice_total.__code__
    assert cached(np.arange(4)) == 12


@pytest.mark.parametrize(
    "data",
    [
        np.arange(12).reshape(4, 3)[:, ::2],
        pd.DataFrame({"a": ["one", "two"]}, dtype=object),
        pd.DataFrame({"a": pd.Series([1, None], dtype="Int64")}),
        pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]}),
    ],
)
def test_fast_hybrid_nested_and_fallback_arguments(tmp_path, hybrid_cache_clear, data):
    calls = []

    def compute(payload):
        calls.append(1)
        return len(payload["data"])

    cached = hybrid_cache(Memory(tmp_path, verbose=0), 2, fast_inaccurate_hashing=True)(compute)
    assert cached({"data": data}) == len(data)
    assert cached({"data": data}) == len(data)
    assert len(calls) == 1
    assert cached({"data": data[:-1]}) == len(data) - 1
    assert len(calls) == 2


def test_fast_disk_cache_preserves_memmap_configuration(tmp_path, hybrid_cache_clear):
    calls = []

    def twice(data):
        calls.append(1)
        return data * 2

    memory = Memory(tmp_path / "cache", mmap_mode="r", verbose=0)
    cached = hybrid_cache(memory, fast_inaccurate_hashing=True)(twice)
    data = np.arange(4, dtype=np.float32)
    result = cached(data)
    assert isinstance(result, np.memmap)
    np.testing.assert_array_equal(result, [0, 2, 4, 6])
    mapped = np.memmap(tmp_path / "input", shape=(4,), dtype=np.float32, mode="w+")
    mapped[:] = data
    np.testing.assert_array_equal(cached(mapped), result)
    assert len(calls) == 1
