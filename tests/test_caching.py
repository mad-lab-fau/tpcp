import warnings
from functools import partial
from typing import Callable

import joblib
import pytest
from joblib import Memory

from tpcp import Algorithm
from tpcp.caching import global_disk_cache, global_ram_cache, hybrid_cache, remove_any_cache


class CacheWarning(UserWarning):
    pass


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


@pytest.fixture()
def joblib_cache():
    memory = joblib.Memory(location=".cache", verbose=0)
    yield memory
    memory.clear()


@pytest.fixture()
def joblib_cache_verbose():
    memory = joblib.Memory(location=".cache", verbose=10)
    yield memory
    memory.clear()


@pytest.fixture()
def hybrid_cache_clear():
    yield None
    hybrid_cache.__cache_registry__.clear()


class TestGlobalCache:
    cache_method: Callable[[type[Algorithm]], type[Algorithm]]

    @pytest.fixture(autouse=True, params=["disk", "ram"])
    def get_cache_method(self, request, joblib_cache):
        if request.param == "disk":
            self.cache_method = partial(global_disk_cache, joblib_cache)
        else:
            self.cache_method = partial(global_ram_cache, None)

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

        with pytest.warns(None) as w:
            getattr(example, action_name)(3)
        assert example.result_1_ == 6 * multiplier
        assert not w

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
        with pytest.warns(None) as w:
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
        with pytest.warns(None) as w:
            getattr(example, action_name)(2)
        assert not w
        assert example.result_1_ == 5 * multiplier
        assert not hasattr(example, "result_2_")


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

        with pytest.warns(None) as w:
            r = cached_func(1, 2)

        assert r == 3
        assert not w

    def test_lru_only(self, hybrid_cache_clear):
        cached_func = hybrid_cache(Memory(None), 2)(example_func)

        with pytest.warns(CacheWarning):
            r = cached_func(1, 2)

        assert r == 3

        with pytest.warns(None) as w:
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

        with pytest.warns(None) as w:
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

        with pytest.warns(None) as w:
            r = cached_func_new(1, 2)

        # This time this should hit the joblib cache, as the lru cache should have been cleared
        out = capfd.readouterr()
        clean_out = out.out.replace("\n", "")

        assert "Loading example_func from" in clean_out

        assert r == 3
        assert not w

        # And now the lru cache should be used again
        with pytest.warns(None) as w:
            r = cached_func_new(1, 2)

        # This time this should hit the joblib cache, as the lru cache should have been cleared
        out = capfd.readouterr()
        clean_out = out.out.replace("\n", "")

        assert clean_out == ""

        assert r == 3
        assert not w
