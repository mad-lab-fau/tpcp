import warnings
from functools import partial
from typing import Callable

import joblib
import pytest

from tpcp import Algorithm
from tpcp.caching import global_disk_cache, global_ram_cache, remove_any_cache


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


class ExampleClassMultiAction(Algorithm):
    _action_methods = ["action", "action_2"]


@pytest.fixture()
def example_class():
    yield ExampleClass
    remove_any_cache(ExampleClass)


@pytest.fixture()
def joblib_cache():
    memory = joblib.Memory(location=".cache", verbose=0)
    yield memory
    memory.clear()


class TestGlobalDiskCache:
    cache_method: Callable[[type[Algorithm]], type[Algorithm]]

    @pytest.fixture(autouse=True, params=["disk", "ram"])
    def get_cache_method(self, request, joblib_cache):
        if request.param == "disk":
            self.cache_method = partial(global_disk_cache, joblib_cache)
        else:
            self.cache_method = partial(global_ram_cache, None)

    def test_caching_twice_same_instance(self, example_class):
        self.cache_method()(example_class)
        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            example.action(3)

        assert example.result_1_ == 6

        with pytest.warns(CacheWarning):
            example.action(2)
        assert example.result_1_ == 5

        with pytest.warns(None) as w:
            example.action(3)
        assert example.result_1_ == 6
        assert not w

    def test_caching_twice_new_instance(self, example_class):
        self.cache_method()(example_class)
        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            example.action(3)
        assert example.result_1_ == 6

        with pytest.warns(CacheWarning):
            example.action(2)
        assert example.result_1_ == 5

        example = example_class(1, 2)
        with pytest.warns(None) as w:
            example.action(3)
        assert example.result_1_ == 6
        assert not w

    def test_cache_invalidated_on_para_change(self, example_class):
        self.cache_method()(example_class)

        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            example.action(3)
        assert example.result_1_ == 6

        example.set_params(a=4)

        with pytest.warns(CacheWarning):
            example.action(3)
        assert example.result_1_ == 9

    def test_cache_only(self, example_class):
        self.cache_method(cache_only=["result_1_"])(example_class)

        # We expect the uncached and the cached version to both not have result_2_ available.

        example = example_class(1, 2)
        with pytest.warns(CacheWarning):
            example.action(2)
        assert example.result_1_ == 5
        assert not hasattr(example, "result_2_")

        example = example.clone()

        # Now in the cached version
        with pytest.warns(None) as w:
            example.action(2)
        assert not w
        assert example.result_1_ == 5
        assert not hasattr(example, "result_2_")

    def test_error_on_classes_with_multiple_action_methods(self):
        with pytest.raises(NotImplementedError):
            self.cache_method()(ExampleClassMultiAction)
