import functools

import joblib
import numpy as np
import pytest

from tpcp import BaseTpcpObject
from tpcp._hash import custom_hash
from tpcp.validate import FloatAggregator


@pytest.fixture
def joblib_cache():
    memory = joblib.Memory(location=".cache", verbose=0)
    yield memory
    memory.clear()


def test_memoize_bug():
    # We test that the memoize bug (https://github.com/joblib/joblib/issues/1283) does not occur with our hasher.

    val = ["test"]
    val2 = ["test"]

    assert custom_hash([{"a": val}, val]) == custom_hash([{"a": val2}, val])

    # We also do a negative test
    assert joblib.hash([{"a": val}, val]) != joblib.hash([{"a": val2}, val])


def test_error_message_recursive_objects():
    rec_obj = {}
    rec_obj["rec"] = rec_obj

    with pytest.raises(ValueError) as e:
        custom_hash(rec_obj)

    assert "The custom hasher used in tpcp does not support hashing" in str(e.value)


def test_hash_nested_object():
    class Class1(BaseTpcpObject):
        def __init__(self, other):
            self.other = other

    class Class2(BaseTpcpObject):
        def __init__(self, val):
            self.val = val

    obj1 = Class1(Class2(1))
    obj2 = Class1(Class2(1))

    assert custom_hash(obj1) == custom_hash(obj2) != custom_hash(Class1(Class2(2)))


def test_hash_nested_object_multiprocessing():
    def get_aggregator():
        def func(a):
            return np.mean(a)

        return FloatAggregator(func)

    outside = custom_hash(get_aggregator())

    assert custom_hash(get_aggregator()) == outside

    # We also test that the hash is the same when using multiprocessing
    def worker_func():
        return custom_hash(get_aggregator())

    assert joblib.Parallel(n_jobs=2)(joblib.delayed(worker_func)() for _ in range(2)) == [outside, outside]


def test_hash_nested_actually_different():
    def get_aggregator():
        def func(a):
            return np.mean(a)

        return FloatAggregator(func)

    def get_aggregator2():
        def func(a):
            return np.median(a)

        return FloatAggregator(func)

    assert custom_hash(get_aggregator()) != custom_hash(get_aggregator2())


def test_hash_nested_wrapped_different():
    def func(a):
        return np.mean(a)

    obj1 = FloatAggregator(func)

    def decorator(func):
        @functools.wraps(func)
        def _func(a):
            return func(a)

        return _func

    obj2 = FloatAggregator(decorator(func))

    assert custom_hash(obj1) != custom_hash(obj2)


def test_hash_lambdas_same():
    def func(a, b):
        return np.mean(a) + b

    def func2():
        return FloatAggregator(lambda a: func(a, 1))

    obj1 = func2()
    obj2 = func2()

    assert custom_hash(obj1) == custom_hash(obj2)


def test_hash_lambdas_different():
    # This is quite interesting, these two lambdas are different, as they have different names, as they are
    # defined in the same scope. in the pevious test, where there was only on lambda defined, the names were the same
    # hence the hash the same.
    obj1 = FloatAggregator(lambda a: np.mean(a))
    obj2 = obj1
    obj1 = FloatAggregator(lambda a: np.mean(a))
    assert custom_hash(obj1) != custom_hash(obj2)


def test_hash_partials_same():
    def func(a, b):
        return np.mean(a) + b

    obj1 = FloatAggregator(functools.partial(func, b=1))
    obj2 = FloatAggregator(functools.partial(func, b=1))

    assert custom_hash(obj1) == custom_hash(obj2)


def test_hash_partials_different():
    def func(a, b):
        return np.mean(a) + b

    obj1 = FloatAggregator(functools.partial(func, b=1))
    obj2 = FloatAggregator(functools.partial(func, b=2))

    assert custom_hash(obj1) != custom_hash(obj2)


def test_hash_partials_different2():
    def func(a, b):
        return np.mean(a) + b

    def func2(a, b):
        return np.mean(a) + b

    obj1 = FloatAggregator(functools.partial(func, b=1))
    obj2 = FloatAggregator(functools.partial(func2, b=1))

    assert custom_hash(obj1) != custom_hash(obj2)
