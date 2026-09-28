import functools

import joblib
import numpy as np
import pandas as pd
import pytest

from tpcp import BaseTpcpObject
from tpcp.misc import custom_hash
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


def test_default_hash_uses_fast_mode_with_explicit_legacy_escape_hatch():
    values = np.arange(12, dtype=np.float32)
    assert custom_hash(values) != custom_hash(values, hash_name="md5")
    # Recorded with the pre-change TPCP hasher. A scalar avoids NumPy/Python
    # version differences in array serialization while checking legacy digests.
    assert custom_hash(42, hash_name="md5") == "d922f805b5eead8c40ee21f14329d6c7"
    assert custom_hash(42, hash_name="sha1") == "c42ff5cf22ebccc4d4cc538db7af4e97f7c7d7e7"


@pytest.mark.parametrize(
    "change", ["value", "shape", "dtype", "columns", "index", "index_name", "column_name", "attrs"]
)
def test_numeric_dataframe_hash_detects_changes(change):
    original = pd.DataFrame(np.arange(12, dtype=np.float32).reshape(4, 3), columns=list("abc"))
    original_hash = custom_hash({"frame": original})
    changed = original
    if change == "value":
        changed.iloc[-1, -1] += 1
    elif change == "shape":
        changed = changed.iloc[:-1]
    elif change == "dtype":
        changed = changed.astype(np.float64)
    elif change == "columns":
        changed.columns = list("abd")
    elif change == "index":
        changed.index = [1, 2, 3, 4]
    elif change == "index_name":
        changed.index.name = "time"
    elif change == "column_name":
        changed.columns.name = "sensor"
    else:
        changed.attrs["unit"] = "g"
    assert original_hash != custom_hash({"frame": changed})


@pytest.mark.parametrize(
    "values",
    [
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.asfortranarray(np.arange(12).reshape(4, 3)),
        np.arange(24)[::2],
        np.array(["a", "b"], dtype=object),
        np.array(42),
        np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")]),
    ],
)
def test_array_hash_is_repeatable_and_detects_mutation(values):
    original_hash = custom_hash({"values": values})
    assert custom_hash({"values": values}) == original_hash
    values.flat[-1] = 0
    assert custom_hash({"values": values}) != original_hash


def test_array_hash_includes_shape_and_dtype():
    values = np.arange(12, dtype=np.int32)
    assert len({custom_hash(values), custom_hash(values.reshape(4, 3)), custom_hash(values.view(np.float32))}) == 3


def test_memmap_coercion(tmp_path):
    values = np.memmap(tmp_path / "array", dtype=np.float32, shape=(4,), mode="w+")
    values[:] = [1, 2, 3, 4]
    plain = np.asarray(values)
    assert custom_hash(values) != custom_hash(plain)
    assert custom_hash(values, coerce_mmap=True) == custom_hash(plain, coerce_mmap=True)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": ["one", "two"]}, dtype=object),
        pd.DataFrame({"a": pd.Series([1, None], dtype="Int64")}),
        pd.DataFrame({"a": pd.Categorical(["one", "two"])}),
        pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]}),
        pd.DataFrame(),
        pd.DataFrame({"a": pd.Series([], dtype="float32")}),
    ],
)
def test_dataframe_hash_is_stable_and_detects_metadata_change(frame):
    original_hash = custom_hash(frame)
    assert custom_hash(frame) == original_hash
    frame.attrs["unit"] = "g"
    assert custom_hash(frame) != original_hash
