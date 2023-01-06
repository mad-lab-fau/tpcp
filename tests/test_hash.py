import joblib
import pytest

from tpcp._hash import custom_hash


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
