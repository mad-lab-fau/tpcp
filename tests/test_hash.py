import joblib
import numpy as np
import pytest
import torch
from torch import nn

from tpcp import Algorithm, clone
from tpcp._hash import custom_hash


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(1024, 1)

    def forward(self, x):
        return self.readout(x)


@pytest.fixture(params=(TorchModel(), torch.tensor([0, 1, 2])))
def torch_objects(request):
    return request.param


def test_hash_model():
    first = custom_hash(TorchModel())
    second = custom_hash(TorchModel())

    assert first == second

    # We also create a negative test, to see that our test dataopbject actually triggers the pytorch problem
    first = joblib.hash(TorchModel())
    second = joblib.hash(TorchModel())

    assert first != second


def test_hash_tensor():
    data = [0, 1, 2]
    tensor = torch.tensor(data)

    first = custom_hash(tensor)
    cloned = custom_hash(clone(tensor))
    second = custom_hash(torch.tensor(data))

    assert first == second
    assert first == cloned


@pytest.mark.parametrize("c", (list, tuple))
def test_container_tensor(torch_objects, c):

    tmp = c([torch_objects])
    assert custom_hash(tmp) == custom_hash(clone(tmp))


def test_dict_tensor(torch_objects):

    tmp = {"a": torch_objects}
    assert custom_hash(tmp) == custom_hash(clone(tmp))


def test_memoize_bug():
    # We test that the memoize bug (https://github.com/joblib/joblib/issues/1283) does not occure with our hasher.

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
