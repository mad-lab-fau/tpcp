import sys

import pytest

torch = pytest.importorskip("torch")

import joblib
from torch import nn

from tpcp import clone
from tpcp.misc import custom_hash


class TorchModel(nn.Module):
    def __init__(self, n_features=1024):
        super().__init__()
        torch.manual_seed(42)
        self.readout = nn.Linear(n_features, 1)
        torch.nn.init.uniform_(self.readout.weight, -1, 1)

    def forward(self, x):
        return self.readout(x)


@pytest.fixture(params=(TorchModel(), torch.tensor([0, 1, 2])))
def torch_objects(request):
    return request.param


def test_hash_model():
    model = TorchModel()
    first = custom_hash(model)
    second = custom_hash(TorchModel())
    cloned = custom_hash(clone(model))

    assert first == second
    assert first == cloned

    # And we test that two different models are not equal
    assert custom_hash(TorchModel(n_features=1024)) != custom_hash(TorchModel(n_features=1025))


def test_negative_example():
    # For some reason this test started passing in Python 3.11
    # Skipping for 3.11
    if sys.version_info[:2][0] == 3 and sys.version_info[:2][1] >= 11:
        pytest.skip("This test started passing in Python 3.11")
    # We also create a negative test, to see that our test data object actually triggers the pytorch problem
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

    # And the negative test
    assert custom_hash(torch.tensor([0, 1, 3])) != custom_hash(torch.tensor([0, 1, 2]))


@pytest.mark.parametrize("c", [list, tuple])
def test_container_tensor(torch_objects, c):
    tmp = c([torch_objects])
    assert custom_hash(tmp) == custom_hash(clone(tmp))


def test_dict_tensor(torch_objects):
    tmp = {"a": torch_objects}
    assert custom_hash(tmp) == custom_hash(clone(tmp))
