import pytest

torch = pytest.importorskip("torch")

import joblib
from torch import nn

from tpcp import clone
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
    model = TorchModel()
    first = custom_hash(model)
    second = custom_hash(TorchModel())
    cloned = custom_hash(clone(model))

    assert first == second
    assert first == cloned

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
