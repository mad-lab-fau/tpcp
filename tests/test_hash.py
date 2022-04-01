import joblib
import torch
from torch import nn

from tpcp._hash import custom_hash


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(1024, 1)

    def forward(self, x):
        return self.readout(x)


def test_hash_model():
    first = custom_hash(TorchModel())
    second = custom_hash(TorchModel())

    assert first == second

    # We also create a negative test, to see that our test dataopbject actually triggers the pytorch problem
    first = joblib.hash(TorchModel())
    second = joblib.hash(TorchModel())

    assert first != second


def test_hash_tensor():
    # This should work even without our custom implementation, but let's be safe and test it.
    data = [0, 1, 2]

    first = custom_hash(torch.tensor(data))
    second = custom_hash(torch.tensor(data))

    assert first == second
