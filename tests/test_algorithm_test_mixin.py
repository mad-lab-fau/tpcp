"""Regression tests for the common algorithm test mixin."""

import pytest

from tpcp import Algorithm, make_action_safe
from tpcp.testing import TestAlgorithmMixin


class DecoratedAlgorithm(Algorithm):
    """Algorithm with a decorated action method for mixin tests."""

    _action_methods = ("run",)

    def __init__(self):
        pass

    @make_action_safe
    def run(self, data: int):
        """Store the input as a result."""
        self.data = data
        self.result_ = data
        return self


class TestDecoratedAlgorithm(TestAlgorithmMixin):
    """Check that the test mixin accepts decorated action methods."""

    ALGORITHM_CLASS = DecoratedAlgorithm
    CHECK_DOCSTRING = False
    __test__ = True

    @pytest.fixture
    def after_action_instance(self):
        """Return an algorithm after its decorated action method runs."""
        return DecoratedAlgorithm().run(1)
