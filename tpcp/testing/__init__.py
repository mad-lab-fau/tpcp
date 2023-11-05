"""Helper for testing of algorithms and pipelines implemented in tpcp."""
import pytest

from tpcp.testing._algorithm_test_mixin import TestAlgorithmMixin
from tpcp.testing._regression_utils import PyTestSnapshotTest

pytest.register_assert_rewrite("tpcp.testing._algorithm_test_mixin")
pytest.register_assert_rewrite("tpcp.testing._regression_utils")

__all__ = ["TestAlgorithmMixin", "PyTestSnapshotTest"]
