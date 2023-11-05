"""Helper for testing of algorithms and pipelines implemented in tpcp."""
import pytest

pytest.register_assert_rewrite("tpcp.testing._algorithm_test_mixin")
pytest.register_assert_rewrite("tpcp.testing._regression_utils")

from tpcp.testing._algorithm_test_mixin import TestAlgorithmMixin  # noqa: E402
from tpcp.testing._regression_utils import PyTestSnapshotTest  # noqa: E402

__all__ = ["TestAlgorithmMixin", "PyTestSnapshotTest"]
