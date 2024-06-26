from typing import Any

import pytest

from tpcp import BaseTpcpObject
from tpcp._base import _BaseTpcpObject


def _get_params_without_nested_class(instance: BaseTpcpObject) -> dict[str, Any]:
    return {k: v for k, v in instance.get_params().items() if not isinstance(v, _BaseTpcpObject)}


@pytest.fixture()
def snapshot(request):
    from tpcp.testing import PyTestSnapshotTest

    with PyTestSnapshotTest(request) as snapshot_test:
        yield snapshot_test


def pytest_addoption(parser):
    group = parser.getgroup("tpcp_snapshots")
    group.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        dest="snapshot_update",
        help="Update the snapshots.",
    )
    group.addoption(
        "--snapshot-only-check",
        action="store_true",
        default=False,
        dest="snapshot_only_check",
        help="Run as normal, but fail if a snapshot file is not found. This is usefull for CI runs.",
    )