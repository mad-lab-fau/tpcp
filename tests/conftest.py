import warnings
from contextlib import contextmanager
from typing import Any

import pytest

from tpcp import BaseTpcpObject
from tpcp._base import _BaseTpcpObject


def _get_params_without_nested_class(instance: BaseTpcpObject) -> dict[str, Any]:
    return {k: v for k, v in instance.get_params().items() if not isinstance(v, _BaseTpcpObject)}


@contextmanager
def warns_or_none(expected_warning):
    if expected_warning is None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield caught
    else:
        with pytest.warns(expected_warning) as caught:
            yield caught
