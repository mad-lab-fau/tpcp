from typing import Any

from tpcp import BaseTpcpObject
from tpcp._base import _BaseTpcpObject


def _get_params_without_nested_class(instance: BaseTpcpObject) -> dict[str, Any]:
    return {k: v for k, v in instance.get_params().items() if not isinstance(v, _BaseTpcpObject)}
