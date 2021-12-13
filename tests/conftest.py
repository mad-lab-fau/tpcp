from typing import Any, Dict

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from tpcp import BaseTpcpObject
from tpcp._base import _BaseTpcpObject


def _get_params_without_nested_class(instance: BaseTpcpObject) -> Dict[str, Any]:
    return {k: v for k, v in instance.get_params().items() if not isinstance(v, _BaseTpcpObject)}
