import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal


def compare_algo_objects(a, b):
    parameters = a._get_params_without_nested_class()
    b_parameters = b._get_params_without_nested_class()

    assert set(parameters.keys()) == set(b_parameters.keys())

    for p, value in parameters.items():
        json_val = b_parameters[p]
        if isinstance(value, np.ndarray):
            assert_array_equal(value, json_val)
        elif isinstance(value, (tuple, list)):
            assert list(value) == list(json_val)
        elif isinstance(value, pd.DataFrame):
            assert_frame_equal(value, json_val, check_dtype=False)
        elif isinstance(value, pd.Series):
            assert_series_equal(value, json_val)
        else:
            assert value == json_val, p
