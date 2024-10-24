from unittest.mock import Mock

import numpy as np
import pytest

from tests.test_pipelines.conftest import DummyDataset, DummyOptimizablePipeline
from tpcp._utils._score import _score


class TestScoreMock:
    @pytest.fixture(autouse=True)
    def create_paras(self):
        dataset = DummyDataset()
        # We mock the scorer and just return the dataset itself and 1 as agg score
        scorer = Mock(return_value=(1, np.asarray(dataset.group_labels)))
        pipe = DummyOptimizablePipeline()

        self.dummy_paras = (pipe, dataset, scorer)

    @pytest.mark.parametrize(
        ("enable_output", "out_para"),
        [
            ("return_parameters", "parameters"),
            ("return_data_labels", "data_labels"),
            ("return_times", "debug__score_time"),
        ],
    )
    def test_result_params(self, enable_output, out_para):
        result = _score(*self.dummy_paras, parameters=None, **{enable_output: True})
        assert out_para in result

    def test_parameter_set(self):
        """Test that the parameters are set before score is called."""
        paras = {"para_1": "para_1_val", "para_2": "para_2_val"}
        _score(*self.dummy_paras, parameters=paras)
        scorer = self.dummy_paras[2]

        out_paras = scorer.call_args[0][0].get_params()
        assert {k: out_paras[k] for k in paras} == paras
        assert id(scorer.call_args[0][0]) == id(self.dummy_paras[0])

    def test_parameter_clone(self):
        nested_obj = DummyOptimizablePipeline()
        paras = {"para_1": "para_1_val", "para_2": nested_obj}
        _score(*self.dummy_paras, parameters=paras)
        scorer = self.dummy_paras[2]

        nested_obj_after_set = scorer.call_args[0][0].get_params()["para_2"]
        assert nested_obj_after_set.get_params() == nested_obj.get_params()
        assert id(nested_obj_after_set) != id(nested_obj)
