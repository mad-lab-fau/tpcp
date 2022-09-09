"""A set of tests for parameters that specifically tests potential issues with different type of type import and uses
the text-annotations future"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typing_extensions import Annotated

from tpcp import BaseTpcpObject, HyperPara, Para
from tpcp._base import _process_tpcp_class
from tpcp._parameters import _ParaTypes


def test_basic_annotation_collection():
    class Test(BaseTpcpObject):
        hyper: HyperPara[int]
        normal: Para[str]
        custom_annotated: Annotated[HyperPara[int], "custom_metadata"]
        normal_no_annot: int

        def __init__(self, hyper: int, normal: str, custom_annotated: int, normal_no_annot: int):
            self.hyper = hyper
            self.normal = normal
            self.custom_annotated = custom_annotated
            self.normal_no_annot = normal_no_annot

    # We manually trigger the second part of the processing of the annotations
    _process_tpcp_class(Test)

    assert Test.__field_annotations__ == {
        "hyper": _ParaTypes.HYPER,
        "normal": _ParaTypes.SIMPLE,
        "custom_annotated": _ParaTypes.HYPER,
        "normal_no_annot": _ParaTypes.SIMPLE,
    }


def test_import_forward():
    if TYPE_CHECKING:
        from tpcp import optimize

        renamed_optimize = optimize.GridSearch

    class Test(BaseTpcpObject):
        hyper: HyperPara[int]
        normal: Para[renamed_optimize]
        custom_annotated: Annotated[HyperPara[int], "custom_metadata"]
        normal_no_annot: int

        def __init__(self, hyper: int, normal: renamed_optimize, custom_annotated: int, normal_no_annot: int):
            self.hyper = hyper
            self.normal = normal
            self.custom_annotated = custom_annotated
            self.normal_no_annot = normal_no_annot

    # We manually trigger the second part of the processing of the annotations
    _process_tpcp_class(Test)

    assert Test.__field_annotations__ == {
        "hyper": _ParaTypes.HYPER,
        "normal": _ParaTypes.SIMPLE,
        "custom_annotated": _ParaTypes.HYPER,
        "normal_no_annot": _ParaTypes.SIMPLE,
    }


def test_import_forward_error():
    if TYPE_CHECKING:
        from tpcp import optimize

    with pytest.raises(RuntimeError) as e:

        class Test(BaseTpcpObject):
            hyper: HyperPara[int]
            normal: Para[optimize.GridSearch]
            custom_annotated: Annotated[HyperPara[int], "custom_metadata"]
            normal_no_annot: int

            def __init__(self, hyper: int, normal: optimize.GridSearch, custom_annotated: int, normal_no_annot: int):
                self.hyper = hyper
                self.normal = normal
                self.custom_annotated = custom_annotated
                self.normal_no_annot = normal_no_annot

    assert "You ran into an edge case" in str(e)


def test_test_str_based_forward():
    class Test(BaseTpcpObject):
        hyper: HyperPara[int]
        normal: Para["Dataset"]
        custom_annotated: Annotated[HyperPara[int], "custom_metadata"]
        normal_no_annot: int

        def __init__(self, hyper: int, normal: "Dataset", custom_annotated: int, normal_no_annot: int):
            self.hyper = hyper
            self.normal = normal
            self.custom_annotated = custom_annotated
            self.normal_no_annot = normal_no_annot

    # We manually trigger the processing of the annotations
    _process_tpcp_class(Test)

    assert Test.__field_annotations__ == {
        "hyper": _ParaTypes.HYPER,
        "normal": _ParaTypes.SIMPLE,
        "custom_annotated": _ParaTypes.HYPER,
        "normal_no_annot": _ParaTypes.SIMPLE,
    }
