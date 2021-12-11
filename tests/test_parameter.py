import pytest
from typing_extensions import Annotated

from tpcp import BaseTpcpObject, HyperPara, Para
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

    assert Test.__field_annotations__ == {
        "hyper": _ParaTypes.HYPER,
        "normal": _ParaTypes.SIMPLE,
        "custom_annotated": _ParaTypes.HYPER,
        "normal_no_annot": _ParaTypes.SIMPLE,
    }


def test_annotation_not_in_init():

    with pytest.raises(ValueError) as e:

        class Test(BaseTpcpObject):
            not_in_init: Para[int]

    assert "The field 'not_in_init' of Test was annotated" in str(e)


def test_annotations_nested():
    class Test(BaseTpcpObject):
        hyper: HyperPara[int]
        nested_hyper__nested_para: HyperPara[str]

        def __init__(self, hyper: int):
            self.hyper = hyper

    assert Test.__field_annotations__ == {"hyper": _ParaTypes.HYPER, "nested_hyper__nested_para": _ParaTypes.HYPER}
