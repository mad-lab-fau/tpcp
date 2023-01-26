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

    # We need to run the init once to trigger the processing of the annotations
    test = Test(1, "", 1, 1)

    assert test.__field_annotations__ == {
        "hyper": _ParaTypes.HYPER,
        "normal": _ParaTypes.SIMPLE,
        "custom_annotated": _ParaTypes.HYPER,
        "normal_no_annot": _ParaTypes.SIMPLE,
    }


def test_annotation_not_in_init():
    class Test(BaseTpcpObject):
        not_in_init: Para[int]

    with pytest.raises(ValueError) as e:
        Test().__field_annotations__

    assert "The field 'not_in_init' of Test was annotated" in str(e)


def test_annotations_nested():
    class Test(BaseTpcpObject):
        hyper: HyperPara[int]
        nested_hyper__nested_para: HyperPara[str]

        def __init__(self, hyper: int):
            self.hyper = hyper

    test = Test(1)

    assert test.__field_annotations__ == {"hyper": _ParaTypes.HYPER, "nested_hyper__nested_para": _ParaTypes.HYPER}


def test_field_annotations_cache():
    class Test(BaseTpcpObject):
        hyper: HyperPara[int]
        nested_hyper__nested_para: HyperPara[str]

        def __init__(self, hyper: int):
            self.hyper = hyper

    test = Test(1)
    annots = test.__field_annotations__
    cache = Test.__field_annotations_cache__
    cache_id = id(cache)
    assert annots == cache == {"hyper": _ParaTypes.HYPER, "nested_hyper__nested_para": _ParaTypes.HYPER}
    # When called again, the cache shouldn't change
    Test(2)
    annots = test.__field_annotations__
    assert id(Test.__field_annotations_cache__) == cache_id
    assert id(annots) == cache_id


def test_annotation_inherting():
    class Test(BaseTpcpObject):
        hyper: HyperPara[int]
        nested_hyper__nested_para: HyperPara[str]

        def __init__(self, hyper: int):
            self.hyper = hyper

    class Test2(Test):
        other_val: HyperPara[float]

        def __init__(self, hyper: int, other_val: float):
            self.other_val = other_val
            super().__init__(hyper)

    assert Test2(1, 2).__field_annotations__ == {
        "hyper": _ParaTypes.HYPER,
        "nested_hyper__nested_para": _ParaTypes.HYPER,
        "other_val": _ParaTypes.HYPER,
    }

    assert Test(1).__field_annotations__ == {
        "hyper": _ParaTypes.HYPER,
        "nested_hyper__nested_para": _ParaTypes.HYPER,
    }
