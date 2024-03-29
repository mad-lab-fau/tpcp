from dataclasses import fields, make_dataclass

import pytest

from tpcp.misc import TypedIterator


def test_error_wrong_type():
    with pytest.raises(TypeError):
        next(TypedIterator(int).iterate([1, 2, 3]))


def test_simple_no_agg():
    rt = make_dataclass("ResultType", ["result_1", "result_2", "result_3"])

    iterator = TypedIterator(rt)

    data = [1, 2, 3]

    check_data = []
    check_results = []
    for i, (d, r) in enumerate(iterator.iterate(data)):
        assert all(getattr(r, f.name) == iterator.NULL_VALUE for f in fields(rt))
        assert isinstance(r, rt)
        r.result_1 = i
        r.result_2 = i * 2
        r.result_3 = i * 3
        check_data.append(d)
        check_results.append(r)

    assert check_data == data == iterator.inputs_
    assert check_results == iterator.raw_results_
    assert iterator.result_1_ == [0, 1, 2]
    assert iterator.result_2_ == [0, 2, 4]
    assert iterator.result_3_ == [0, 3, 6]


def test_simple_with_agg():
    rt = make_dataclass("ResultType", ["result_1", "result_2", "result_3"])

    iterator = TypedIterator[rt](
        rt, aggregations=[("result_1", lambda i, r: sum(i)), ("result_2", lambda i, r: sum(r))]
    )

    data = [1, 2, 3]
    for i, r in iterator.iterate(data):
        r.result_1 = i - 1
        r.result_2 = i * 2
        r.result_3 = i * 3

    result_obj = iterator.results_

    assert isinstance(result_obj, rt)
    assert iterator.result_1_ == result_obj.result_1 == 6
    assert iterator.result_2_ == result_obj.result_2 == 12
    assert iterator.result_3_ == result_obj.result_3 == [3, 6, 9]


def test_warning_incomplete_iterate():
    rt = make_dataclass("ResultType", ["result_1", "result_2", "result_3"])
    iterator = TypedIterator(rt)
    data = [1, 2, 3]

    next(iterator.iterate(data))
    with pytest.warns(UserWarning):
        partial_results = iterator.raw_results_

    assert partial_results == [
        rt(result_1=TypedIterator.NULL_VALUE, result_2=TypedIterator.NULL_VALUE, result_3=TypedIterator.NULL_VALUE)
    ]

    with pytest.warns(UserWarning):
        partial_results = iterator.result_1_

    assert partial_results == [TypedIterator.NULL_VALUE]


def test_invalid_attr_error():
    field_names = ["result_1", "result_2", "result_3"]
    rt = make_dataclass("ResultType", field_names)
    iterator = TypedIterator(rt)
    data = [1, 2, 3]

    [next(iterator.iterate(data)) for _ in range(3)]

    with pytest.raises(AttributeError) as e:
        iterator.invalid_attr_

    assert "invalid_attr_" in str(e.value)
    for f in field_names:
        assert f"{f}_" in str(e.value)


def test_not_allowed_attr_error():
    field_names = ["results"]

    rt = make_dataclass("ResultType", field_names)
    iterator = TypedIterator(rt)
    data = [1, 2, 3]

    with pytest.raises(ValueError):
        [next(iterator.iterate(data)) for _ in range(3)]


def test_agg_with_empty():
    rt = make_dataclass("ResultType", ["result_1", "result_2", "result_3"])

    iterator = TypedIterator[rt](
        rt, aggregations=[("result_1", lambda i, r: sum(i)), ("result_2", lambda i, r: sum(r))]
    )

    data = [1, 2, 3]
    for i, r in iterator.iterate(data):
        r.result_1 = i - 1
        # We Don't set result 2 -> it will remain an empty value and should skip agg
        r.result_3 = i * 3

    result_obj = iterator.results_

    assert isinstance(result_obj, rt)
    assert iterator.result_1_ == result_obj.result_1 == 6
    assert iterator.result_2_ == [iterator.NULL_VALUE, iterator.NULL_VALUE, iterator.NULL_VALUE]
    assert iterator.result_3_ == result_obj.result_3 == [3, 6, 9]
