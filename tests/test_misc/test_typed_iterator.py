import re
import warnings
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

    inputs = [r.input for r in iterator.raw_results_]
    assert check_data == data == inputs
    raw_output = [r.result for r in iterator.raw_results_]
    assert check_results == raw_output
    assert iterator.results_.result_1 == [0, 1, 2]
    assert iterator.results_.result_2 == [0, 2, 4]
    assert iterator.results_.result_3 == [0, 3, 6]


def test_simple_with_agg():
    rt = make_dataclass("ResultType", ["result_1", "result_2", "result_3"])

    iterator = TypedIterator(
        rt,
        aggregations=[
            ("result_1", lambda re: sum(r.input for r in re)),
            ("result_2", lambda re: sum(r.result.result_2 for r in re)),
        ],
    )

    data = [1, 2, 3]
    for i, r in iterator.iterate(data):
        r.result_1 = i - 1
        r.result_2 = i * 2
        r.result_3 = i * 3

    result_obj = iterator.results_

    assert isinstance(result_obj, rt)
    assert iterator.results_.result_1 == 6
    assert iterator.results_.result_2 == 12
    assert iterator.results_.result_3 == [3, 6, 9]


def test_warning_incomplete_iterate():
    rt = make_dataclass("ResultType", ["result_1", "result_2", "result_3"])
    iterator = TypedIterator(rt)
    data = [1, 2, 3]

    next(iterator.iterate(data))
    with pytest.warns(UserWarning):
        partial_results = [r.result for r in iterator.raw_results_]

    assert partial_results == [
        rt(result_1=TypedIterator.NULL_VALUE, result_2=TypedIterator.NULL_VALUE, result_3=TypedIterator.NULL_VALUE)
    ]

    with pytest.warns(UserWarning):
        partial_results = iterator.results_.result_1

    assert partial_results == [TypedIterator.NULL_VALUE]


def test_iterator_body_warning_contains_iteration_context():
    rt = make_dataclass("ResultType", ["result"])
    iterator = TypedIterator(rt)

    def emit_iteration_warning():
        for data_point, result in iterator.iterate([1]):
            with iterator.warning_error_context(
                "typed_iterator_iteration",
                iteration_name="__main__",
                input=data_point,
                iteration_context={},
            ):
                warnings.warn("iteration warning", UserWarning, stacklevel=1)
                result.result = 1

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "[typed_iterator_iteration: iteration_name='__main__', input=1, iteration_context={}] iteration warning"
        ),
    ):
        emit_iteration_warning()


def test_iterator_body_exception_contains_explicit_context():
    """The explicit iterator context annotates errors raised by the loop body."""
    rt = make_dataclass("ResultType", ["result"])
    iterator = TypedIterator(rt)

    def raise_iteration_error():
        for data_point, _ in iterator.iterate([1]):
            with iterator.warning_error_context("typed_iterator", input=data_point, stage="processing"):
                raise ValueError("iteration error")

    with pytest.raises(ValueError, match="iteration error") as error:
        raise_iteration_error()

    assert error.value.__notes__ == ["Context: typed_iterator: input=1, stage='processing'"]


def test_additional_aggregations():
    rt = make_dataclass("ResultType", ["result_1", "result_2", "result_3"])

    iterator = TypedIterator(
        rt,
        aggregations=[
            ("result_1", lambda re: sum(r.input for r in re)),
            ("result_2", lambda re: sum(r.result.result_2 for r in re)),
            ("additional_results", lambda re: sum(r.result.result_2 + 1 for r in re)),
        ],
    )

    data = [1, 2, 3]
    for i, r in iterator.iterate(data):
        r.result_1 = i - 1
        r.result_2 = i * 2
        r.result_3 = i * 3

    result_obj = iterator.results_

    assert isinstance(result_obj, rt)
    assert iterator.results_.result_1 == 6
    assert iterator.results_.result_2 == 12
    assert iterator.results_.result_3 == [3, 6, 9]

    assert iterator.additional_results_["additional_results"] == iterator.results_.result_2 + len(data)
