"""Behavioral tests for dataset-label splitter composition."""

from collections.abc import Iterator

import pytest
from sklearn.model_selection import KFold

from tests.test_pipelines.conftest import DummyDataset, DummyGroupedDataset, DummyOptimizablePipeline
from tpcp import Dataset, clone
from tpcp.optimize import GridSearchCV, Optimize
from tpcp.validate import BaseDatasetSplitter, CombinedSplitter, DatasetSplitter, NoSplit, SplitterPart, cross_validate


def _score(_pipeline, data_point):
    return 1 if data_point.group_label[0] is not None else 0


def test_combined_named_part_parameters_change_folds_without_changing_other_parts():
    dataset = DummyGroupedDataset()
    fixed = SplitterPart(lambda ds: ds.get_subset(v1="c"), NoSplit(None, test=lambda ds: ds))
    splitter = CombinedSplitter(
        parts=[
            ("training", SplitterPart(lambda ds: ds.get_subset(v1="a"), NoSplit(2, train=lambda ds: ds))),
            ("fixed", fixed),
        ]
    )
    assert splitter.get_params()["parts__training__splitter__n_splits"] == 2
    replacement = SplitterPart(lambda ds: ds.get_subset(v1="b"), NoSplit(3, train=lambda ds: ds))
    splitter.set_params(parts__training=replacement)
    assert list(splitter.split(dataset)) == [([("b", i) for i in range(5)], [("c", i) for i in range(5)])] * 3
    splitter.set_params(
        parts__training__splitter__n_splits=1, parts__training__selector=lambda ds: ds.get_subset(v1="a")
    )
    assert list(splitter.split(dataset)) == [([("a", i) for i in range(5)], [("c", i) for i in range(5)])]
    assert splitter.parts[1][1] is fixed


def test_no_split_repeats_labels_and_selects_once():
    dataset = DummyDataset()[[3, 1, 4, 0, 2]]
    calls = []

    def train(ds):
        calls.append("train")
        return ds.get_subset(group_labels=[ds.group_labels[1], ds.group_labels[0]])

    def test(ds):
        calls.append("test")
        return ds.get_subset(group_labels=[ds.group_labels[2]])

    splitter = NoSplit(3, train=train, test=test)
    expected = ([(1,), (3,)], [(4,)])
    assert splitter.get_n_splits(dataset) == 3
    assert list(splitter.split(dataset)) == [expected] * 3
    assert calls == ["train", "test"]


def test_split_can_request_fewer_folds_or_bind_an_unbounded_splitter():
    dataset = DummyDataset()
    assert len(list(DatasetSplitter(5).split(dataset, n_splits=2))) == 2
    assert list(NoSplit(3).split(dataset, n_splits=2)) == [([], [])] * 2
    assert list(NoSplit(None).split(dataset, n_splits=2)) == [([], [])] * 2
    with pytest.raises(ValueError, match="exceeds"):
        list(DatasetSplitter(5).split(dataset, n_splits=6))
    with pytest.raises(ValueError, match="exceeds"):
        list(NoSplit(3).split(dataset, n_splits=4))


def test_no_split_omitted_sides_and_overlap():
    dataset = DummyDataset()
    assert list(NoSplit(2).split(dataset)) == [([], [])] * 2
    assert list(NoSplit(2, train=lambda ds: ds).split(dataset)) == [(dataset.group_labels, [])] * 2
    with pytest.raises(ValueError, match="overlap"):
        list(NoSplit(1, train=lambda ds: ds, test=lambda ds: ds).split(dataset))


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_no_split_requires_positive_integer_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        NoSplit(count).get_n_splits(DummyDataset())


def test_combined_splitter_uses_known_count_for_no_split_parts():
    dataset = DummyGroupedDataset()
    splitter = CombinedSplitter(
        parts=[
            ("part_0", SplitterPart(lambda ds: ds.get_subset(v1="a"), DatasetSplitter(5))),
            ("part_1", SplitterPart(lambda ds: ds.get_subset(v1="b"), NoSplit(None, train=lambda ds: ds))),
            ("part_2", SplitterPart(lambda ds: ds.get_subset(v1="c"), NoSplit(None, test=lambda ds: ds))),
        ]
    )
    assert NoSplit(None).get_n_splits(dataset) is None
    assert splitter.get_n_splits(dataset) == 5
    folds = list(splitter.split(dataset))
    assert len(folds) == 5
    assert folds[0] == (
        [("a", i) for i in range(1, 5)] + [("b", i) for i in range(5)],
        [("a", 0)] + [("c", i) for i in range(5)],
    )
    assert splitter.parts[1][1].splitter.n_splits is None
    assert list(splitter.split(dataset, n_splits=2)) == folds[:2]
    with pytest.raises(ValueError, match="exceeds"):
        list(splitter.split(dataset, n_splits=6))


def test_combined_splitter_requires_a_known_fold_count():
    dataset = DummyDataset()
    splitter = CombinedSplitter(parts=[("part_0", SplitterPart(lambda ds: ds, NoSplit(None)))])
    with pytest.raises(ValueError, match=r"at least one.*number of folds"):
        splitter.get_n_splits(dataset)
    with pytest.raises(ValueError, match=r"at least one.*number of folds"):
        list(splitter.split(dataset))
    with pytest.raises(ValueError, match="requires a fold count"):
        list(NoSplit(None).split(dataset))


class _UnboundedSplitter(BaseDatasetSplitter):
    def get_n_splits(self, dataset: Dataset) -> None:
        return None

    def split(self, dataset: Dataset, n_splits: int | None = None) -> Iterator:
        if n_splits is None:
            raise ValueError("A fold count is required.")
        for _ in range(n_splits):
            yield dataset.group_labels, []


def test_combined_splitter_passes_count_to_any_unbounded_child():
    dataset = DummyDataset()
    splitter = CombinedSplitter(
        parts=[
            ("part_0", SplitterPart(lambda ds: ds, NoSplit(3))),
            ("part_1", SplitterPart(lambda ds: ds, _UnboundedSplitter())),
        ]
    )
    assert list(splitter.split(dataset, n_splits=2)) == [(dataset.group_labels, [])] * 2


def test_combined_splitter_rejects_extra_folds_from_unbounded_child():
    class ExtraFoldSplitter(_UnboundedSplitter):
        def split(self, dataset: Dataset, n_splits: int | None = None) -> Iterator:
            yield from super().split(dataset, n_splits=n_splits)
            yield dataset.group_labels, []

    dataset = DummyDataset()
    splitter = CombinedSplitter(
        parts=[
            ("part_0", SplitterPart(lambda ds: ds, NoSplit(3))),
            ("part_1", SplitterPart(lambda ds: ds, ExtraFoldSplitter())),
        ]
    )
    with pytest.raises(ValueError, match="more folds"):
        list(splitter.split(dataset, n_splits=2))


def test_combined_splitter_pairs_selected_folds_and_excludes_other_groups():
    dataset = DummyGroupedDataset()
    splitter = CombinedSplitter(
        parts=[
            ("part_0", SplitterPart(lambda ds: ds.get_subset(v1="a"), KFold(5))),
            ("part_1", SplitterPart(lambda ds: ds.get_subset(v1="b"), DatasetSplitter(5))),
        ]
    )
    folds = list(splitter.split(dataset))
    assert splitter.get_n_splits(dataset) == 5
    assert len(folds) == 5
    assert folds[0][0] == [("a", i) for i in range(1, 5)] + [("b", i) for i in range(1, 5)]
    assert folds[0][1] == [("a", 0), ("b", 0)]
    assert all("c" not in label for fold in folds for side in fold for label in side)


def test_combined_splitter_deduplicates_ordered_labels_and_detects_overlap():
    dataset = DummyDataset()
    splitter = CombinedSplitter(
        parts=[("part_0", SplitterPart(lambda ds: ds, KFold(5))), ("part_1", SplitterPart(lambda ds: ds, KFold(5)))]
    )
    train, test = next(splitter.split(dataset))
    assert train == dataset.group_labels[1:]
    assert test == dataset.group_labels[:1]
    with pytest.raises(ValueError, match="overlap"):
        list(
            CombinedSplitter(
                parts=[
                    ("part_0", SplitterPart(lambda ds: ds, NoSplit(1, train=lambda ds: ds))),
                    ("part_1", SplitterPart(lambda ds: ds, NoSplit(1, test=lambda ds: ds))),
                ]
            ).split(dataset)
        )


def test_nested_combined_splitter_uses_current_subset():
    dataset = DummyGroupedDataset()
    inner = CombinedSplitter(
        parts=[
            ("part_0", SplitterPart(lambda ds: ds.get_subset(v2=[0, 1]), NoSplit(2, train=lambda ds: ds))),
            ("part_1", SplitterPart(lambda ds: ds.get_subset(v2=2), NoSplit(2, test=lambda ds: ds))),
        ]
    )
    outer = CombinedSplitter(parts=[("part_0", SplitterPart(lambda ds: ds.get_subset(v1="b"), inner))])
    assert list(outer.split(dataset)) == [([("b", 0), ("b", 1)], [("b", 2)])] * 2


def test_combined_splitter_keeps_tpcp_parameters_and_clones():
    splitter = CombinedSplitter(parts=[("part_0", SplitterPart(lambda ds: ds, NoSplit(2, train=lambda ds: ds)))])
    cloned = clone(splitter)
    assert cloned is not splitter
    assert cloned.parts[0][1] is not splitter.parts[0][1]
    assert cloned.parts[0][1].splitter is not splitter.parts[0][1].splitter
    assert list(cloned.split(DummyDataset())) == list(splitter.split(DummyDataset()))
    cloned.set_params(parts__part_0__splitter__n_splits=3)
    assert len(list(cloned.split(DummyDataset()))) == 3
    assert len(list(splitter.split(DummyDataset()))) == 2
    cloned.set_params(parts__part_0__splitter=NoSplit(1, test=lambda ds: ds))
    assert list(cloned.split(DummyDataset())) == [([], DummyDataset().group_labels)]


class _WrongCountSplitter(BaseDatasetSplitter):
    def __init__(self, declared, actual):
        self.declared = declared
        self.actual = actual

    def get_n_splits(self, dataset: Dataset) -> int:
        return self.declared

    def split(self, dataset: Dataset) -> Iterator:
        for _ in range(self.actual):
            yield dataset.group_labels, []


def test_combined_splitter_rejects_declared_and_actual_count_mismatches():
    dataset = DummyDataset()
    mismatch = CombinedSplitter(
        parts=[("part_0", SplitterPart(lambda ds: ds, NoSplit(2))), ("part_1", SplitterPart(lambda ds: ds, NoSplit(3)))]
    )
    with pytest.raises(ValueError, match="same number"):
        next(mismatch.split(dataset))
    for actual in (1, 3):
        with pytest.raises(ValueError, match="fold count"):
            list(
                CombinedSplitter(parts=[("part_0", SplitterPart(lambda ds: ds, _WrongCountSplitter(2, actual)))]).split(
                    dataset
                )
            )


def test_selectors_reject_unknown_group_labels():
    dataset = DummyGroupedDataset().groupby("v1")
    selector = lambda ds: ds.get_subset(index=ds.index.assign(v1="z"))
    with pytest.raises(ValueError, match="subset"):
        list(NoSplit(1, train=selector).split(dataset))
    with pytest.raises(ValueError, match="subset"):
        list(CombinedSplitter(parts=[("part_0", SplitterPart(selector, NoSplit(1)))]).split(dataset))


def test_selectors_use_group_labels_for_partial_or_changed_rows():
    dataset = DummyGroupedDataset().groupby("v1")
    partial = lambda ds: ds.get_subset(bool_map=[True] + [False] * 14)
    changed_rows = lambda ds: ds.get_subset(index=ds.index.rename(columns={"v2": "other"}))
    assert list(NoSplit(1, train=partial).split(dataset)) == [([("a",)], [])]
    assert list(
        CombinedSplitter(parts=[("part_0", SplitterPart(partial, NoSplit(1, train=lambda ds: ds)))]).split(dataset)
    ) == [([("a",)], [])]
    assert len(dataset.get_subset(group_labels=[("a",)]).index) == 5
    assert list(NoSplit(1, train=changed_rows).split(dataset)) == [(dataset.group_labels, [])]


def test_partial_selector_assigns_the_whole_group_to_validation():
    dataset = DummyGroupedDataset().groupby("v1")
    splitter = NoSplit(
        1,
        train=lambda ds: ds.get_subset(v1="b"),
        test=lambda ds: ds.get_subset(bool_map=[True] + [False] * 14),
    )
    result = cross_validate(
        Optimize(DummyOptimizablePipeline()),
        dataset,
        cv=splitter,
        scoring=lambda _pipeline, data_point: len(data_point.index),
        progress_bar=False,
    )
    assert result["test__agg__score"] == [5]


def test_valid_empty_selection_and_reordered_group_labels():
    dataset = DummyGroupedDataset().groupby("v1")
    empty = lambda ds: ds.get_subset(bool_map=[False] * len(ds.index))
    reversed_groups = lambda ds: ds.get_subset(group_labels=list(reversed(ds.group_labels)))
    assert list(NoSplit(1, train=empty).split(dataset)) == [([], [])]
    assert list(NoSplit(1, train=reversed_groups).split(dataset)) == [([("c",), ("b",), ("a",)], [])]


def test_native_splitters_work_in_validation_and_grid_search():
    dataset = DummyDataset()
    splitter = CombinedSplitter(parts=[("part_0", SplitterPart(lambda ds: ds, DatasetSplitter(2)))])
    result = cross_validate(
        Optimize(DummyOptimizablePipeline()), dataset, cv=splitter, scoring=_score, progress_bar=False
    )
    assert len(result["test__agg__score"]) == 2
    optimizer = GridSearchCV(
        DummyOptimizablePipeline(), [{"para_1": 1}], cv=splitter, scoring=_score, progress_bar=False
    )
    optimizer.optimize(dataset)
    assert len(optimizer.cv_results_["split0__test__agg__score"]) == 1
    assert len(optimizer.cv_results_["split1__test__agg__score"]) == 1


def test_positional_fold_list_is_adapted_on_reordered_dataset():
    dataset = DummyDataset()[[4, 1, 3, 0, 2]]
    positional_folds = [([0, 1, 2], [3, 4]), ([3, 4], [0, 1, 2])]
    assert list(DatasetSplitter(positional_folds).split(dataset)) == [
        ([(4,), (1,), (3,)], [(0,), (2,)]),
        ([(0,), (2,)], [(4,), (1,), (3,)]),
    ]
    result = cross_validate(
        Optimize(DummyOptimizablePipeline()), dataset, cv=positional_folds, scoring=_score, progress_bar=False
    )
    assert result["test__data_labels"] == [[(0,), (2,)], [(4,), (1,), (3,)]]


def test_positional_fold_list_survives_count_split_and_clone():
    dataset = DummyDataset()[[4, 1, 3, 0, 2]]
    folds = [([0, 1, 2], [3, 4]), ([3, 4], [0, 1, 2])]
    splitter = DatasetSplitter(folds)
    cloned = splitter.clone()
    expected = [
        ([(4,), (1,), (3,)], [(0,), (2,)]),
        ([(0,), (2,)], [(4,), (1,), (3,)]),
    ]
    assert splitter.get_n_splits(dataset) == 2
    assert list(splitter.split(dataset)) == expected
    assert list(splitter.split(dataset)) == expected
    assert list(cloned.split(dataset)) == expected


def test_combined_raw_fold_list_survives_count_split_and_grid_search():
    dataset = DummyDataset()
    folds = [([0, 1, 2], [3, 4]), ([3, 4], [0, 1, 2])]
    splitter = CombinedSplitter(parts=[("part_0", SplitterPart(lambda ds: ds, folds))])
    cloned = splitter.clone()
    expected = [
        ([(0,), (1,), (2,)], [(3,), (4,)]),
        ([(3,), (4,)], [(0,), (1,), (2,)]),
    ]
    assert splitter.get_n_splits(dataset) == 2
    assert list(splitter.split(dataset)) == expected
    assert list(splitter.split(dataset)) == expected
    assert list(cloned.split(dataset)) == expected
    optimizer = GridSearchCV(
        DummyOptimizablePipeline(), [{"para_1": 1}], cv=splitter, scoring=_score, progress_bar=False
    )
    optimizer.optimize(dataset)
    assert len(optimizer.cv_results_["split0__test__agg__score"]) == 1
    assert len(optimizer.cv_results_["split1__test__agg__score"]) == 1


@pytest.mark.parametrize("make_folds", [lambda: iter([([0, 1], [2, 3, 4])]), lambda: (([0, 1], [2, 3, 4]),)])
def test_non_list_positional_folds_require_explicit_list_conversion(make_folds):
    dataset = DummyDataset()
    with pytest.raises(ValueError, match=r"list\(folds\)"):
        DatasetSplitter(make_folds()).get_n_splits(dataset)
    with pytest.raises(ValueError, match=r"list\(folds\)"):
        CombinedSplitter(parts=[("part_0", SplitterPart(lambda ds: ds, make_folds()))]).get_n_splits(dataset)
    with pytest.raises(ValueError, match=r"list\(folds\)"):
        cross_validate(
            Optimize(DummyOptimizablePipeline()), dataset, cv=make_folds(), scoring=_score, progress_bar=False
        )
    with pytest.raises(ValueError, match=r"list\(folds\)"):
        GridSearchCV(
            DummyOptimizablePipeline(), [{"para_1": 1}], cv=make_folds(), scoring=_score, progress_bar=False
        ).optimize(dataset)
