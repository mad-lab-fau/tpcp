"""Behavioral tests for dataset-label splitter composition."""

from collections.abc import Iterator

import pytest
from sklearn.model_selection import KFold

from tests.test_pipelines.conftest import DummyDataset, DummyGroupedDataset, DummyOptimizablePipeline
from tpcp import Dataset, clone
from tpcp.optimize import GridSearchCV, Optimize
from tpcp.validate import BaseDatasetSplitter, CombinedSplitter, DatasetSplitter, NoSplit, cross_validate


def _score(_pipeline, data_point):
    return 1 if data_point.group_label[0] is not None else 0


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


def test_combined_splitter_pairs_selected_folds_and_excludes_other_groups():
    dataset = DummyGroupedDataset()
    splitter = CombinedSplitter(
        (lambda ds: ds.get_subset(v1="a"), KFold(5)),
        (lambda ds: ds.get_subset(v1="b"), DatasetSplitter(5)),
    )
    folds = list(splitter.split(dataset))
    assert splitter.get_n_splits(dataset) == 5
    assert len(folds) == 5
    assert folds[0][0] == [("a", i) for i in range(1, 5)] + [("b", i) for i in range(1, 5)]
    assert folds[0][1] == [("a", 0), ("b", 0)]
    assert all("c" not in label for fold in folds for side in fold for label in side)


def test_combined_splitter_deduplicates_ordered_labels_and_detects_overlap():
    dataset = DummyDataset()
    splitter = CombinedSplitter((lambda ds: ds, KFold(5)), (lambda ds: ds, KFold(5)))
    train, test = next(splitter.split(dataset))
    assert train == dataset.group_labels[1:]
    assert test == dataset.group_labels[:1]
    with pytest.raises(ValueError, match="overlap"):
        list(
            CombinedSplitter(
                (lambda ds: ds, NoSplit(1, train=lambda ds: ds)), (lambda ds: ds, NoSplit(1, test=lambda ds: ds))
            ).split(dataset)
        )


def test_nested_combined_splitter_uses_current_subset():
    dataset = DummyGroupedDataset()
    inner = CombinedSplitter(
        (lambda ds: ds.get_subset(v2=[0, 1]), NoSplit(2, train=lambda ds: ds)),
        (lambda ds: ds.get_subset(v2=2), NoSplit(2, test=lambda ds: ds)),
    )
    outer = CombinedSplitter((lambda ds: ds.get_subset(v1="b"), inner))
    assert list(outer.split(dataset)) == [([("b", 0), ("b", 1)], [("b", 2)])] * 2


def test_combined_splitter_keeps_tpcp_parameters_and_clones():
    splitter = CombinedSplitter((lambda ds: ds, NoSplit(2, train=lambda ds: ds)))
    cloned = clone(splitter)
    assert cloned is not splitter
    assert cloned.parts[0][1] is not splitter.parts[0][1]
    assert list(cloned.split(DummyDataset())) == list(splitter.split(DummyDataset()))


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
    mismatch = CombinedSplitter((lambda ds: ds, NoSplit(2)), (lambda ds: ds, NoSplit(3)))
    with pytest.raises(ValueError, match="same number"):
        next(mismatch.split(dataset))
    for actual in (1, 3):
        with pytest.raises(ValueError, match="fold count"):
            list(CombinedSplitter((lambda ds: ds, _WrongCountSplitter(2, actual))).split(dataset))


@pytest.mark.parametrize(
    "selector",
    [
        lambda ds: ds.groupby("v2"),
        lambda ds: ds.get_subset(bool_map=[True] + [False] * 14),
        lambda ds: ds.get_subset(index=ds.index.rename(columns={"v2": "other"})),
        lambda ds: ds.get_subset(index=ds.index.iloc[[0, 0, *range(1, 15)]]),
        lambda ds: ds.get_subset(index=ds.index.assign(v2=99)),
        lambda ds: ds.get_subset(index=ds.index.iloc[[1, 0, *range(2, 15)]]),
    ],
)
def test_selectors_reject_invalid_group_subsets(selector):
    dataset = DummyGroupedDataset().groupby("v1")
    with pytest.raises(ValueError, match="subset"):
        list(NoSplit(1, train=selector).split(dataset))
    with pytest.raises(ValueError, match="subset"):
        list(CombinedSplitter((selector, NoSplit(1))).split(dataset))


def test_valid_empty_selection_and_reordered_whole_groups():
    dataset = DummyGroupedDataset().groupby("v1")
    empty = lambda ds: ds.get_subset(bool_map=[False] * len(ds.index))
    reversed_groups = lambda ds: ds.get_subset(group_labels=list(reversed(ds.group_labels)))
    assert list(NoSplit(1, train=empty).split(dataset)) == [([], [])]
    assert list(NoSplit(1, train=reversed_groups).split(dataset)) == [([("c",), ("b",), ("a",)], [])]


def test_native_splitters_work_in_validation_and_grid_search():
    dataset = DummyDataset()
    splitter = CombinedSplitter((lambda ds: ds, DatasetSplitter(2)))
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


def test_positional_iterable_is_adapted_on_reordered_dataset():
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
