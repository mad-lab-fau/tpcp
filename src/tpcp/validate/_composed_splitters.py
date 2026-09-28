"""Dataset-subset selection and composition for label-based cross-validation splitters."""

import numbers
from collections.abc import Callable, Iterator
from typing import Optional

from tpcp import Dataset
from tpcp._dataset import GroupLabelT
from tpcp.validate._cross_val_helper import BaseDatasetSplitter, _normalize_splitter

DatasetSelector = Callable[[Dataset], Dataset]


def _validate_subset(parent: Dataset, selected: Dataset) -> list[GroupLabelT]:
    """Require selected group labels to exist in the input dataset."""
    labels = selected.group_labels
    if not set(labels).issubset(parent.group_labels):
        raise ValueError("Selector returned a subset containing a group outside its input dataset.")
    return labels


def _selected_labels(dataset: Dataset, selector: Optional[DatasetSelector]) -> list[GroupLabelT]:
    return [] if selector is None else _validate_subset(dataset, selector(dataset))


def _check_disjoint(train: list[GroupLabelT], test: list[GroupLabelT]) -> None:
    if set(train) & set(test):
        raise ValueError("Train and test group labels overlap.")


class NoSplit(BaseDatasetSplitter):
    """Repeat a fixed train/test assignment selected from the current dataset.

    Omitted selectors contribute no labels. Each selector is called once per ``split`` iteration.

    Parameters
    ----------
    n_splits
        A positive integer number of identical folds to yield.
    train, test
        Optional callables that receive the current dataset and return a dataset with group labels from it.
    """

    def __init__(
        self, n_splits: int, *, train: Optional[DatasetSelector] = None, test: Optional[DatasetSelector] = None
    ) -> None:
        self.n_splits = n_splits
        self.train = train
        self.test = test

    def get_n_splits(self, dataset: Dataset) -> int:  # noqa: ARG002
        """Return the configured number of folds."""
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, numbers.Integral) or self.n_splits < 1:
            raise ValueError("n_splits must be a positive integer.")
        return int(self.n_splits)

    def split(self, dataset: Dataset) -> Iterator[tuple[list[GroupLabelT], list[GroupLabelT]]]:
        """Yield the selected group labels exactly ``n_splits`` times."""
        count = self.get_n_splits(dataset)
        train = _selected_labels(dataset, self.train)
        test = _selected_labels(dataset, self.test)
        _check_disjoint(train, test)
        for _ in range(count):
            yield train.copy(), test.copy()


class _VariadicPairsMeta(type):
    def __call__(cls, *pairs, **kwargs):
        if kwargs:
            return super().__call__(**kwargs)
        return super().__call__(pairs)


class CombinedSplitter(BaseDatasetSplitter, metaclass=_VariadicPairsMeta):
    """Combine corresponding folds of splitters applied to selected dataset parts.

    Construct with one or more ``(selector, splitter)`` pairs. Each selector receives the current
    dataset and returns a subset. A child splitter can be a native tpcp splitter or any input
    accepted by :class:`DatasetSplitter`, including raw sklearn splitters.

    Parameters
    ----------
    parts
        The selector and splitter pairs, supplied as variadic positional arguments.
    """

    def __init__(self, parts: tuple[tuple[DatasetSelector, object], ...]) -> None:
        self.parts = parts

    def _prepare(self, dataset: Dataset) -> list[tuple[Dataset, BaseDatasetSplitter]]:
        if not self.parts:
            raise ValueError("CombinedSplitter requires at least one selector and splitter pair.")
        prepared = []
        for selector, splitter in self.parts:
            selected = selector(dataset)
            _validate_subset(dataset, selected)
            prepared.append((selected, _normalize_splitter(splitter)))
        return prepared

    @staticmethod
    def _fold_count(prepared: list[tuple[Dataset, BaseDatasetSplitter]]) -> int:
        counts = [splitter.get_n_splits(dataset) for dataset, splitter in prepared]
        if len(set(counts)) != 1:
            raise ValueError("All child splitters must report the same number of folds.")
        return counts[0]

    def get_n_splits(self, dataset: Dataset) -> int:
        """Return the shared number of folds after selecting each part."""
        return self._fold_count(self._prepare(dataset))

    def split(self, dataset: Dataset) -> Iterator[tuple[list[GroupLabelT], list[GroupLabelT]]]:
        """Yield deduplicated, disjoint train and test labels for corresponding child folds."""
        prepared = self._prepare(dataset)
        count = self._fold_count(prepared)
        iterators = [iter(splitter.split(selected)) for selected, splitter in prepared]
        for _ in range(count):
            folds = []
            for child in iterators:
                try:
                    folds.append(next(child))
                except StopIteration as e:
                    raise ValueError("A child splitter yielded fewer folds than its reported fold count.") from e
            train = list(dict.fromkeys(label for fold in folds for label in fold[0]))
            test = list(dict.fromkeys(label for fold in folds for label in fold[1]))
            _check_disjoint(train, test)
            yield train, test
        for child in iterators:
            try:
                next(child)
            except StopIteration:
                continue
            raise ValueError("A child splitter yielded more folds than its reported fold count.")
