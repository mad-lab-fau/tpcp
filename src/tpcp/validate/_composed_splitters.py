"""Dataset-subset selection and composition for label-based cross-validation splitters."""

import numbers
from collections.abc import Callable, Iterator

from tpcp import BaseTpcpObject, Dataset
from tpcp._dataset import GroupLabelT
from tpcp.validate._cross_val_helper import BaseDatasetSplitter, _normalize_splitter, _requested_fold_count

DatasetSelector = Callable[[Dataset], Dataset]


def _validate_subset(parent: Dataset, selected: Dataset) -> list[GroupLabelT]:
    """Require selected group labels to exist in the input dataset."""
    labels = selected.group_labels
    if not set(labels).issubset(parent.group_labels):
        raise ValueError("Selector returned a subset containing a group outside its input dataset.")
    return labels


def _selected_labels(dataset: Dataset, selector: DatasetSelector | None) -> list[GroupLabelT]:
    return [] if selector is None else _validate_subset(dataset, selector(dataset))


def _check_disjoint(train: list[GroupLabelT], test: list[GroupLabelT]) -> None:
    if set(train) & set(test):
        raise ValueError("Train and test group labels overlap.")


class NoSplit(BaseDatasetSplitter):
    """Repeat a fixed train/test assignment selected from the current dataset.

    Omitted selectors contribute no labels. Each selector is called once per ``split`` iteration.
    The selected group labels determine the assignment; selecting only some rows of a group
    assigns the original whole group when a validation consumer selects it by label.

    Parameters
    ----------
    n_splits
        A positive integer number of identical folds to yield. Use ``None`` to
        supply the count through ``split(dataset, n_splits=...)`` or through
        another child of :class:`CombinedSplitter`.
    train, test
        Optional callables that receive the current dataset and return a dataset with group labels from it.
    """

    def __init__(
        self, n_splits: int | None, *, train: DatasetSelector | None = None, test: DatasetSelector | None = None
    ) -> None:
        self.n_splits = n_splits
        self.train = train
        self.test = test

    def get_n_splits(self, dataset: Dataset) -> int | None:  # noqa: ARG002
        """Return the configured number of folds, or ``None`` if unspecified."""
        if self.n_splits is None:
            return None
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, numbers.Integral) or self.n_splits < 1:
            raise ValueError("n_splits must be a positive integer.")
        return int(self.n_splits)

    def split(
        self, dataset: Dataset, n_splits: int | None = None
    ) -> Iterator[tuple[list[GroupLabelT], list[GroupLabelT]]]:
        """Yield the selected group labels for the requested number of folds."""
        count = _requested_fold_count(n_splits, self.get_n_splits(dataset))
        train = _selected_labels(dataset, self.train)
        test = _selected_labels(dataset, self.test)
        _check_disjoint(train, test)
        for _ in range(count):
            yield train.copy(), test.copy()


class SplitterPart(BaseTpcpObject):
    """Select a dataset subset and configure its splitter as tpcp parameters.

    Parameters
    ----------
    selector
        Callable receiving the current dataset and returning a subset whose group labels belong to it.
    splitter
        A native tpcp splitter or any input accepted by :class:`DatasetSplitter`, including raw sklearn
        splitters and explicit lists of positional folds. Wrap raw inputs in :class:`DatasetSplitter`
        to expose its parameters through nested ``get_params`` and ``set_params`` calls.
    """

    def __init__(self, selector: DatasetSelector, splitter: object) -> None:
        self.selector = selector
        self.splitter = splitter


class CombinedSplitter(BaseDatasetSplitter):
    """Combine corresponding folds of splitters applied to selected dataset parts.

    Pass a list of one or more ``(name, SplitterPart(selector, splitter))`` pairs as ``parts``.
    Names are unique strings without ``__``, following tpcp's composite parameter convention.
    Each selector receives the current dataset and returns a subset. A child splitter can be a native
    tpcp splitter or any input accepted by :class:`DatasetSplitter`, including raw sklearn splitters.
    Positional fold assignments must be supplied as lists, including for raw child inputs.

    All specified fold counts must match, and at least one child must specify a count.
    Children that report ``None`` receive that count through their ``split`` method.
    All children must yield exactly that many folds. Train and test labels are
    deduplicated separately in first-occurrence order, and overlapping assignments
    raise ``ValueError``. Selectors receive the current input dataset, including
    within nested splits, and may return any dataset whose group labels belong to
    that input.

    For example, cross-validate real recordings and always train on artificial recordings::

        from tpcp.validate import (
            CombinedSplitter,
            DatasetSplitter,
            NoSplit,
            SplitterPart,
        )

        cv = CombinedSplitter(
            parts=[
                (
                    "real",
                    SplitterPart(
                        lambda ds: ds.get_subset(recording_type="real"),
                        DatasetSplitter(5, groupby="participant"),
                    ),
                ),
                (
                    "artificial",
                    SplitterPart(
                        lambda ds: ds.get_subset(recording_type="artificial"),
                        NoSplit(None, train=lambda ds: ds),
                    ),
                ),
            ],
        )

        cv.set_params(parts__real__splitter__base_splitter=3)

    Replace a whole part with ``set_params(parts__real=new_part)``, or replace its selector or splitter
    with ``parts__real__selector`` or ``parts__real__splitter``. Adding or removing parts requires
    replacing the entire ``parts`` list.

    Parameters
    ----------
    parts
        A composite parameter list of named :class:`SplitterPart` objects, in the order their fold
        contributions are combined. Migrate old ``(selector, splitter)`` entries to
        ``(name, SplitterPart(selector, splitter))``.
    """

    _composite_params = ("parts",)

    def __init__(self, parts: list[tuple[str, SplitterPart]]) -> None:
        self.parts = parts

    def _prepare(self, dataset: Dataset) -> list[tuple[Dataset, BaseDatasetSplitter]]:
        if not self.parts:
            raise ValueError("CombinedSplitter requires at least one named SplitterPart.")
        prepared = []
        for _, part in self.parts:
            selected = part.selector(dataset)
            _validate_subset(dataset, selected)
            prepared.append((selected, _normalize_splitter(part.splitter)))
        return prepared

    @staticmethod
    def _fold_count(prepared: list[tuple[Dataset, BaseDatasetSplitter]]) -> tuple[int, list[int | None]]:
        reported = [splitter.get_n_splits(dataset) for dataset, splitter in prepared]
        counts = [count for count in reported if count is not None]
        if not counts:
            raise ValueError("CombinedSplitter requires at least one child to report an integer number of folds.")
        if len(set(counts)) != 1:
            raise ValueError("All child splitters must report the same number of folds.")
        return counts[0], reported

    def get_n_splits(self, dataset: Dataset) -> int:
        """Return the shared number of folds after selecting each part."""
        return self._fold_count(self._prepare(dataset))[0]

    def split(
        self, dataset: Dataset, n_splits: int | None = None
    ) -> Iterator[tuple[list[GroupLabelT], list[GroupLabelT]]]:
        """Yield deduplicated, disjoint train and test labels for corresponding child folds."""
        prepared = self._prepare(dataset)
        available, reported = self._fold_count(prepared)
        count = _requested_fold_count(n_splits, available)
        iterators = [
            iter(splitter.split(selected, n_splits=count)) if child_count is None else iter(splitter.split(selected))
            for (selected, splitter), child_count in zip(prepared, reported, strict=True)
        ]
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
        for child, child_count in zip(iterators, reported, strict=True):
            if child_count is not None and count < available:
                continue
            try:
                next(child)
            except StopIteration:
                continue
            raise ValueError("A child splitter yielded more folds than its reported fold count.")
