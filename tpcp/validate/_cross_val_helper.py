from collections.abc import Iterator
from typing import Optional, Union

from sklearn.model_selection import BaseCrossValidator, check_cv

from tpcp import BaseTpcpObject, Dataset


class DatasetSplitter(BaseTpcpObject):
    """Wrapper around sklearn cross-validation splitters to support grouping and stratification with tpcp-Datasets.

    This wrapper can be used instead of a sklearn-style splitter with all methods that support a ``cv`` parameter.
    Whenever you want to do complicated cv-logic (like grouping or stratification's), this wrapper is the way to go.

    .. warning:: We don't validate if the selected ``base_splitter`` does anything useful with the provided
        ``groupby`` and ``stratify`` information.
        This wrapper just ensures, that the information is correctly extracted from the dataset and passed to the
        ``split`` method of the ``base_splitter``.
        So if you are using a normal ``KFold`` splitter, the ``groupby`` and ``stratify`` arguments will have no effect.

    Parameters
    ----------
    base_splitter
        The base splitter to use. Can be an integer (for ``KFold``), an iterator, or any other valid sklearn-splitter.
        The default is None, which will use the sklearn default ``KFold`` splitter with 5 splits.
    groupby
        The column(s) to group by. If None, no grouping is done.
        Must be a subset of the columns in the dataset.

        This will generate a set of unique string labels with the same shape as the dataset.
        This will passed to the base splitter as the ``groups`` parameter.
        It is up to the base splitter to decide what to do with the generated labels.
    stratify
        The column(s) to stratify by. If None, no stratification is done.
        Must be a subset of the columns in the dataset.

        This will generate a set of unique string labels with the same shape as the dataset.
        This will passed to the base splitter as the ``y`` parameter, acting as "mock" target labels, as sklearn only
        support stratification on classification outcome targets.
        It is up to the base splitter to decide what to do with the generated labels.

    """

    def __init__(
        self,
        base_splitter: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
        *,
        groupby: Optional[Union[str, list[str]]] = None,
        stratify: Optional[Union[str, list[str]]] = None,
    ):
        self.base_splitter = base_splitter
        self.stratify = stratify
        self.groupby = groupby

    def _get_splitter(self):
        return check_cv(self.base_splitter, y=None, classifier=True)

    def _get_labels(self, dataset: Dataset, labels: Union[None, str, list[str]]):
        if labels:
            return dataset.create_string_group_labels(labels)
        return None

    def split(self, dataset: Dataset) -> Iterator[tuple[list[int], list[int]]]:
        """Split the dataset into train and test sets."""
        return self._get_splitter().split(
            dataset, y=self._get_labels(dataset, self.stratify), groups=self._get_labels(dataset, self.groupby)
        )

    def get_n_splits(self, dataset: Dataset) -> int:
        """Get the number of splits."""
        return self._get_splitter().get_n_splits(
            dataset, y=self._get_labels(dataset, self.stratify), groups=self._get_labels(dataset, self.groupby)
        )
