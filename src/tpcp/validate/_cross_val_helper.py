import numbers
import warnings
from collections.abc import Iterator
from typing import Optional, Union

from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedGroupKFold, StratifiedKFold, check_cv

from tpcp import BaseTpcpObject, Dataset


class DatasetSplitter(BaseTpcpObject):
    """Wrapper around sklearn cross-validation splitters to support grouping and stratification with tpcp-Datasets.

    This wrapper can be used instead of a sklearn-style splitter with all methods that support a ``cv`` parameter.
    Whenever you want to do complicated cv-logic (like grouping or stratification's), this wrapper is the way to go.

    You can either select your own base splitter, or we will select from KFold, StratifiedKFold, GroupKFold, or
    StratifiedGroupKFold, depending on the provided ``groupby`` and ``stratify`` parameters.

    .. warning:: If you use a custom splitter, that does not support grouping or stratification, these parameters might
        be silently ignored.

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
    ignore_potentially_invalid_splitter_warning
        We are trying to detect if the provided splitter supports grouping and stratification.
        If they are not supported, but you provided groupby or stratify columns, we will warn you.
        Note, that this warning is not a perfect check, as it is not possible to detect all cases.
        If you know what you are doing, and you want to disable this warning, set this parameter to True.
    """

    def __init__(
        self,
        base_splitter: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
        *,
        groupby: Optional[Union[str, list[str]]] = None,
        stratify: Optional[Union[str, list[str]]] = None,
        ignore_potentially_invalid_splitter_warning: bool = False,
    ):
        self.base_splitter = base_splitter
        self.stratify = stratify
        self.groupby = groupby
        self.ignore_potentially_invalid_splitter_warning = ignore_potentially_invalid_splitter_warning

    def _get_splitter(self):
        cv = self.base_splitter
        cv = 5 if cv is None else cv
        if isinstance(cv, numbers.Integral):
            if self.groupby is not None and self.stratify is not None:
                cv = StratifiedGroupKFold(n_splits=cv)
            elif self.groupby is not None:
                cv = GroupKFold(n_splits=cv)
            elif self.stratify is not None:
                cv = StratifiedKFold(n_splits=cv)
        cv = check_cv(cv, y=None, classifier=True)

        if self.ignore_potentially_invalid_splitter_warning:
            return cv

        # The checks below might be redundant, but it makes the code structure easier to follow.
        msg = None
        if self.groupby and "Group" not in cv.__class__.__name__:
            msg = (
                "You specified groupby columns for the splitter, but it looks like you did not select any of the "
                "typical sklearn splitters that do support grouping. "
                "Splitters that don't support grouping will silently ignore the grouping information.",
            )
        if self.stratify and "Stratified" not in cv.__class__.__name__:
            msg = (
                "You specified stratify columns for the splitter, but it looks like you did not select any of the "
                "typical sklearn splitters that do support stratification. "
                "Splitters that don't support stratification will silently ignore the stratification information.",
            )
        if msg is not None:
            warnings.warn(
                (
                    f"{msg}"
                    "\nTo fix this issue pass a splitter that supports the required functionality as `base_splitter`."
                    "For a list of available splitters see "
                    "https://scikit-learn.org/stable/api/sklearn.model_selection.html "
                    "\nIf you provided a custom splitter, and you know what you are doing, you can disable this "
                    "warning, by setting the `ignore_potentially_invalid_splitter_warning=True` when creating the "
                    "DatasetSplitter object."
                ),
                UserWarning,
                stacklevel=2,
            )
        return cv

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
