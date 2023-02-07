"""Base class for all datasets."""
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, cast, overload

import numpy as np
import pandas as pd
from typing_extensions import Self

from tpcp._base import BaseTpcpObject

DatasetT = TypeVar("DatasetT", bound="_Dataset")


class _Dataset(BaseTpcpObject):
    groupby_cols: Optional[Union[List[str], str]]
    subset_index: Optional[pd.DataFrame]

    @property
    def index(self) -> pd.DataFrame:
        """Get index."""
        if self.subset_index is None:
            # Note, in the past we recreated the index when ever there was a call to `self.index` and their was no
            # subset index.
            # To avoid unnecessary overhead and potential edgecases where the index creation is not deterministic (
            # i.e. the data on disc changes between calls), we now only create the index once.
            # This still doesn't solve the issues of having a non-deterministic index creation (still bad!),
            # but the errors that you are running into are hopefully less obscure.
            self.subset_index = self._create_check_index()

        return self.subset_index

    def _create_check_index(self):
        """Check the index creation.

        We create the index twice to check if the index creation is deterministic.
        If not we raise an error.
        This is fundamentally important for datasets to be deterministic.
        While we can not catch all related issues (i.e. determinism across different machines), this should catch the
        most obvious ones.

        In case, creating the index twice is too expensive, users can overwrite this method.
        But better to catch errors early.
        """
        index_1 = self.create_index()
        index_2 = self.create_index()
        if not index_1.equals(index_2):
            raise RuntimeError(
                "Index creation is not deterministic! "
                "This is a fundamental requirement for datasets to be deterministic otherwise you might run into all "
                "kinds of issues. "
                "Please check your implementation of the `create_index` method.\n\n"
                "Typically sources of non-determinism are:\n"
                " - Using `random` somewhere in the code\n"
                " - Storing (intermediate) data in non-sorted containers (e.g. `set`)\n"
                " - Relying on the ordering of files from the file system\n\n"
                "For the last to cases we recommend to sort the dataframe you return from `create_index` "
                "explicitly using `sort_values`."
            )

        return index_1

    @property
    def groups(self) -> List[Union[str, Tuple[str, ...]]]:
        """Get all groups based on the set groupby level.

        This will either return a list of strings/integers, if there is only a single group level or index column.
        If there are multiple groupy levels/index columns, it will return a list of named tuples.

        Note, that if one of the groupby levels/index columns is not a valid Python attribute name (e.g. in contains
        spaces or starts with a number), the named tuple will not contain the correct column name!
        For more information see the documentation of the `rename` parameter of :func:`collections.namedtuple`.
        """
        if len(self._get_groupby_columns()) == 1:
            return self._get_unique_groups().to_list()
        return list(self._get_unique_groups().to_frame().itertuples(index=False, name=type(self).__name__))

    @property
    def group(self) -> Union[str, Tuple[str, ...]]:
        """Get the current group.

        Note, this attribute can only be used, if there is just a single group.
        If there is only a single groupby column or column in the index, this will return a string.
        Otherwise, this will return a named tuple.
        """
        self.assert_is_single_group("group")
        return self.groups[0]

    def __len__(self) -> int:
        """Get the length of the dataset.

        This is equal to the number of rows in the index, if `self.groupby_cols=None`.
        Otherwise, it is equal to the number of unique groups.
        """
        return len(self.groups)

    @property
    def shape(self) -> Tuple[int]:
        """Get the shape of the dataset.

        This only reports a single dimension.
        This is equal to the number of rows in the index, if `self.groupby_cols=None`.
        Otherwise, it is equal to the number of unique groups.
        """
        return (len(self),)

    @property
    def grouped_index(self) -> pd.DataFrame:
        """Return the index with the `groupby` columns set as multiindex."""
        groupby_cols = self._get_groupby_columns()
        try:
            return self.index.set_index(groupby_cols, drop=False)
        except KeyError as e:
            raise KeyError(
                f"You can only groupby columns that are part of the index columns ({list(self.index.columns)}) and not"
                f" {self.groupby_cols}"
            ) from e

    def _get_groupby_columns(self) -> List[str]:
        """Get the groupby columns."""
        if self.groupby_cols is None:
            return self.index.columns.to_list()
        return _ensure_is_list(self.groupby_cols)

    def _get_unique_groups(self) -> Union[pd.MultiIndex, pd.Index]:
        return self.grouped_index.index.unique()

    def __getitem__(self, subscript: Union[int, Sequence[int], np.ndarray, slice]) -> Self:
        """Return a dataset object containing only the selected row indices of `self.groups`."""
        multi_index = self._get_unique_groups()[subscript]
        if not isinstance(multi_index, pd.Index):
            multi_index = [multi_index]

        return self.clone().set_params(subset_index=self.grouped_index.loc[multi_index].reset_index(drop=True))

    def groupby(self, groupby_cols: Optional[Union[List[str], str]]) -> Self:
        """Return a copy of the dataset grouped by the specified columns.

        Each unique group represents a single data point in the resulting dataset.

        Parameters
        ----------
        groupby_cols
            None (no grouping) or a valid subset of the columns available in the dataset index.

        """
        grouped_ds = self.clone().set_params(groupby_cols=groupby_cols)
        # Get grouped index to raise an error here, in case the `groupby_cols` are invalid.
        _ = grouped_ds.grouped_index
        return grouped_ds

    def get_subset(
        self,
        *,
        groups: Optional[List[Union[str, Tuple[str, ...]]]] = None,
        index: Optional[pd.DataFrame] = None,
        bool_map: Optional[Sequence[bool]] = None,
        **kwargs: Union[List[str], str],
    ) -> Self:
        """Get a subset of the dataset.

        .. note::
            All arguments are mutable exclusive!

        Parameters
        ----------
        groups
            A valid row locator or slice that can be passed to `self.grouped_index.loc[locator, :]`.
            This basically needs to be a subset of `self.groups`.
            Note that this is the only indexer that works on the grouped index.
            All other indexers work on the pure index.
        index
            `pd.DataFrame` that is a valid subset of the current dataset index.
        bool_map
            bool-map that is used to index the current index-dataframe.
            The list **must** be of same length as the number of rows in the index.
        **kwargs
            The key **must** be the name of an index column.
            The value is a list containing strings that correspond to the categories that should be kept.
            For examples see above.

        Returns
        -------
        subset
            New dataset object filtered by specified parameters.

        """
        if [x is None or (isinstance(x, dict) and len(x) == 0) for x in [groups, index, bool_map, kwargs]].count(
            False
        ) > 1:
            raise ValueError("Only one of `groups`, `selected_keys`, `index`, `bool_map` or kwarg can be set!")

        if groups is not None:
            return self.clone().set_params(subset_index=self.grouped_index.loc[groups, :].reset_index(drop=True))

        if index is not None:
            if len(index) == 0:
                raise ValueError("Provided index is not formatted correctly. Make sure it is not empty!")

            return self.clone().set_params(subset_index=index.reset_index(drop=True))

        if bool_map is not None:
            if len(bool_map) != self.index.shape[0]:
                raise ValueError(f"Parameter bool_map must have length {self.index.shape[0]} but has {len(bool_map)}!")

            return self.clone().set_params(subset_index=self.index[bool_map].reset_index(drop=True))

        if len(kwargs) > 0:
            cleaned_kwargs = cast(Dict[str, List[str]], {k: _ensure_is_list(v) for k, v in kwargs.items()})

            # Check if all values are actually in their respective columns.
            # This is not strictly required, but avoids user error
            _assert_all_in_df(self.index, cleaned_kwargs)

            subset_index = self.index.loc[
                self.index[list(cleaned_kwargs.keys())].isin(cleaned_kwargs).all(axis=1)
            ].reset_index(drop=True)
            if len(subset_index) == 0:
                raise KeyError(f"No datapoint in the dataset matched the following filter: {cleaned_kwargs}")

            return self.clone().set_params(subset_index=subset_index)

        raise ValueError("At least one of `groups`, `selected_keys`, `index`, `bool_map` or kwarg must not be None!")

    def __repr__(self) -> str:
        """Return string representation of the dataset object."""
        repr_index = self.index if self.groupby_cols is None else self.grouped_index
        repr_index = str(repr_index).replace("\n", "\n   ")
        return f"{self.__class__.__name__} [{self.shape[0]} groups/rows]\n\n   {repr_index}\n\n   "[:-5]

    def _repr_html_(self) -> str:
        """Return html representation of the dataset object."""
        repr_index = self.index if self.groupby_cols is None else self.grouped_index

        df_repr = (
            repr_index._repr_html_()
            .replace("<div>", '<div style="margin-top: 0em">')
            .replace('<table border="1" class="dataframe"', '<table style="margin-left: 3em;"')
            .replace("<th>", '<th style="text-align: center;">')
            .replace("<td>", '<td style="text-align: center; padding-left: 2em; padding-right: 2em;">')
        )
        return (
            f'<h4 style="margin-bottom: 0.1em;">{self.__class__.__name__} [{self.shape[0]} groups/rows]</h3>\n'
            + df_repr
        )

    def __iter__(self) -> Iterator[Self]:
        """Return generator object containing a subset for every combination up to and including the selected level."""
        return (self.__getitem__(i) for i in range(self.shape[0]))

    def iter_level(self, level: str) -> Iterator[Self]:
        """Return generator object containing a subset for every category from the selected level.

        Parameters
        ----------
        level
            Optional `str` that sets the level which shall be used for iterating.
            This **must** be one of the columns names of the index.

        Returns
        -------
        subset
            New dataset object containing only one category in the specified `level`.

        """
        if level not in self.index.columns:
            raise ValueError(f"`level` must be one of {list(self.index.columns)}")

        return (self.get_subset(**{level: category}) for category in self.index[level].unique())

    def is_single(self, groupby_cols: Optional[Union[str, List[str]]]) -> bool:
        """Return True if index contains only one row/group with the given groupby settings.

        If `groupby_cols=None` this checks if there is only a single row left.
        If you want to check if there is only a single group within the current grouping, use `is_single_group` instead.

        Parameters
        ----------
        groupby_cols
            None (no grouping) or a valid subset of the columns available in the dataset index.

        """
        return len(self.groupby(groupby_cols)) == 1

    def is_single_group(self) -> bool:
        """Return True if index contains only one group."""
        return len(self) == 1

    def assert_is_single(self, groupby_cols: Optional[Union[str, List[str]]], property_name) -> None:
        """Raise error if index does contain more than one group/row with the given groupby settings.

        This should be used when implementing access to data values, which can only be accessed when only a single
        trail/participant/etc. exist in the dataset.

        Parameters
        ----------
        groupby_cols
            None (no grouping) or a valid subset of the columns available in the dataset index.
        property_name
            Name of the property this check is used in.
            Used to format the error message.

        """
        if not self.is_single(groupby_cols):
            if groupby_cols is None:
                groupby_cols = self.index.columns.to_list()
            raise ValueError(
                f"The attribute `{property_name}` of dataset {self.__class__.__name__} can only be accessed if there "
                f"is only a single combination of the columns {groupby_cols} left in a data subset,"
            )

    def assert_is_single_group(self, property_name) -> None:
        """Raise error if index does contain more than one group/row.

        Note that this is different from `assert_is_single` as it is aware of the current grouping.
        Instead of checking that a certain combination of columns is left in the dataset, it checks that only a
        single group exists with the already selected grouping as defined by `self.groupby_cols`.

        Parameters
        ----------
        property_name
            Name of the property this check is used in.
            Used to format the error message.

        """
        if not self.is_single_group():
            if self.groupby_cols is None:
                group_error_str = (
                    "Currently the dataset is not grouped. "
                    "This means a single group is identical to having only a single row in the "
                    "dataset index."
                )
            else:
                group_error_str = f"Currently the dataset is grouped by {self.groupby_cols}."

            raise ValueError(
                f"The attribute `{property_name}` of dataset {self.__class__.__name__} can only be accessed if there is"
                f" only a single group left in a data subset. " + group_error_str
            )

    def create_group_labels(self, label_cols: Union[str, List[str]]) -> List[str]:
        """Generate a list of labels for each group/row in the dataset.

        .. note::
            This has a different use case than the dataset-wide groupby.
            Using `groupby` reduces the effective size of the dataset to the number of groups.
            This method produces a group label for each group/row that is already in the dataset, without changing the
            dataset.

        The output of this method can be used in combination with :class:`~sklearn.model_selection.GroupKFold` as
        the group label.

        Parameters
        ----------
        label_cols
            The columns that should be included in the label.
            If the dataset is already grouped, this must be a subset of  `self.groupby_cols`.

        """
        unique_index = self._get_unique_groups().to_frame()
        try:
            return [str(g) for g in unique_index.set_index(label_cols).index.to_list()]
        except KeyError as e:
            if self.groupby_cols is not None:
                raise KeyError(
                    "When using `create_group_labels` with a grouped dataset, the selected columns must "
                    f"be a subset of `self.groupby_cols` ({self.groupby_cols}) and not ({label_cols})"
                ) from e
            raise KeyError(
                f"The selected label columns ({label_cols}) are not in the index of the dataset "
                f"({list(self.index.columns)})."
            ) from e

    def create_index(self) -> pd.DataFrame:
        """Create the full index for the dataset.

        This needs to be implemented by the subclass.

        .. warning:: Make absolutely sure that the dataframe you return is deterministic and does not change between
                     runs!
                     This can lead to some nasty bugs!
                     We try to catch them internally, but it is not always possible.
                     As tips, avoid reliance on random numbers and make sure that the order is not depend on things
                     like file system order, when creating an index by scanning a directory.
                     Particularly nasty are cases when using non-sorted container like `set`, that sometimes maintain
                     their order, but sometimes don't.
                     At the very least, we recommend to sort the final dataframe you return in `create_index`.

        """
        raise NotImplementedError()


class Dataset(_Dataset):
    """Baseclass for tpcp Dataset objects.

    This class provides fundamental functionality like iteration, getting subsets, and compatibility with `sklearn`'s
    cross validation helpers.

    For more information check out the examples and user guides on datasets.

    Parameters
    ----------
    groupby_cols
        A column name or a list of column names that should be used to group the index before iterating over it.
        For examples see below.
    subset_index
        For all classes that inherit from this class, subset_index **must** be None by default.
        But the subclasses require a `create_index` method that returns a DataFrame representing the index.

    Attributes
    ----------
    index
        The index of the dataset.
        This returns either the `subset_index` or the base index returned by `create_index`.
        Note, that after the first call to index, the index will be cached in `self.subset_index`.
    grouped_index
        The index, but all groupby columns are represented as MultiIndex.
        Note that the order can be different as the order of index.
    groups
        Returns all possible combinations based on the specified `groupby` columns.
        If `groupby` is None, this returns the row indices.
        These are also the groups/indices used when iterating over the dataset.
        The groups are sorted by name.
    shape
        Represents the number of all groups encapsulated in a tuple.
        This is only necessary if `sklearn.model_selection.KFold` is used for splitting the dataset.

    Examples
    --------
    This class is usually not meant to be used directly, but the following code snippets show some common operations
    that can be expected to work for all dataset subclasses.

    >>> import pandas as pd
    >>> from itertools import product
    >>>
    >>> from tpcp import Dataset
    >>>
    >>> test_index = pd.DataFrame(
    ...     list(product(("patient_1", "patient_2", "patient_3"), ("test_1", "test_2"), ("1", "2"))),
    ...     columns=["patient", "test", "extra"],
    ... )
    >>> # We create a little dummy dataset by passing an index directly to `test_index`
    >>> # Usually we would create a subclass with a `create_index` method that returns a DataFrame representing the
    >>> # index.
    >>> dataset = Dataset(subset_index=test_index)
    >>> dataset
    Dataset [12 groups/rows]
    <BLANKLINE>
             patient    test extra
       0   patient_1  test_1     1
       1   patient_1  test_1     2
       2   patient_1  test_2     1
       3   patient_1  test_2     2
       4   patient_2  test_1     1
       5   patient_2  test_1     2
       6   patient_2  test_2     1
       7   patient_2  test_2     2
       8   patient_3  test_1     1
       9   patient_3  test_1     2
       10  patient_3  test_2     1
       11  patient_3  test_2     2

    We can loop over the dataset.
    By default, we will loop over each row.

    >>> for r in dataset[:2]:
    ...     print(r)
    Dataset [1 groups/rows]
    <BLANKLINE>
            patient    test extra
       0  patient_1  test_1     1
    Dataset [1 groups/rows]
    <BLANKLINE>
            patient    test extra
       0  patient_1  test_1     2

    We can also change `groupby` (either in the init or afterwards), to loop over other combinations.
    If we select the level `test`, we will loop over all `patient`-`test` combinations.

    >>> grouped_dataset = dataset.groupby(["patient", "test"])
    >>> grouped_dataset  # doctest: +NORMALIZE_WHITESPACE
    Dataset [6 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     1
                 test_1  patient_1  test_1     2
                 test_2  patient_1  test_2     1
                 test_2  patient_1  test_2     2
       patient_2 test_1  patient_2  test_1     1
                 test_1  patient_2  test_1     2
                 test_2  patient_2  test_2     1
                 test_2  patient_2  test_2     2
       patient_3 test_1  patient_3  test_1     1
                 test_1  patient_3  test_1     2
                 test_2  patient_3  test_2     1
                 test_2  patient_3  test_2     2

    >>> for r in grouped_dataset[:2]:
    ...     print(r)  # doctest: +NORMALIZE_WHITESPACE
    Dataset [1 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     1
                 test_1  patient_1  test_1     2
    Dataset [1 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_2  patient_1  test_2     1
                 test_2  patient_1  test_2     2

    To iterate over the unique values of a specific level use the "iter_level" function:

    >>> for r in list(grouped_dataset.iter_level("patient"))[:2]:
    ...     print(r)  # doctest: +NORMALIZE_WHITESPACE
    Dataset [2 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     1
                 test_1  patient_1  test_1     2
                 test_2  patient_1  test_2     1
                 test_2  patient_1  test_2     2
    Dataset [2 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_2 test_1  patient_2  test_1     1
                 test_1  patient_2  test_1     2
                 test_2  patient_2  test_2     1
                 test_2  patient_2  test_2     2

    We can also get arbitary subsets from the dataset:

    >>> subset = grouped_dataset.get_subset(patient=["patient_1", "patient_2"], extra="2")
    >>> subset  # doctest: +NORMALIZE_WHITESPACE
    Dataset [4 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     2
                 test_2  patient_1  test_2     2
       patient_2 test_1  patient_2  test_1     2
                 test_2  patient_2  test_2     2

    If we want to use datasets in combination with :class:`~sklearn.model_selection.GroupKFold`, we can generate
    valid group labels as follows.
    These grouplabels are strings representing the unique value of the index at the specified levels.

    .. note::
        You usually don't want to use that in combination with `self.groupby`.

    >>> # We are using the ungrouped dataset again!
    >>> group_labels = dataset.create_group_labels(["patient", "test"])
    >>> pd.concat([dataset.index, pd.Series(group_labels, name="group_labels")], axis=1)
          patient    test extra             group_labels
    0   patient_1  test_1     1  ('patient_1', 'test_1')
    1   patient_1  test_1     2  ('patient_1', 'test_1')
    2   patient_1  test_2     1  ('patient_1', 'test_2')
    3   patient_1  test_2     2  ('patient_1', 'test_2')
    4   patient_2  test_1     1  ('patient_2', 'test_1')
    5   patient_2  test_1     2  ('patient_2', 'test_1')
    6   patient_2  test_2     1  ('patient_2', 'test_2')
    7   patient_2  test_2     2  ('patient_2', 'test_2')
    8   patient_3  test_1     1  ('patient_3', 'test_1')
    9   patient_3  test_1     2  ('patient_3', 'test_1')
    10  patient_3  test_2     1  ('patient_3', 'test_2')
    11  patient_3  test_2     2  ('patient_3', 'test_2')

    """

    def __init__(
        self,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.groupby_cols = groupby_cols
        self.subset_index = subset_index

    @staticmethod
    def as_dataclass():
        """Return a version of the Dataset class that can be subclassed using dataclasses."""
        import dataclasses  # pylint: disable=import-outside-toplevel

        @dataclasses.dataclass(eq=False, repr=False, order=False)
        class DatasetDc(_Dataset):
            """Dataclass version of Dataset."""

            groupby_cols: Optional[Union[List[str], str]] = None
            subset_index: Optional[pd.DataFrame] = None

        return DatasetDc

    @staticmethod
    def as_attrs():
        """Return a version of the Dataset class that can be subclassed using `attrs` defined classes.

        Note, this requires `attrs` to be installed!
        """
        from attrs import define  # pylint: disable=import-outside-toplevel

        @define(eq=False, repr=False, order=False, kw_only=True, slots=False)
        class DatasetAt(_Dataset):
            """Attrs version of Dataset."""

            groupby_cols: Optional[Union[List[str], str]] = None
            subset_index: Optional[pd.DataFrame] = None

        return DatasetAt


T = TypeVar("T")


@overload
def _ensure_is_list(x: List[T]) -> List[T]:
    ...


@overload
def _ensure_is_list(x: T) -> List[T]:
    ...


def _ensure_is_list(x):
    return x if isinstance(x, list) else [x]


def _assert_all_in_df(df, dic):
    """Check that all values of the dictionary are in the column 'key' of the pandas dataframe."""
    for key, value in dic.items():
        try:
            index_level = df[key]
        except KeyError as e:
            raise KeyError(f"Can not filter by key `{key}`! Key must be one of {list(df.columns)}!") from e
        if not set(value).issubset(index_level):
            raise KeyError(f"At least one of {value} is not in level {key}")
