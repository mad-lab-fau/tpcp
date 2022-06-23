from collections import namedtuple
from itertools import product
from operator import itemgetter

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from tpcp import Dataset


def _create_valid_index(input_dict=None, columns_names=None):
    if input_dict is None:
        return pd.DataFrame(
            {
                "patients": [
                    "patient_1",
                    "patient_1",
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_2",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                ],
                "tests": [
                    "test_1",
                    "test_1",
                    "test_2",
                    "test_2",
                    "test_1",
                    "test_1",
                    "test_1",
                    "test_1",
                    "test_2",
                    "test_2",
                    "test_3",
                    "test_3",
                ],
                "extra with space": ["0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1"],
            }
        )

    output = {column_name: [] for column_name in columns_names}

    for key, value in input_dict.items():
        combinations = list(product(*([[key]] + list(map(itemgetter(1), value.items())))))

        for i in range(len(combinations[0])):
            for val in map(itemgetter(i), combinations):
                output[columns_names[i]].append(val)

    return pd.DataFrame(output)


def _create_random_bool_map(n, seed):
    np.random.seed(seed)
    return list(map(lambda x: x >= 0.5, np.random.rand(n)))


class TestDataset:
    @pytest.mark.parametrize(
        "groupby,length",
        [
            ("patients", 3),
            (["patients"], 3),
            (["patients", "tests"], 6),
            (["patients", "tests", "extra with space"], 12),
            (["extra with space", "patients", "tests"], 12),
            ("extra with space", 2),
            (None, 12),
        ],
    )
    def test_groupby(self, groupby, length):
        ds = Dataset(subset_index=_create_valid_index())
        grouped = ds.groupby(groupby)

        assert id(grouped) != id(ds)
        assert grouped.groupby_cols == groupby
        assert grouped.shape[0] == length

        if isinstance(groupby, list):
            index_level = len(groupby)
        elif groupby is None:
            index_level = len(grouped.index.columns)
        else:
            index_level = 1
        assert grouped.grouped_index.index.nlevels == index_level
        assert len(grouped.groups) == length
        if index_level > 1:
            assert len(grouped.groups[0]) == index_level
        else:
            assert isinstance(grouped.groups[0], (str, int))

    def test_groupby_error(self):
        ds = Dataset(subset_index=_create_valid_index())
        with pytest.raises(KeyError, match="You can only groupby columns that are part of the index columns"):
            ds.groupby("invalid_column_name")

    @pytest.mark.parametrize(
        "index,bool_map,kwargs,what_to_expect,expect_error",
        [
            (
                None,
                None,
                None,
                "At least one of `groups`, `selected_keys`, `index`, `bool_map` or kwarg must not be None!",
                ValueError,
            ),
            (
                _create_valid_index(),
                _create_random_bool_map(12, 432),
                None,
                "Only one of `groups`, `selected_keys`, `index`, `bool_map` or kwarg can be set!",
                ValueError,
            ),
        ],
    )
    def test_get_subset_generic_errors(self, index, bool_map, kwargs, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index())
        kwargs = kwargs or {}

        with pytest.raises(expect_error, match=what_to_expect):
            df.get_subset(index=index, bool_map=bool_map, **kwargs)

    @pytest.mark.parametrize(
        "index,what_to_expect",
        [
            (
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra with space"],
                ),
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
        ],
    )
    def test_get_subset_index_valid_input(self, index, what_to_expect):
        pd.testing.assert_frame_equal(
            left=what_to_expect, right=Dataset(subset_index=_create_valid_index()).get_subset(index=index).index
        )

    @pytest.mark.parametrize(
        "groups,what_to_expect",
        [
            (
                [("patient_1", "test_2", "0")],
                _create_valid_index(
                    {"patient_1": {"a": ["test_2"], "b": ["0"]}},
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                [("patient_1", "test_2", "0"), ("patient_1", "test_2", "1")],
                _create_valid_index(
                    {"patient_1": {"a": ["test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
        ],
    )
    def test_get_subset_groups_valid_input(self, groups, what_to_expect):
        pd.testing.assert_frame_equal(
            left=what_to_expect, right=Dataset(subset_index=_create_valid_index()).get_subset(groups=groups).index
        )

    @pytest.mark.parametrize(
        "groups,what_to_expect",
        [
            (
                [("patient_1", "test_2")],
                _create_valid_index(
                    {"patient_1": {"a": ["test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                [("patient_1", "test_1"), ("patient_1", "test_2")],
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
        ],
    )
    def test_get_subset_groups_valid_input_grouped(self, groups, what_to_expect):
        pd.testing.assert_frame_equal(
            left=what_to_expect,
            right=Dataset(subset_index=_create_valid_index())
            .groupby(["patients", "tests"])
            .get_subset(groups=groups)
            .index,
        )

    @pytest.mark.parametrize(
        "index,what_to_expect",
        [
            (
                pd.DataFrame(),
                "Provided index is not formatted correctly",
            ),
        ],
    )
    def test_get_subset_index_error_input(self, index, what_to_expect):
        with pytest.raises(ValueError, match=what_to_expect):
            Dataset(subset_index=_create_valid_index()).get_subset(index=index)

    @pytest.mark.parametrize(
        "bool_map,what_to_expect",
        [
            (
                _create_random_bool_map(12, 68752868),
                _create_valid_index()[_create_random_bool_map(12, 68752868)].reset_index(drop=True),
            ),
        ],
    )
    def test_get_subset_bool_map_valid_input(self, bool_map, what_to_expect):
        pd.testing.assert_frame_equal(
            left=what_to_expect, right=Dataset(subset_index=_create_valid_index()).get_subset(bool_map=bool_map).index
        )

    @pytest.mark.parametrize(
        "bool_map,what_to_expect",
        [
            (
                _create_random_bool_map(12, 68752868)[:-1],
                "Parameter bool_map must have length",
            ),
        ],
    )
    def test_get_subset_bool_map_error_input(self, bool_map, what_to_expect):
        with pytest.raises(ValueError, match=what_to_expect):
            Dataset(subset_index=_create_valid_index()).get_subset(bool_map=bool_map)

    @pytest.mark.parametrize(
        "kwargs,what_to_expect",
        [
            (
                {"patients": ["patient_1", "patient_3"], "tests": ["test_2", "test_3"], "extra with space": ["0"]},
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_2"], "b": ["0"]},
                        "patient_3": {"a": ["test_2", "test_3"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
        ],
    )
    def test_get_subset_kwargs_valid_input(self, kwargs, what_to_expect):
        pd.testing.assert_frame_equal(
            left=what_to_expect, right=Dataset(subset_index=_create_valid_index()).get_subset(**kwargs).index
        )

    @pytest.mark.parametrize(
        "kwargs,what_to_expect",
        [
            (
                {"wrong": ["patient_1", "patient_3"], "tests": ["test_2", "test_3"], "extra with space": ["0"]},
                "Can not filter by key `wrong`!",
            ),
        ],
    )
    def test_get_subset_kwargs_error_input(self, kwargs, what_to_expect):
        with pytest.raises(KeyError, match=what_to_expect):
            Dataset(subset_index=_create_valid_index()).get_subset(**kwargs)

    @pytest.mark.parametrize(
        "subscript,groupby,what_to_expect",
        [
            (
                0,
                ["patients", "tests", "extra with space"],
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                [0, 1],
                "patients",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                        "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                [0, 4],
                ["patients", "tests"],
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_3": {"a": ["test_2"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
        ],
    )
    def test_getitem_valid_input(self, subscript, groupby, what_to_expect):
        pd.testing.assert_frame_equal(
            left=what_to_expect,
            right=Dataset(subset_index=_create_valid_index(), groupby_cols=groupby)[subscript].index,
        )

    @pytest.mark.parametrize(
        "subscript,select_lvl,what_to_expect",
        [
            (
                4,
                "patients",
                "out of bounds",
            ),
            (
                [0, 1, 4],
                "patients",
                "out of bounds",
            ),
        ],
    )
    def test_getitem_error_input(self, subscript, select_lvl, what_to_expect):
        with pytest.raises(IndexError, match=what_to_expect):
            _ = Dataset(subset_index=_create_valid_index(), groupby_cols=select_lvl)[subscript]

    @pytest.mark.parametrize(
        "groupby_level", (["patients"], ["patients", "tests"], ["patients", "tests", "extra with space"])
    )
    @pytest.mark.parametrize(
        "index,is_single_level",
        (
            (
                _create_valid_index(
                    {
                        "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
                [],
            ),
            (
                _create_valid_index(
                    {
                        "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
                [["patients"]],
            ),
            (
                _create_valid_index(
                    {
                        "patient_3": {"a": ["test_1"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
                [["patients"], ["patients", "tests"]],
            ),
            (
                _create_valid_index(
                    {
                        "patient_3": {"a": ["test_1"], "b": ["1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
                [["patients"], ["patients", "tests"], ["patients", "tests", "extra with space"], None],
            ),
        ),
    )
    def test_is_single(self, index, is_single_level, groupby_level):
        ds = Dataset(subset_index=index, groupby_cols=groupby_level)
        for gr in ["patients"], ["patients", "tests"], ["patients", "tests", "extra with space"], None:
            if gr in is_single_level:
                assert ds.is_single(gr) is True
                # Check that no error is raised
                ds.assert_is_single(gr, "test")

            else:
                assert ds.is_single(gr) is False
                with pytest.raises(ValueError):
                    ds.assert_is_single(gr, "test")
            if len(ds) == 1:
                assert ds.is_single_group() is True
                ds.assert_is_single_group("test")
            else:
                assert ds.is_single_group() is False
                with pytest.raises(ValueError):
                    ds.assert_is_single_group("test")

    def test_assert_is_single_error(self):
        index = _create_valid_index(
            {
                "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0", "1"]},
            },
            columns_names=["patients", "tests", "extra with space"],
        )
        ds = Dataset(subset_index=index)
        with pytest.raises(ValueError) as e:
            ds.assert_is_single(None, "test_name")

        assert "test_name" in str(e)

    def test_create_index_call(self):
        with pytest.raises(NotImplementedError):
            _ = Dataset().index

    @pytest.mark.parametrize(
        "n_splits,groupby,what_to_expect",
        [
            (
                5,
                ["patients", "tests"],
                (
                    _create_valid_index(
                        {
                            "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                            "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0", "1"]},
                        },
                        columns_names=["patients", "tests", "extra with space"],
                    ),
                    _create_valid_index(
                        {
                            "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                        },
                        columns_names=["patients", "tests", "extra with space"],
                    ),
                ),
            ),
        ],
    )
    def test_dataset_with_kfold_valid_input(self, n_splits, groupby, what_to_expect):
        df = Dataset(subset_index=_create_valid_index(), groupby_cols=groupby)
        train, test = next(KFold(n_splits=n_splits).split(df))
        pd.testing.assert_frame_equal(left=what_to_expect[0], right=df[train].index)
        pd.testing.assert_frame_equal(left=what_to_expect[1], right=df[test].index)

    @pytest.mark.parametrize(
        "n_splits,select_lvl,what_to_expect",
        [
            (13, "extra with space", "Cannot have number of splits"),
        ],
    )
    def test_dataset_with_kfold_error_input(self, n_splits, select_lvl, what_to_expect):
        with pytest.raises(ValueError, match=what_to_expect):
            next(KFold(n_splits=n_splits).split(Dataset(subset_index=_create_valid_index(), groupby_cols=select_lvl)))

    @pytest.mark.parametrize(
        "select_lvl,what_to_expect",
        [
            (
                "patients",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                ["patients", "tests"],
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                ["patients", "tests", "extra with space"],
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
        ],
    )
    def test_iter(self, select_lvl, what_to_expect):
        pd.testing.assert_frame_equal(
            left=what_to_expect,
            right=next(Dataset(subset_index=_create_valid_index(), groupby_cols=select_lvl).__iter__()).index,
        )

    @pytest.mark.parametrize(
        "level,what_to_expect",
        [
            (
                "extra with space",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0"]},
                        "patient_2": {"a": ["test_1"], "b": ["0"]},
                        "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                "tests",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_3": {"a": ["test_1"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
            (
                "patients",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra with space"],
                ),
            ),
        ],
    )
    def test_iter_level_valid_input(self, level, what_to_expect):
        # Create the generator
        values = list(Dataset(subset_index=_create_valid_index()).iter_level(level=level))
        pd.testing.assert_frame_equal(left=what_to_expect, right=values[0].index)

    @pytest.mark.parametrize(
        "level,what_to_expect",
        [
            (
                "wrong",
                "`level` must be one of",
            ),
        ],
    )
    def test_iter_level_error_input(self, level, what_to_expect):
        with pytest.raises(ValueError, match=what_to_expect):
            next(Dataset(subset_index=_create_valid_index()).iter_level(level=level))

    @pytest.mark.parametrize(
        "groupby,groupby_labels,unique",
        (
            (None, "patients", 3),
            (None, ["patients"], 3),
            (None, ["patients", "extra with space"], 6),
            (["patients", "tests"], ["patients"], 3),
        ),
    )
    def test_create_group_labels(self, groupby, groupby_labels, unique):
        ds = Dataset(subset_index=_create_valid_index(), groupby_cols=groupby)
        labels = ds.create_group_labels(groupby_labels)

        assert len(labels) == len(ds)
        assert len(set(labels)) == unique

    def test_create_group_labels_error_groupby(self):
        ds = Dataset(subset_index=_create_valid_index(), groupby_cols="patients")
        with pytest.raises(KeyError, match="When using `create_group_labels` with a grouped dataset"):
            ds.create_group_labels("recording")

    def test_create_group_labels_error_not_in_index(self):
        ds = Dataset(subset_index=_create_valid_index())
        with pytest.raises(KeyError, match="The selected label columns"):
            ds.create_group_labels("something_wrong")

    @pytest.mark.parametrize("groupby", ["patients", "tests", ["patients", "tests"], None])
    def test_group(self, groupby):
        ds = Dataset(subset_index=_create_valid_index(), groupby_cols=groupby)
        group = ds[0].group

        assert group == ds[0].groups[0]
        if isinstance(groupby, str):
            assert group == ds.index.iloc[0][groupby]
        elif groupby is None:
            assert group == tuple(ds.index.iloc[0])
        else:
            assert group._fields == tuple(groupby)
            assert group == tuple(ds.index.iloc[0][groupby])

    def test_group_with_invalid_col_name(self):
        ds = Dataset(subset_index=_create_valid_index())
        group = ds[0].group

        # the named tuple will rename columns that are invalid, therefore instead of `extra with spaces`,
        # it should be `_2`
        assert group._fields == ("patients", "tests", "_2")

    def test_group_only_for_single_group(self):
        ds = Dataset(subset_index=_create_valid_index())

        with pytest.raises(ValueError, match="only a single group left"):
            _ = ds.group
