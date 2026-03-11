"""Utils to perform snapshot tests easily.

This is inspired by github.com/syrusakbary/snapshottest.
Note that it can not be used in combination with this module!
"""

import difflib
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


class SnapshotNotFoundError(Exception):
    pass


class SnapshotAssertionError(AssertionError):
    pass


class PyTestSnapshotTest:
    """Perform snapshot tests in pytest.

    This supports standard datatypes and scientific datatypes like numpy arrays and pandas DataFrames.

    This plugin will be automatically registered when you install tpcp.
    It adds the `snapshot` fixture to your tests.
    Further, it will register the `--snapshot-update` commandline flag, which you can use to update the snapshots.
    You can also run pytest with the `--snapshot-only-check` flag to fail if a snapshot file is not found.
    Without that flag, missing snapshots will be automatically created.

    To use the fixture in your tests, simply add it as a parameter to your test function:

    .. code-block:: python

        def test_my_test(snapshot):
            result = my_calculation()
            snapshot.assert_match(result, "my_result_1")

    This will store the result of `my_calculation()` in a snapshot file in a folder called `snapshot` in the same folder
    as the test file.
    The name of the snapshot file will be the name of the test function, suffixed with `_my_result_1`.
    When the test is run again, the result will be compared to the stored snapshot.

    To update a snapshot, either delete the snapshot file and manually run the test again or run pytest with the
    `--snapshot-update` flag.

    """

    curr_snapshot: str

    def __init__(self, request=None) -> None:
        self.request = request
        self.curr_snapshot_number = 0
        super().__init__()

    @property
    def _update(self):
        return self.request.config.option.snapshot_update

    @property
    def _only_check(self):
        return self.request.config.option.snapshot_only_check

    @property
    def _module(self):
        return Path(self.request.node.fspath.strpath).parent

    @property
    def _snapshot_folder(self):
        return self._module / "snapshot"

    @property
    def _file_name_json(self):
        return self._snapshot_folder / f"{self._test_name}.json"

    @property
    def _file_name_csv(self):
        return self._snapshot_folder / f"{self._test_name}.csv"

    @property
    def _file_name_txt(self):
        return self._snapshot_folder / f"{self._test_name}.txt"

    def _snapshot_file(self, dtype):
        if dtype is pd.DataFrame:
            return self._file_name_json
        if dtype is np.ndarray:
            return self._file_name_csv
        if dtype is str:
            return self._file_name_txt
        raise ValueError(f"The dtype {dtype} is not supported for snapshot testing")

    @property
    def _test_name(self):
        cls_name = getattr(self.request.node.cls, "__name__", "")
        flattened_node_name = re.sub(r"\s+", " ", self.request.node.name.replace(r"\n", " "))
        return "{}{}_{}".format(f"{cls_name}." if cls_name else "", flattened_node_name, self.curr_snapshot)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _store(self, value):
        self._snapshot_folder.mkdir(parents=True, exist_ok=True)
        if isinstance(value, pd.DataFrame):
            # non-unique indices are not stored as index in the snapshot
            if not value.index.is_unique:
                raise ValueError(
                    "Input DataFrame has non-unique index. This is currently not supported for snapshot testing."
                    "Consider calling `reset_index()` before passing data to `assert_match`."
                )
            # "index" is not accepted as index or column name
            if "index" in value.columns or "index" in value.index.names:
                raise ValueError(
                    "Input DataFrame has a column named 'index'. This is currently not supported for snapshot testing."
                    "Consider renaming this column before passing data to `assert_match`."
                )
            # prevent dtype assertion errors
            self._check_for_non_default_dtypes(value)
            # convert datetime columns to a format that can be written and read back from json without errors
            value = self._sanitize_datetime_entries(value)

            value.to_json(self._file_name_json, indent=4, orient="table", date_unit="ns", index=True)
        elif isinstance(value, np.ndarray):
            np.savetxt(self._file_name_csv, value, delimiter=",")
        elif isinstance(value, str):
            with Path(self._file_name_txt).open("w") as f:
                f.write(value)
        else:
            raise TypeError(f"The dtype {type(value)} is not supported for snapshot testing")

    def _retrieve(self, dtype):
        if dtype is pd.DataFrame:
            filename = self._file_name_json
            if not filename.is_file():
                raise SnapshotNotFoundError
            return pd.read_json(filename, orient="table")
        if dtype is np.ndarray:
            filename = self._file_name_csv
            if not filename.is_file():
                raise SnapshotNotFoundError
            return np.genfromtxt(filename, delimiter=",")
        if dtype is str:
            filename = self._file_name_txt
            if not filename.is_file():
                raise SnapshotNotFoundError
            with Path(self._file_name_txt).open() as f:
                value = f.read()
            return value
        raise ValueError(f"The dtype {dtype} is not supported for snapshot testing")

    @staticmethod
    def _format_string_difference(value: str, prev_snapshot: str) -> str:
        diff = "".join(
            difflib.unified_diff(
                prev_snapshot.splitlines(keepends=True),
                value.splitlines(keepends=True),
                fromfile="stored",
                tofile="current",
            )
        )
        if diff:
            return diff
        return f"stored={prev_snapshot!r}\ncurrent={value!r}"

    def _format_mismatch(self, dtype, value, prev_snapshot, error: Optional[AssertionError] = None) -> str:
        if dtype is str:
            details = self._format_string_difference(value, prev_snapshot)
        else:
            details = str(error) if error else "The snapshot content does not match the current value."

        return (
            f"Snapshot mismatch for '{self._test_name}'.\n"
            f"Snapshot file: {self._snapshot_file(dtype)}\n\n"
            f"{details}\n\n"
            "Run pytest with --snapshot-update to accept the new snapshot."
        )

    def _compare(self, value, prev_snapshot, value_dtype, **kwargs):
        __tracebackhide__ = True

        if isinstance(value, pd.DataFrame):
            # convert datetime columns to match with the stored format
            value = self._sanitize_datetime_entries(value)
            assert_frame_equal(value, prev_snapshot, **kwargs)
            return
        if isinstance(value, np.ndarray):
            np.testing.assert_array_almost_equal(value, prev_snapshot, **kwargs)
            return
        if isinstance(value, str):
            if value != prev_snapshot:
                raise SnapshotAssertionError(self._format_mismatch(value_dtype, value, prev_snapshot))
            return
        raise TypeError(f"The dtype {value_dtype} is not supported for snapshot testing")

    def assert_match(self, value: Union[str, pd.DataFrame, np.ndarray], name: Optional[str] = None, **kwargs):
        """Assert that the value matches the snapshot.

        This compares the value with a stored snapshot of the same name.
        If no snapshot exists, it will be created.

        The snapshot name is automatically generated from the test name and the `name` parameter passed to this
        function.
        If no name is passed, the name will be suffixed with a number, starting at 0.
        If you have multiple snapshots in one test, we highly recommend to pass a name to this function.
        Otherwise, changing the order of the snapshots will break your tests.

        Parameters
        ----------
        value
            The value to compare with the snapshot.
            We support strings, numpy arrays and pandas DataFrames.
            For other datatypes like floats or short lists, we recommend to just use the standard pytest assertions
            and hardcode the expected value.
        name
            Optional name suffix of the snapshot-file.
            If not provided the name will be suffixed with a number, starting at 0.
        kwargs
            Additional keyword arguments passed to the comparison function.
            This is only supported for DataFrames and numpy arrays.
            There they will be passed to `assert_frame_equal` and `assert_array_almost_equal` respectively.

        """
        __tracebackhide__ = True
        self.curr_snapshot = name or str(self.curr_snapshot_number)
        if self._update:
            self._store(value)
        else:
            value_dtype = type(value)
            try:
                prev_snapshot = self._retrieve(value_dtype)
            except SnapshotNotFoundError as e:
                if self._only_check:
                    raise SnapshotNotFoundError(
                        "No corresponding snapshot file could be found. "
                        "Run pytest without the--snapshot-only-check flag to create a new "
                        "snapshot and store it in git"
                    ) from e
                self._store(value)  # first time this test has been seen
            except:
                raise
            else:
                try:
                    self._compare(value, prev_snapshot, value_dtype, **kwargs)
                except SnapshotAssertionError:
                    raise
                except AssertionError as e:
                    raise SnapshotAssertionError(self._format_mismatch(value_dtype, value, prev_snapshot, e)) from None

        self.curr_snapshot_number += 1

    @staticmethod
    def _check_for_non_default_dtypes(df: pd.DataFrame):
        df.select_dtypes(include=[float]).columns.equals(df.select_dtypes(include=["float32"]).columns)
        float_cols = df.select_dtypes(include=[float]).columns
        default_float_cols = df.select_dtypes(include=["float64"]).columns
        if not float_cols.equals(default_float_cols):
            raise ValueError(
                f"DataFrame contains non-default float dtypes: {df[float_cols].dtypes}, which are not supported for "
                "snapshot testing. Consider converting them to 'float64' or to use the flag `check_dtype=False`."
            )
        int_cols = df.select_dtypes(include=[int]).columns
        default_int_cols = df.select_dtypes(include=["int64"]).columns
        if not int_cols.equals(default_int_cols):
            raise ValueError(
                f"DataFrame contains non-default int dtypes: {df[int_cols].dtypes}, which are not supported for "
                "snapshot testing. Consider converting them to 'int64' or to use the flag `check_dtype=False`."
            )

    @staticmethod
    def _sanitize_datetime_entries(df: pd.DataFrame) -> pd.DataFrame:
        sanitized_df = df.copy()
        datetime_types = ["datetime", "datetime64", "datetime64[ns]", "datetimetz"]
        # check columns
        datetime_cols = sanitized_df.select_dtypes(include=datetime_types).columns
        # remove timezone information to prevent this read-write issue:
        # https://github.com/pandas-dev/pandas/issues/53473
        for col in datetime_cols:
            col_values = sanitized_df[col]
            if isinstance(col_values.dtype, pd.DatetimeTZDtype):
                col_values = col_values.dt.tz_localize(None)
            sanitized_df[col] = col_values.astype("datetime64[ns]")
        # check index
        if sanitized_df.index.nlevels > 1:
            for level in range(sanitized_df.index.nlevels):
                level_values = sanitized_df.index.levels[level]
                if level_values.inferred_type in datetime_types:
                    if isinstance(level_values.dtype, pd.DatetimeTZDtype):
                        level_values = level_values.tz_localize(None)
                    sanitized_df.index = sanitized_df.index.set_levels(
                        level_values.astype("datetime64[ns]"), level=level
                    )
        elif sanitized_df.index.inferred_type in datetime_types:
            index_values = sanitized_df.index
            if isinstance(index_values.dtype, pd.DatetimeTZDtype):
                index_values = index_values.tz_localize(None)
            sanitized_df.index = index_values.astype("datetime64[ns]")
        return sanitized_df


@pytest.fixture
def snapshot(request):
    with PyTestSnapshotTest(request) as snapshot_test:
        yield snapshot_test


def pytest_addoption(parser):
    group = parser.getgroup("tpcp_snapshots")
    group.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        dest="snapshot_update",
        help="Update the snapshots.",
    )
    group.addoption(
        "--snapshot-only-check",
        action="store_true",
        default=False,
        dest="snapshot_only_check",
        help="Run as normal, but fail if a snapshot file is not found. This is usefull for CI runs.",
    )
