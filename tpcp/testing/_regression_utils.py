"""Utils to perform snapshot tests easily.

This is inspired by github.com/syrusakbary/snapshottest.
Note that it can not be used in combination with this module!
"""
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


class SnapshotNotFoundError(Exception):
    pass


class PyTestSnapshotTest:
    """Perform snapshot tests in pytest.

    This supports standard datatypes and scientific datatypes like numpy arrays and pandas DataFrames.

    To use this in your tests, add the following lines to your conftest.py:

    .. code-block:: python

        @pytest.fixture()
        def snapshot(request):
            with PyTestSnapshotTest(request) as snapshot_test:
                yield snapshot_test


        def pytest_addoption(parser):
            group = parser.getgroup("snapshottest")
            group.addoption(
                "--snapshot-update",
                action="store_true",
                default=False,
                dest="snapshot_update",
                help="Update the snapshots.",
            )

    This will register the snapshot fixture that you can use in your tests.
    Further, it will register the `--snapshot-update` commandline flag, which you can use to update the snapshots.

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
            value.to_json(self._file_name_json, indent=4, orient="table", date_unit="ns")
        elif isinstance(value, np.ndarray):
            np.savetxt(self._file_name_csv, value, delimiter=",")
        elif isinstance(value, str):
            with Path(self._file_name_txt).open("w") as f:
                f.write(value)
        else:
            raise TypeError(f"The dtype {type(value)} is not supported for snapshot testing")

    def _retrieve(self, dtype):
        if dtype == pd.DataFrame:
            filename = self._file_name_json
            if not filename.is_file():
                raise SnapshotNotFoundError
            return pd.read_json(filename, orient="table")
        if dtype == np.ndarray:
            filename = self._file_name_csv
            if not filename.is_file():
                raise SnapshotNotFoundError
            return np.genfromtxt(filename, delimiter=",")
        if dtype == str:
            filename = self._file_name_txt
            if not filename.is_file():
                raise SnapshotNotFoundError
            with Path(self._file_name_txt).open() as f:
                value = f.read()
            return value
        raise ValueError(f"The dtype {dtype} is not supported for snapshot testing")

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
        self.curr_snapshot = name or str(self.curr_snapshot_number)
        if self._update:
            self._store(value)
        else:
            value_dtype = type(value)
            try:
                prev_snapshot = self._retrieve(value_dtype)
            except SnapshotNotFoundError:
                self._store(value)  # first time this test has been seen
            except:
                raise
            else:
                if isinstance(value, pd.DataFrame):
                    assert_frame_equal(value, prev_snapshot, **kwargs)
                elif isinstance(value, np.ndarray):
                    np.testing.assert_array_almost_equal(value, prev_snapshot, **kwargs)
                elif isinstance(value, str):
                    # Display the string diff line by line as part of error message using difflib
                    import difflib

                    diff = difflib.ndiff(value.splitlines(keepends=True), prev_snapshot.splitlines(keepends=True))
                    diff = "".join(diff)
                    assert value == prev_snapshot, diff
                else:
                    raise TypeError(f"The dtype {value_dtype} is not supported for snapshot testing")

        self.curr_snapshot_number += 1
