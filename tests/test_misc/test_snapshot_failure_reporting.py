"""Integration tests for snapshot mismatch reporting."""

from textwrap import dedent

pytest_plugins = ("pytester",)


def test_snapshot_failure_reports_dataframe_details(pytester):
    """A dataframe mismatch should include the detailed pandas assertion output."""
    test_file = pytester.path / "test_snapshot_failure.py"
    test_file.write_text(
        dedent(
            """
            import pandas as pd

            def test_df(snapshot):
                df = pd.DataFrame({"a": [1, 3]}).set_index(pd.Index([1, 2], name="idx"))
                snapshot.assert_match(df, "df")
            """
        ),
        encoding="utf-8",
    )

    result = pytester.runpytest_subprocess("--snapshot-update")
    result.assert_outcomes(passed=1)

    test_file.write_text(
        dedent(
            """
            import pandas as pd

            def test_df(snapshot):
                df = pd.DataFrame({"a": [1, 4]}).set_index(pd.Index([1, 2], name="idx"))
                snapshot.assert_match(df, "df")
            """
        ),
        encoding="utf-8",
    )

    result = pytester.runpytest_subprocess("-vv")
    result.assert_outcomes(failed=1)
    stdout = result.stdout.str()

    assert "Snapshot mismatch for 'test_df_df'." in stdout
    assert "Snapshot file:" in stdout
    assert 'DataFrame.iloc[:, 0] (column name="a") values are different (50.0 %)' in stdout
    assert "[left]:  [1, 4]" in stdout
    assert "[right]: [1, 3]" in stdout
    assert "Run pytest with --snapshot-update to accept the new snapshot." in stdout


def test_snapshot_failure_reports_string_diff(pytester):
    """A string mismatch should include a readable unified diff."""
    test_file = pytester.path / "test_snapshot_failure.py"
    test_file.write_text(
        dedent(
            """
            def test_str(snapshot):
                snapshot.assert_match("line1\\nline2\\n", "txt")
            """
        ),
        encoding="utf-8",
    )

    result = pytester.runpytest_subprocess("--snapshot-update")
    result.assert_outcomes(passed=1)

    test_file.write_text(
        dedent(
            """
            def test_str(snapshot):
                snapshot.assert_match("line1\\nLINE2\\n", "txt")
            """
        ),
        encoding="utf-8",
    )

    result = pytester.runpytest_subprocess("-vv")
    result.assert_outcomes(failed=1)
    stdout = result.stdout.str()

    assert "Snapshot mismatch for 'test_str_txt'." in stdout
    assert "--- stored" in stdout
    assert "+++ current" in stdout
    assert "-line2" in stdout
    assert "+LINE2" in stdout
