import pandas as pd
import pytest


class TestSnapshot:
    def test_snapshot_df_non_default_type_error(self, snapshot):
        df = pd.DataFrame(
            {"dummy_index": [1, 2], "col_int": [1, 2], "col_str": ["a", "b"], "col_float": [2.0, 1 / 3]}
        ).set_index("dummy_index")
        df = df.astype({"col_int": "int32", "col_str": "string", "col_float": "float32"})
        with pytest.raises(ValueError):
            snapshot.assert_match(df, "dummy_df")
        df = df.astype({"col_float": "float64", "col_int": "int64"})
        snapshot.assert_match(df, "dummy_df_default_types")

    def test_snapshot_df_multiindex(self, snapshot):
        df = pd.DataFrame([1, 2], columns=["example_col"])
        df.index = pd.MultiIndex.from_tuples([(1, 2), ("a", "b")], names=["index_level_a", "index_level_b"])
        snapshot.assert_match(df, "dummy_df_multiindex")

    @pytest.mark.parametrize("freq", ["s", "h", "t", "d", "ms", "us", "ns"])
    def test_snapshot_df_timeseries_without_tz(self, freq, snapshot):
        df_len = 10
        df = pd.DataFrame(
            {
                "dummy_index": pd.date_range("2021-01-01", periods=df_len, freq="ns"),
                "col_int": range(df_len),
                "col_date": pd.date_range("2021-01-01", periods=df_len, freq="ns"),
            }
        ).set_index("dummy_index")
        snapshot.assert_match(df, "dummy_df_with_timeseries")

    @pytest.mark.parametrize("freq", ["s", "h", "t", "d", "ms", "us", "ns"])
    def test_snapshot_df_timeseries_with_tz(self, freq, snapshot):
        # TODO
        df_len = 10
        df = pd.DataFrame(
            {
                "dummy_index": pd.date_range("2021-01-01", periods=df_len, freq=freq, tz="Europe/Berlin"),
                "col_int": range(df_len),
                "col_date": pd.date_range("2021-01-01", periods=df_len, freq=freq, tz="Europe/Berlin"),
            }
        ).set_index("dummy_index")
        snapshot.assert_match(df, "dummy_df_with_timeseries")

    def test_snapshot_df_index_not_unique_error(self, snapshot):
        df = pd.DataFrame([[1, 2], [1, 2]], columns=["dummy_index", "column"]).set_index("dummy_index")
        with pytest.raises(ValueError):
            snapshot.assert_match(df, "dummy_df_index_not_unique")
        snapshot.assert_match(df.reset_index(), "dummy_df_index_not_unique_reset_index")

    def test_snapshot_df_multiindex_not_unique_error(self, snapshot):
        df = pd.DataFrame([1, 2], columns=["example_col"])
        df.index = pd.MultiIndex.from_tuples([(2, 2), (2, 2)], names=["index_level_a", "index_level_b"])
        with pytest.raises(ValueError):
            snapshot.assert_match(df, "dummy_df_multiindex_not_unique")
        snapshot.assert_match(df.reset_index(), "dummy_df_multiindex_not_unique_reset_index")

    def test_snapshot_df_index_name_error(self, snapshot):
        df = pd.DataFrame([[1, 1], [2, 2]], columns=["index", "column"]).set_index("index")
        with pytest.raises(ValueError):
            snapshot.assert_match(df, "dummy_df_index_name_error")
        with pytest.raises(ValueError):
            snapshot.assert_match(df.reset_index(), "dummy_df_index_name_error_reset_index")
