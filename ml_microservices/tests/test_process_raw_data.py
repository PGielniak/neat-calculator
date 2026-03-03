"""
Unit tests for shared/process_raw_data.py
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from shared.process_raw_data import (
    validate_directory,
    validate_labels_csv,
    remove_duplicates,
    resample_data,
    create_sliding_windows,
    extract_features_from_windows,
    rename_features,
    filter_features_to_match_kaggle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENSOR_COLS = [
    "accelerometerX",
    "accelerometerY",
    "accelerometerZ",
    "gyroscopeX",
    "gyroscopeY",
    "gyroscopeZ",
]


def _make_sensor_record(**overrides):
    """Return a minimal valid sensor dict."""
    base = {
        "accelerometerX": 0.1,
        "accelerometerY": 0.2,
        "accelerometerZ": 9.8,
        "gyroscopeX": 0.01,
        "gyroscopeY": 0.02,
        "gyroscopeZ": 0.03,
        "timestamp": 1_700_000_000_000,  # ms, modern epoch
        "timestampNanos": 1_700_000_000_000_000_000,  # ns
        "label": "WALKING",
    }
    base.update(overrides)
    return base


def _make_resampled_df(n: int = 256, label: str = "WALKING", start_ns: int = 1_700_000_000_000_000_000) -> pd.DataFrame:
    """Build a DataFrame that looks like output of remove_duplicates (DatetimeIndex, 50 Hz)."""
    step_ns = 20_000_000  # 20 ms = 50 Hz
    timestamps_ns = [start_ns + i * step_ns for i in range(n)]
    timestamps_ms = [ts // 1_000_000 for ts in timestamps_ns]
    dt_index = pd.to_datetime(timestamps_ns, unit="ns")
    rng = np.random.default_rng(0)
    data = {col: rng.standard_normal(n) for col in SENSOR_COLS}
    data["label"] = label
    data["timestamp"] = timestamps_ms
    data["timestampNanos"] = timestamps_ns
    df = pd.DataFrame(data, index=dt_index)
    df.index.name = "t"
    return df


# ---------------------------------------------------------------------------
# validate_directory
# ---------------------------------------------------------------------------

class TestValidateDirectory:
    def test_valid_directory_does_not_raise(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([_make_sensor_record()]))
        validate_directory(str(tmp_path))

    def test_nonexistent_directory_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            validate_directory("/nonexistent/path/xyz")

    def test_file_path_raises_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            validate_directory(str(f))

    def test_empty_directory_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            validate_directory(str(tmp_path))

    def test_all_files_skipped_raises_value_error(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([_make_sensor_record()]))
        with pytest.raises(ValueError, match="skipped"):
            validate_directory(str(tmp_path), skipped_files=["data.json"])

    def test_invalid_json_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not-valid-json")
        with pytest.raises(ValueError):
            validate_directory(str(tmp_path))

    def test_non_json_file_is_added_to_skipped(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("some text")
        # Should not raise — non-JSON files are silently added to skipped
        skipped: list[str] = []
        validate_directory(str(tmp_path), skipped_files=skipped)
        assert "data.txt" in skipped


# ---------------------------------------------------------------------------
# validate_labels_csv
# ---------------------------------------------------------------------------

class TestValidateLabelsCSV:
    VALID_LABELS = ["STANDING", "SITTING", "LAYING", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]

    def test_valid_dataframe_passes_through(self):
        df = pd.DataFrame({"Activity": self.VALID_LABELS})
        result = validate_labels_csv(df)
        assert len(result) == len(df)

    def test_missing_column_raises_value_error(self):
        df = pd.DataFrame({"NotActivity": ["WALKING"]})
        with pytest.raises(ValueError, match="not found"):
            validate_labels_csv(df)

    def test_invalid_labels_are_removed(self):
        df = pd.DataFrame({"Activity": ["WALKING", "FLYING", "RUNNING", "STANDING"]})
        result = validate_labels_csv(df)
        assert "FLYING" not in result["Activity"].values
        assert "RUNNING" not in result["Activity"].values
        assert "WALKING" in result["Activity"].values

    def test_custom_label_column(self):
        df = pd.DataFrame({"label": ["WALKING", "SITTING"]})
        result = validate_labels_csv(df, label_column="label")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_removes_duplicate_timestamps(self):
        ts_ns = 1_700_000_000_000_000_000
        ts_ms = ts_ns // 1_000_000
        records = [
            {**{c: 1.0 for c in SENSOR_COLS}, "label": "WALKING", "timestamp": ts_ms, "timestampNanos": ts_ns},
            {**{c: 3.0 for c in SENSOR_COLS}, "label": "WALKING", "timestamp": ts_ms, "timestampNanos": ts_ns},
        ]
        df = pd.DataFrame(records)
        result = remove_duplicates(df, sensor_cols=SENSOR_COLS)
        # After dedup there should be only one row for that timestamp
        assert len(result) == 1

    def test_unique_timestamps_preserved(self):
        ts_ns_base = 1_700_000_000_000_000_000
        step_ns = 20_000_000
        records = []
        for i in range(5):
            ts_ns = ts_ns_base + i * step_ns
            records.append(
                {**{c: float(i) for c in SENSOR_COLS},
                 "label": "WALKING",
                 "timestamp": ts_ns // 1_000_000,
                 "timestampNanos": ts_ns}
            )
        df = pd.DataFrame(records)
        result = remove_duplicates(df, sensor_cols=SENSOR_COLS)
        assert len(result) == 5

    def test_returns_datetime_index(self):
        ts_ns = 1_700_000_000_000_000_000
        records = [{**{c: 0.5 for c in SENSOR_COLS}, "label": "WALKING",
                    "timestamp": ts_ns // 1_000_000, "timestampNanos": ts_ns}]
        df = pd.DataFrame(records)
        result = remove_duplicates(df, sensor_cols=SENSOR_COLS)
        assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# resample_data
# ---------------------------------------------------------------------------

class TestResampleData:
    def test_output_columns_contain_sensor_cols(self):
        df = _make_resampled_df(n=256)
        result = resample_data(df, target_freq=50, sensor_cols=SENSOR_COLS)
        for col in SENSOR_COLS:
            assert col in result.columns

    def test_output_is_approximately_target_frequency(self):
        df = _make_resampled_df(n=300)
        result = resample_data(df, target_freq=50, sensor_cols=SENSOR_COLS)
        # At 50 Hz over ~6 seconds we expect ~300 rows
        assert len(result) > 100

    def test_returns_dataframe(self):
        df = _make_resampled_df(n=256)
        result = resample_data(df, target_freq=50, sensor_cols=SENSOR_COLS)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# create_sliding_windows
# ---------------------------------------------------------------------------

class TestCreateSlidingWindows:
    def test_basic_window_creation(self):
        df = _make_resampled_df(n=300)
        _, X, y, ts = create_sliding_windows(df, window_size=128, step_size=64, sensor_cols=SENSOR_COLS)
        # With 300 samples, window=128, step=64: we get at least 2 windows
        assert X.shape[0] >= 2
        assert X.shape[1] == 128
        assert X.shape[2] == len(SENSOR_COLS)
        assert len(y) == X.shape[0]
        assert len(ts) == X.shape[0]

    def test_insufficient_data_raises_value_error(self):
        """When no windows can be formed, np.stack raises ValueError on empty list."""
        df = _make_resampled_df(n=50)
        with pytest.raises(ValueError):
            create_sliding_windows(df, window_size=128, step_size=64, sensor_cols=SENSOR_COLS)

    def test_returns_correct_types(self):
        df = _make_resampled_df(n=300)
        result_df, X, y, ts = create_sliding_windows(df, window_size=128, step_size=64, sensor_cols=SENSOR_COLS)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(ts, np.ndarray)


# ---------------------------------------------------------------------------
# extract_features_from_windows
# ---------------------------------------------------------------------------

class TestExtractFeaturesFromWindows:
    def test_returns_dataframe_with_activity_column(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((3, 128, 6))
        y = np.array(["WALKING", "SITTING", "STANDING"])
        ts = np.array([1_700_000_000_000, 1_700_000_000_020, 1_700_000_000_040])
        df_placeholder = pd.DataFrame()
        result = extract_features_from_windows(df_placeholder, X, y, ts)
        assert isinstance(result, pd.DataFrame)
        assert "label" in result.columns
        assert "timestamp" in result.columns

    def test_row_count_matches_windows(self):
        rng = np.random.default_rng(11)
        n = 4
        X = rng.standard_normal((n, 128, 6))
        y = np.array(["WALKING"] * n)
        ts = np.arange(n, dtype=float)
        result = extract_features_from_windows(pd.DataFrame(), X, y, ts)
        assert len(result) == n


# ---------------------------------------------------------------------------
# rename_features
# ---------------------------------------------------------------------------

class TestRenameFeatures:
    def test_label_renamed_to_activity(self):
        df = pd.DataFrame({"tBodyAcc-mean()-X": [1.0], "label": ["WALKING"]})
        result = rename_features(df)
        assert "Activity" in result.columns
        assert "label" not in result.columns

    def test_subject_renamed(self):
        df = pd.DataFrame({"feature": [1.0], "subject": ["S1"]})
        result = rename_features(df)
        assert "Subject" in result.columns
        assert "subject" not in result.columns

    def test_other_columns_unchanged(self):
        df = pd.DataFrame({"feature_A": [1.0], "label": ["X"]})
        result = rename_features(df)
        assert "feature_A" in result.columns


# ---------------------------------------------------------------------------
# filter_features_to_match_kaggle
# ---------------------------------------------------------------------------

class TestFilterFeaturesToMatchKaggle:
    @pytest.fixture()
    def kaggle_csv(self, tmp_path):
        """Write a minimal 'Kaggle' CSV with a few known columns."""
        cols = ["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "Activity", "timestamp"]
        df = pd.DataFrame([["0.1", "0.2", "WALKING", "12345"]], columns=cols)
        path = tmp_path / "kaggle.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_keeps_matching_columns(self, kaggle_csv):
        df = pd.DataFrame({
            "tBodyAcc-mean()-X": [0.5],
            "tBodyAcc-mean()-Y": [0.3],
            "Activity": ["WALKING"],
            "timestamp": [12345],
            "extra_col": [99],
        })
        result = filter_features_to_match_kaggle(df, kaggle_csv_path=kaggle_csv)
        assert "tBodyAcc-mean()-X" in result.columns
        assert "tBodyAcc-mean()-Y" in result.columns

    def test_drops_non_kaggle_columns(self, kaggle_csv):
        df = pd.DataFrame({
            "tBodyAcc-mean()-X": [0.5],
            "Activity": ["WALKING"],
            "timestamp": [12345],
            "extra_col": [99],
        })
        result = filter_features_to_match_kaggle(df, kaggle_csv_path=kaggle_csv)
        assert "extra_col" not in result.columns

    def test_always_has_timestamp(self, kaggle_csv):
        df = pd.DataFrame({
            "tBodyAcc-mean()-X": [0.5],
            "Activity": ["WALKING"],
            "timestamp": [12345],
        })
        result = filter_features_to_match_kaggle(df, kaggle_csv_path=kaggle_csv)
        assert "timestamp" in result.columns
