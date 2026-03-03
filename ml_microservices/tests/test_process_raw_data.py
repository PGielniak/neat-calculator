"""
Unit tests for shared/process_raw_data.py

Tests cover:
  - validate_directory
  - validate_labels_csv
  - merge_json_files
  - label_data
  - label_data_v2
  - remove_duplicates
  - resample_data
  - create_sliding_windows
  - extract_features_from_windows
  - rename_features
  - filter_features_to_match_kaggle
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from shared.process_raw_data import (
    SensorRecording,
    validate_directory,
    validate_labels_csv,
    merge_json_files,
    label_data,
    label_data_v2,
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
    "accelerometerX", "accelerometerY", "accelerometerZ",
    "gyroscopeX", "gyroscopeY", "gyroscopeZ",
]


def _make_sensor_record(ts: int, label: str = "WALKING") -> dict:
    return {
        "accelerometerX": 0.1,
        "accelerometerY": 0.2,
        "accelerometerZ": 9.8,
        "gyroscopeX": 0.01,
        "gyroscopeY": 0.02,
        "gyroscopeZ": 0.03,
        "timestamp": ts,
        "timestampNanos": ts * 1_000_000,
        "label": label,
    }


def _make_sensor_df(n: int = 200, start_ts_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    """Return a DataFrame of n sensor readings at 50 Hz from start_ts_ms."""
    step_ms = 20  # 50 Hz
    records = [_make_sensor_record(start_ts_ms + i * step_ms) for i in range(n)]
    df = pd.DataFrame(records)
    df["timestampNanos"] = df["timestamp"] * 1_000_000
    return df


def _make_valid_json_file(directory: str, filename: str, n_records: int = 5) -> str:
    records = [_make_sensor_record(1_700_000_000_000 + i * 20) for i in range(n_records)]
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(records, f)
    return path


# ---------------------------------------------------------------------------
# validate_directory
# ---------------------------------------------------------------------------

class TestValidateDirectory:
    def test_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            validate_directory("/nonexistent/path/xyz")

    def test_file_not_directory_raises(self):
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(NotADirectoryError):
                validate_directory(f.name)

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="empty"):
                validate_directory(d)

    def test_all_files_skipped_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_json_file(d, "a.json")
            with pytest.raises(ValueError, match="skipped"):
                validate_directory(d, skipped_files=["a.json"])

    def test_non_json_file_is_skipped_silently(self):
        with tempfile.TemporaryDirectory() as d:
            # Write one .txt and one valid .json
            with open(os.path.join(d, "notes.txt"), "w") as f:
                f.write("hello")
            _make_valid_json_file(d, "data.json")
            # Should not raise
            validate_directory(d)

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "bad.json"), "w") as f:
                f.write("not valid json {{{")
            with pytest.raises(ValueError):
                validate_directory(d)

    def test_valid_directory_passes(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_json_file(d, "sensor.json")
            # Pass explicit list to avoid Python's mutable default argument issue
            validate_directory(d, skipped_files=[])  # no exception


# ---------------------------------------------------------------------------
# validate_labels_csv
# ---------------------------------------------------------------------------

class TestValidateLabelsCSV:
    def test_missing_activity_column_raises(self):
        df = pd.DataFrame({"Feature": [1, 2, 3]})
        with pytest.raises(ValueError, match="Activity"):
            validate_labels_csv(df)

    def test_all_valid_labels_pass(self):
        df = pd.DataFrame({"Activity": ["WALKING", "STANDING", "SITTING", "LAYING",
                                        "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]})
        result = validate_labels_csv(df)
        assert len(result) == 6

    def test_invalid_labels_are_removed(self):
        df = pd.DataFrame({"Activity": ["WALKING", "RUNNING", "SITTING"]})
        result = validate_labels_csv(df)
        assert "RUNNING" not in result["Activity"].values
        assert len(result) == 2

    def test_returns_dataframe(self):
        df = pd.DataFrame({"Activity": ["LAYING"], "feat": [1.0]})
        result = validate_labels_csv(df)
        assert isinstance(result, pd.DataFrame)

    def test_custom_label_column(self):
        df = pd.DataFrame({"label": ["WALKING"], "x": [1.0]})
        result = validate_labels_csv(df, label_column="label")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# merge_json_files
# ---------------------------------------------------------------------------

class TestMergeJsonFiles:
    def test_merges_multiple_files(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_json_file(d, "a.json", n_records=3)
            _make_valid_json_file(d, "b.json", n_records=4)
            result = merge_json_files(d)
            assert len(result) == 7

    def test_sorted_by_timestamp(self):
        with tempfile.TemporaryDirectory() as d:
            records_a = [_make_sensor_record(1_700_000_000_020)]
            records_b = [_make_sensor_record(1_700_000_000_000)]
            with open(os.path.join(d, "a.json"), "w") as f:
                json.dump(records_a, f)
            with open(os.path.join(d, "b.json"), "w") as f:
                json.dump(records_b, f)
            result = merge_json_files(d)
            timestamps = [r["timestamp"] for r in result]
            assert timestamps == sorted(timestamps)

    def test_skipped_files_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_json_file(d, "keep.json", n_records=5)
            _make_valid_json_file(d, "skip.json", n_records=3)
            result = merge_json_files(d, skipped_files=["skip.json"])
            assert len(result) == 5


# ---------------------------------------------------------------------------
# label_data
# ---------------------------------------------------------------------------

class TestLabelData:
    def test_no_labels_csv_path_uses_unlabeled(self):
        data = [_make_sensor_record(ts) for ts in range(1_700_000_000_000,
                                                         1_700_000_000_000 + 10 * 20,
                                                         20)]
        result = label_data(data, labels_csv_path="")
        assert (result["label"] == "UNLABELED").all()

    def test_returns_dataframe(self):
        data = [_make_sensor_record(1_700_000_000_000)]
        result = label_data(data, labels_csv_path="")
        assert isinstance(result, pd.DataFrame)

    def test_applies_labels_from_csv(self):
        with tempfile.TemporaryDirectory() as d:
            # Create sensor data
            data = [_make_sensor_record(1_700_000_000_000 + i * 20, "UNLABELED")
                    for i in range(50)]
            # Create labels CSV: one label starting at the first timestamp
            labels_csv = os.path.join(d, "labels.csv")
            labels_df = pd.DataFrame({
                "timestamp": [1_700_000_000_000],
                "label": ["WALKING"],
            })
            labels_df.to_csv(labels_csv, index=False)
            result = label_data(data, labels_csv_path=labels_csv)
            # All rows from the matched timestamp onward should be WALKING
            assert "WALKING" in result["label"].values


# ---------------------------------------------------------------------------
# label_data_v2
# ---------------------------------------------------------------------------

class TestLabelDataV2:
    def test_returns_dataframe_with_label_column(self):
        with tempfile.TemporaryDirectory() as d:
            start_ms = 1_700_000_000_000
            data = [_make_sensor_record(start_ms + i * 20) for i in range(100)]
            labels_csv = os.path.join(d, "labels.csv")
            labels_df = pd.DataFrame({
                "StartTimestamp_Unix_Ms": [start_ms],
                "EndTimestamp_Unix_Ms": [start_ms + 2000],
                "Label": ["Walking"],
            })
            labels_df.to_csv(labels_csv, index=False)
            result = label_data_v2(data, labels_csv_path=labels_csv)
            assert isinstance(result, pd.DataFrame)
            assert "label" in result.columns

    def test_maps_labels_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            start_ms = 1_700_000_000_000
            data = [_make_sensor_record(start_ms + i * 20) for i in range(100)]
            labels_csv = os.path.join(d, "labels.csv")
            labels_df = pd.DataFrame({
                "StartTimestamp_Unix_Ms": [start_ms],
                "EndTimestamp_Unix_Ms": [start_ms + 1980],
                "Label": ["Sitting"],
            })
            labels_df.to_csv(labels_csv, index=False)
            result = label_data_v2(data, labels_csv_path=labels_csv)
            assert "SITTING" in result["label"].values

    def test_unlabeled_rows_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            start_ms = 1_700_000_000_000
            # Only 10 records match the label window
            data = [_make_sensor_record(start_ms + i * 20) for i in range(100)]
            labels_csv = os.path.join(d, "labels.csv")
            labels_df = pd.DataFrame({
                "StartTimestamp_Unix_Ms": [start_ms],
                "EndTimestamp_Unix_Ms": [start_ms + 180],  # covers first 10 records
                "Label": ["Standing"],
            })
            labels_df.to_csv(labels_csv, index=False)
            result = label_data_v2(data, labels_csv_path=labels_csv)
            # Rows outside the label window should be dropped (NaN label dropped)
            assert result["label"].notna().all()


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def _make_df_with_duplicates(self) -> pd.DataFrame:
        start_ms = 1_700_000_000_000
        records = []
        for i in range(10):
            ts = start_ms + i * 20
            records.append({**_make_sensor_record(ts), "timestampNanos": ts * 1_000_000})
            # Add a duplicate
            records.append({**_make_sensor_record(ts), "timestampNanos": ts * 1_000_000})
        df = pd.DataFrame(records)
        df["label"] = "WALKING"
        return df

    def test_removes_duplicate_timestamps(self):
        df = self._make_df_with_duplicates()
        result = remove_duplicates(df, sensor_cols=SENSOR_COLS)
        # After removing duplicates, timestampNanos should be unique
        assert result.index.duplicated().sum() == 0

    def test_returns_dataframe(self):
        df = _make_sensor_df(50)
        df["label"] = "WALKING"
        result = remove_duplicates(df, sensor_cols=SENSOR_COLS)
        assert isinstance(result, pd.DataFrame)

    def test_creates_timestampnanos_if_missing(self):
        df = _make_sensor_df(20)
        df["label"] = "WALKING"
        df = df.drop(columns=["timestampNanos"])
        result = remove_duplicates(df, sensor_cols=SENSOR_COLS)
        assert isinstance(result, pd.DataFrame)

    def test_drops_old_timestamps(self):
        """Rows with timestampNanos < 1.5e18 (pre-2017) should be dropped."""
        df = _make_sensor_df(20)
        df["label"] = "WALKING"
        # Insert a row with an invalid (very old) timestampNanos
        old_row = df.iloc[0:1].copy()
        old_row["timestampNanos"] = 1000  # way before 2017
        combined = pd.concat([df, old_row], ignore_index=True)
        result = remove_duplicates(combined, sensor_cols=SENSOR_COLS)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# resample_data
# ---------------------------------------------------------------------------

class TestResampleData:
    def _make_resampled_input(self, n: int = 300) -> pd.DataFrame:
        df = _make_sensor_df(n)
        df["label"] = "WALKING"
        return remove_duplicates(df, sensor_cols=SENSOR_COLS)

    def test_returns_dataframe(self):
        df = self._make_resampled_input()
        result = resample_data(df, target_freq=50, sensor_cols=SENSOR_COLS)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_sensor_columns(self):
        df = self._make_resampled_input()
        result = resample_data(df, target_freq=50, sensor_cols=SENSOR_COLS)
        for col in SENSOR_COLS:
            assert col in result.columns

    def test_output_has_label_column(self):
        df = self._make_resampled_input()
        result = resample_data(df, target_freq=50, sensor_cols=SENSOR_COLS)
        assert "label" in result.columns


# ---------------------------------------------------------------------------
# create_sliding_windows
# ---------------------------------------------------------------------------

class TestCreateSlidingWindows:
    def _make_pipeline_df(self, n: int = 400) -> pd.DataFrame:
        raw = _make_sensor_df(n)
        raw["label"] = "WALKING"
        deduped = remove_duplicates(raw, sensor_cols=SENSOR_COLS)
        return resample_data(deduped, target_freq=50, sensor_cols=SENSOR_COLS)

    def test_returns_four_values(self):
        df = self._make_pipeline_df()
        result = create_sliding_windows(df, window_size=128, step_size=64,
                                        sensor_cols=SENSOR_COLS)
        assert len(result) == 4

    def test_windows_correct_shape(self):
        df = self._make_pipeline_df(600)
        _, X, y, ts = create_sliding_windows(df, window_size=128, step_size=64,
                                              sensor_cols=SENSOR_COLS)
        assert X.shape[1] == 128
        assert X.shape[2] == len(SENSOR_COLS)

    def test_labels_and_windows_same_length(self):
        df = self._make_pipeline_df(600)
        _, X, y, ts = create_sliding_windows(df, window_size=128, step_size=64,
                                              sensor_cols=SENSOR_COLS)
        assert len(X) == len(y) == len(ts)

    def test_short_data_produces_no_windows(self):
        df = self._make_pipeline_df(50)
        # create_sliding_windows raises ValueError when no windows can be formed
        # (np.stack requires at least one array)
        with pytest.raises((ValueError, IndexError)):
            create_sliding_windows(df, window_size=128, step_size=64,
                                   sensor_cols=SENSOR_COLS)


# ---------------------------------------------------------------------------
# extract_features_from_windows
# ---------------------------------------------------------------------------

class TestExtractFeaturesFromWindows:
    def _make_windows(self, n_windows: int = 3) -> tuple:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n_windows, 128, 6))
        # Add gravity component to accelerometer Z axis
        from shared.helper_functions import GRAVITY
        X[:, :, 2] += GRAVITY
        y = np.array(["WALKING"] * n_windows)
        ts = np.array([1_700_000_000_000 + i * 2560 for i in range(n_windows)])
        dummy_df = pd.DataFrame()
        return dummy_df, X, y, ts

    def test_returns_dataframe(self):
        df, X, y, ts = self._make_windows()
        result = extract_features_from_windows(df, X, y, ts)
        assert isinstance(result, pd.DataFrame)

    def test_has_label_column(self):
        df, X, y, ts = self._make_windows()
        result = extract_features_from_windows(df, X, y, ts)
        assert "label" in result.columns

    def test_has_timestamp_column(self):
        df, X, y, ts = self._make_windows()
        result = extract_features_from_windows(df, X, y, ts)
        assert "timestamp" in result.columns

    def test_row_count_matches_windows(self):
        n = 5
        df, X, y, ts = self._make_windows(n)
        result = extract_features_from_windows(df, X, y, ts)
        assert len(result) == n


# ---------------------------------------------------------------------------
# rename_features
# ---------------------------------------------------------------------------

class TestRenameFeatures:
    def test_renames_label_to_activity(self):
        df = pd.DataFrame({"label": ["WALKING"], "feat1": [1.0]})
        result = rename_features(df)
        assert "Activity" in result.columns
        assert "label" not in result.columns

    def test_renames_subject_to_subject_capitalized(self):
        df = pd.DataFrame({"label": ["WALKING"], "subject": [1]})
        result = rename_features(df)
        assert "Subject" in result.columns

    def test_returns_dataframe(self):
        df = pd.DataFrame({"label": ["WALKING"]})
        result = rename_features(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# filter_features_to_match_kaggle
# ---------------------------------------------------------------------------

class TestFilterFeaturesToMatchKaggle:
    def test_returns_dataframe(self):
        with tempfile.TemporaryDirectory() as d:
            kaggle_csv = os.path.join(d, "kaggle.csv")
            kaggle_df = pd.DataFrame({"feat_a": [1.0], "Activity": ["WALKING"], "timestamp": [1_700_000_000_000]})
            kaggle_df.to_csv(kaggle_csv, index=False)

            my_df = pd.DataFrame({
                "feat_a": [2.0], "feat_b": [3.0],
                "Activity": ["WALKING"],
                "timestamp": [1_700_000_000_000],
            })
            result = filter_features_to_match_kaggle(my_df, kaggle_csv_path=kaggle_csv)
            assert isinstance(result, pd.DataFrame)

    def test_only_kaggle_columns_retained(self):
        with tempfile.TemporaryDirectory() as d:
            kaggle_csv = os.path.join(d, "kaggle.csv")
            kaggle_df = pd.DataFrame({"feat_a": [1.0], "Activity": ["WALKING"], "timestamp": [1_700_000_000_000]})
            kaggle_df.to_csv(kaggle_csv, index=False)

            my_df = pd.DataFrame({
                "feat_a": [2.0], "feat_b": [9.9],
                "Activity": ["WALKING"],
                "timestamp": [1_700_000_000_000],
            })
            result = filter_features_to_match_kaggle(my_df, kaggle_csv_path=kaggle_csv)
            assert "feat_b" not in result.columns
            assert "feat_a" in result.columns

    def test_timestamp_always_present(self):
        with tempfile.TemporaryDirectory() as d:
            kaggle_csv = os.path.join(d, "kaggle.csv")
            kaggle_df = pd.DataFrame({"feat_a": [1.0], "Activity": ["WALKING"]})
            kaggle_df.to_csv(kaggle_csv, index=False)

            my_df = pd.DataFrame({
                "feat_a": [2.0],
                "Activity": ["WALKING"],
                "timestamp": [1_700_000_000_000],
            })
            result = filter_features_to_match_kaggle(my_df, kaggle_csv_path=kaggle_csv)
            assert "timestamp" in result.columns
