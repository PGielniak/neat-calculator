"""
Unit tests for data_pipeline/data_pipeline.py

External I/O (blob storage, PostgreSQL, process_raw_sensor_data) is mocked so
that the tests run offline.
"""
import json
import logging
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipeline.data_pipeline import (
    _build_file_manifest,
    _persist_sensor_data,
    _process_sensor_data,
    _register_files,
    _resolve_labels_source,
    _resolve_raw_data_source,
)
from data_pipeline.models import ProcessedFile
from database.database_utils import SQLiteEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


@pytest.fixture()
def sqlite_engine(tmp_path):
    engine = SQLiteEngine(db_path=str(tmp_path / "pipeline_test.db"))
    yield engine
    engine.close()


@pytest.fixture()
def raw_data_dir(tmp_path):
    """A directory with two minimal JSON sensor files."""
    sensor_record = {
        "accelerometerX": 0.1, "accelerometerY": 0.2, "accelerometerZ": 9.8,
        "gyroscopeX": 0.01, "gyroscopeY": 0.02, "gyroscopeZ": 0.03,
        "timestamp": 1_700_000_000_000,
        "timestampNanos": 1_700_000_000_000_000_000,
        "label": "WALKING",
    }
    for name in ("session_a.json", "session_b.json"):
        (tmp_path / name).write_text(json.dumps([sensor_record]))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# _resolve_raw_data_source
# ---------------------------------------------------------------------------

class TestResolveRawDataSource:
    def test_returns_local_dir_when_provided(self, raw_data_dir):
        result = _resolve_raw_data_source(raw_data_dir, "", logger)
        assert result == raw_data_dir

    def test_raises_when_neither_source_given(self):
        with pytest.raises(ValueError, match="must be provided"):
            _resolve_raw_data_source("", "", logger)

    def test_downloads_from_blob_when_uri_given(self, tmp_path):
        with patch("data_pipeline.data_pipeline.download_blob_to_dir") as mock_dl:
            result = _resolve_raw_data_source(
                "", "wasbs://container@account.blob.core.windows.net/path", logger
            )
            mock_dl.assert_called_once()
            assert result == "raw_sensor_data_files"


# ---------------------------------------------------------------------------
# _resolve_labels_source
# ---------------------------------------------------------------------------

class TestResolveLabelsSource:
    def test_returns_local_path_when_provided(self, tmp_path):
        csv_path = str(tmp_path / "labels.csv")
        open(csv_path, "w").close()
        result = _resolve_labels_source(csv_path, "", logger)
        assert result == csv_path

    def test_raises_when_neither_source_given(self):
        with pytest.raises(ValueError, match="must be provided"):
            _resolve_labels_source("", "", logger)

    def test_downloads_from_blob_when_uri_given(self, tmp_path):
        with patch("data_pipeline.data_pipeline.download_blob_to_dir") as mock_dl:
            result = _resolve_labels_source(
                "", "wasbs://container@account.blob.core.windows.net/labels.csv", logger
            )
            mock_dl.assert_called_once()
            # Local path is constructed from basename of the URI
            assert result.endswith("labels.csv")


# ---------------------------------------------------------------------------
# _build_file_manifest
# ---------------------------------------------------------------------------

class TestBuildFileManifest:
    def test_returns_dataframe_with_expected_columns(self, raw_data_dir):
        manifest = _build_file_manifest(raw_data_dir, "run-123")
        assert set(manifest.columns) >= {"file_name", "pipeline_run_id", "checksum"}

    def test_pipeline_run_id_populated(self, raw_data_dir):
        manifest = _build_file_manifest(raw_data_dir, "run-abc")
        assert (manifest["pipeline_run_id"] == "run-abc").all()

    def test_checksum_is_md5_hex(self, raw_data_dir):
        manifest = _build_file_manifest(raw_data_dir, "run-xyz")
        for checksum in manifest["checksum"]:
            assert len(checksum) == 32  # MD5 hex = 32 chars

    def test_row_count_matches_files(self, raw_data_dir):
        file_count = len(os.listdir(raw_data_dir))
        manifest = _build_file_manifest(raw_data_dir, "run-1")
        assert len(manifest) == file_count


# ---------------------------------------------------------------------------
# _register_files
# ---------------------------------------------------------------------------

class TestRegisterFiles:
    def _init_tables(self, engine: SQLiteEngine) -> None:
        """Create the processed_files table used by _register_files."""
        from sqlalchemy import text

        ddl = """
        CREATE TABLE IF NOT EXISTS processed_files (
            file_id TEXT PRIMARY KEY,
            file_name TEXT,
            pipeline_run_id TEXT,
            processed_at TEXT,
            checksum TEXT UNIQUE
        )
        """
        with engine.engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()

    def test_new_files_are_registered(self, sqlite_engine, raw_data_dir):
        self._init_tables(sqlite_engine)
        manifest = _build_file_manifest(raw_data_dir, "run-new")
        # save_to_db has a broken 'infra' import; patch at the call site
        with patch("data_pipeline.data_pipeline.save_to_db") as mock_save:
            mock_save.return_value = None
            processed, skipped = _register_files(manifest, sqlite_engine, logger)
        assert len(processed) == len(manifest)
        assert len(skipped) == 0

    def test_duplicate_files_are_skipped(self, sqlite_engine, raw_data_dir):
        self._init_tables(sqlite_engine)
        manifest = _build_file_manifest(raw_data_dir, "run-dup")
        # Register once
        with patch("data_pipeline.data_pipeline.save_to_db"):
            _register_files(manifest, sqlite_engine, logger)
        # Register again — duplicates raise IntegrityError in save_to_db.
        # We simulate this by letting save_to_db raise IntegrityError.
        from sqlalchemy.exc import IntegrityError as SAIntegrityError

        def _raise_dup(data, table_name, db_engine):
            raise SAIntegrityError(
                statement=None,
                params=None,
                orig=Exception(
                    'duplicate key value violates unique constraint "processed_files_checksum_key"'
                ),
            )

        with patch("data_pipeline.data_pipeline.save_to_db", side_effect=_raise_dup):
            processed, skipped = _register_files(manifest, sqlite_engine, logger)
        assert len(processed) == 0
        assert len(skipped) == len(manifest)


# ---------------------------------------------------------------------------
# _persist_sensor_data
# ---------------------------------------------------------------------------

class TestPersistSensorData:
    def test_saves_data_and_returns_true(self, sqlite_engine):
        sensor_data = pd.DataFrame({
            "tBodyAcc-mean()-X": [0.1, 0.2],
            "Activity": ["WALKING", "SITTING"],
            "timestamp": [1000, 1020],
        })

        with patch("data_pipeline.data_pipeline.save_to_db") as mock_save:
            mock_save.return_value = None
            result = _persist_sensor_data(sensor_data, sqlite_engine, logger)
        assert result is True

    def test_returns_false_on_exception(self, sqlite_engine):
        sensor_data = pd.DataFrame({"col": [1]})
        with patch("data_pipeline.data_pipeline.save_to_db", side_effect=RuntimeError("DB down")):
            result = _persist_sensor_data(sensor_data, sqlite_engine, logger)
        assert result is False


# ---------------------------------------------------------------------------
# _process_sensor_data
# ---------------------------------------------------------------------------

class TestProcessSensorData:
    def test_delegates_to_process_raw_sensor_data(self, raw_data_dir, tmp_path):
        fake_df = pd.DataFrame({"feature": [1.0], "Activity": ["WALKING"]})

        with patch("data_pipeline.data_pipeline.process_raw_sensor_data", return_value=fake_df) as mock_proc:
            result = _process_sensor_data(
                raw_data_dir=raw_data_dir,
                labels_csv_path=str(tmp_path / "labels.csv"),
                kaggle_csv_path=str(tmp_path / "kaggle.csv"),
                skipped_files=[],
                use_v2_features=True,
                logger=logger,
            )
            mock_proc.assert_called_once()
            assert isinstance(result, pd.DataFrame)
