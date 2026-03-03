"""
Unit tests for data_pipeline/data_pipeline.py

Tests cover:
  - _resolve_raw_data_source
  - _resolve_labels_source
  - _build_file_manifest
  - _register_files
  - _process_sensor_data
  - _persist_sensor_data
"""
import hashlib
import logging
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from data_pipeline.data_pipeline import (
    _resolve_raw_data_source,
    _resolve_labels_source,
    _build_file_manifest,
    _register_files,
    _process_sensor_data,
    _persist_sensor_data,
)
from database.database_utils import SQLiteEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_engine(tmp_path):
    from data_pipeline.database_service import initialize_tables
    eng = SQLiteEngine(db_path=str(tmp_path / "test.db"))
    initialize_tables(eng)
    yield eng
    eng.close()


# ---------------------------------------------------------------------------
# _resolve_raw_data_source
# ---------------------------------------------------------------------------

class TestResolveRawDataSource:
    def test_raises_when_neither_source_provided(self):
        with pytest.raises(ValueError, match="Either"):
            _resolve_raw_data_source("", "", logger)

    def test_returns_provided_directory(self, tmp_path):
        result = _resolve_raw_data_source(str(tmp_path), "", logger)
        assert result == str(tmp_path)

    def test_downloads_when_container_uri_given(self, tmp_path):
        with patch("data_pipeline.data_pipeline.download_blob_to_dir") as mock_dl:
            result = _resolve_raw_data_source(
                "", "abfss://cont@acct.blob.core.windows.net/data", logger
            )
            mock_dl.assert_called_once()

    def test_uses_default_dir_name_when_only_uri_given(self, tmp_path):
        with patch("data_pipeline.data_pipeline.download_blob_to_dir"):
            result = _resolve_raw_data_source(
                "", "abfss://cont@acct.blob.core.windows.net/data", logger
            )
            assert result == "raw_sensor_data_files"


# ---------------------------------------------------------------------------
# _resolve_labels_source
# ---------------------------------------------------------------------------

class TestResolveLabelsSource:
    def test_raises_when_neither_source_provided(self):
        with pytest.raises(ValueError, match="Either"):
            _resolve_labels_source("", "", logger)

    def test_returns_provided_csv_path(self, tmp_path):
        csv_path = str(tmp_path / "labels.csv")
        result = _resolve_labels_source(csv_path, "", logger)
        assert result == csv_path

    def test_downloads_when_blob_uri_given(self, tmp_path):
        with patch("data_pipeline.data_pipeline.download_blob_to_dir") as mock_dl:
            result = _resolve_labels_source(
                "", "abfss://cont@acct.blob.core.windows.net/labels.csv", logger
            )
            mock_dl.assert_called_once()


# ---------------------------------------------------------------------------
# _build_file_manifest
# ---------------------------------------------------------------------------

class TestBuildFileManifest:
    def test_returns_dataframe(self, tmp_path):
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")
        manifest = _build_file_manifest(str(tmp_path), "run-1")
        assert isinstance(manifest, pd.DataFrame)

    def test_contains_expected_columns(self, tmp_path):
        (tmp_path / "c.json").write_text("{}")
        manifest = _build_file_manifest(str(tmp_path), "run-1")
        assert set(["file_name", "pipeline_run_id", "checksum"]).issubset(manifest.columns)

    def test_checksum_is_md5_of_file_name(self, tmp_path):
        (tmp_path / "file.json").write_text("{}")
        manifest = _build_file_manifest(str(tmp_path), "run-1")
        expected = hashlib.md5("file.json".encode()).hexdigest()
        assert manifest.iloc[0]["checksum"] == expected

    def test_pipeline_run_id_populated(self, tmp_path):
        (tmp_path / "x.json").write_text("{}")
        manifest = _build_file_manifest(str(tmp_path), "run-abc")
        assert (manifest["pipeline_run_id"] == "run-abc").all()

    def test_files_sorted(self, tmp_path):
        for name in ["z.json", "a.json", "m.json"]:
            (tmp_path / name).write_text("{}")
        manifest = _build_file_manifest(str(tmp_path), "run-1")
        assert list(manifest["file_name"]) == sorted(manifest["file_name"])


# ---------------------------------------------------------------------------
# _register_files
# ---------------------------------------------------------------------------

class TestRegisterFiles:
    def _make_manifest(self, names: list[str], run_id: str = "run-1") -> pd.DataFrame:
        rows = [
            {
                "file_name": n,
                "pipeline_run_id": run_id,
                "checksum": hashlib.md5(n.encode()).hexdigest(),
            }
            for n in names
        ]
        return pd.DataFrame(rows)

    def test_registers_new_files(self, sqlite_engine):
        from data_pipeline.database_service import save_to_db
        from data_pipeline.models import PipelineRun
        run = PipelineRun(run_id="run-1", status="STARTED",
                          started_at=datetime.now(), folder_path="/", labels_csv_path="/")
        save_to_db(run, table_name="pipeline_runs", db_engine=sqlite_engine)

        manifest = self._make_manifest(["a.json", "b.json"])
        processed, skipped = _register_files(manifest, sqlite_engine, logger)
        assert len(processed) == 2
        assert skipped == []

    def test_skips_duplicate_checksums(self, sqlite_engine):
        from data_pipeline.database_service import save_to_db
        from data_pipeline.models import PipelineRun
        run = PipelineRun(run_id="run-dup", status="STARTED",
                          started_at=datetime.now(), folder_path="/", labels_csv_path="/")
        save_to_db(run, table_name="pipeline_runs", db_engine=sqlite_engine)

        manifest = self._make_manifest(["dup.json"], run_id="run-dup")
        _register_files(manifest, sqlite_engine, logger)  # first insertion

        # Insert again: same checksum → should be skipped
        run2 = PipelineRun(run_id="run-dup2", status="STARTED",
                           started_at=datetime.now(), folder_path="/", labels_csv_path="/")
        save_to_db(run2, table_name="pipeline_runs", db_engine=sqlite_engine)
        manifest2 = self._make_manifest(["dup.json"], run_id="run-dup2")

        with patch("data_pipeline.data_pipeline.save_to_db") as mock_save:
            dup_error = IntegrityError(
                statement=None,
                params=None,
                orig=Exception(
                    'duplicate key value violates unique constraint "processed_files_checksum_key"'
                ),
            )
            mock_save.side_effect = dup_error
            processed, skipped = _register_files(manifest2, sqlite_engine, logger)

        assert "dup.json" in skipped


# ---------------------------------------------------------------------------
# _process_sensor_data
# ---------------------------------------------------------------------------

class TestProcessSensorData:
    def test_calls_process_raw_sensor_data(self, tmp_path):
        mock_df = pd.DataFrame({"Activity": ["WALKING"], "timestamp": [1700000000000]})
        with patch("data_pipeline.data_pipeline.process_raw_sensor_data",
                   return_value=mock_df) as mock_proc:
            result = _process_sensor_data(
                raw_data_dir=str(tmp_path),
                labels_csv_path="/labels.csv",
                kaggle_csv_path="/kaggle.csv",
                skipped_files=[],
                use_v2_features=False,
                logger=logger,
            )
        mock_proc.assert_called_once()
        assert len(result) == 1

    def test_uses_v2_when_flag_set(self, tmp_path):
        mock_df = pd.DataFrame({"Activity": ["SITTING"], "timestamp": [1700000000000]})
        with patch("data_pipeline.data_pipeline.process_raw_sensor_data",
                   return_value=mock_df) as mock_proc:
            _process_sensor_data(
                raw_data_dir=str(tmp_path),
                labels_csv_path="/labels.csv",
                kaggle_csv_path="/kaggle.csv",
                skipped_files=[],
                use_v2_features=True,
                logger=logger,
            )
        _, kwargs = mock_proc.call_args
        assert kwargs.get("version") == "2"


# ---------------------------------------------------------------------------
# _persist_sensor_data
# ---------------------------------------------------------------------------

class TestPersistSensorData:
    def test_returns_true_on_success(self):
        mock_engine = MagicMock()
        with patch("data_pipeline.data_pipeline.save_to_db"):
            result = _persist_sensor_data(
                pd.DataFrame({"a": [1]}), mock_engine, logger
            )
        assert result is True

    def test_returns_false_on_exception(self):
        mock_engine = MagicMock()
        with patch("data_pipeline.data_pipeline.save_to_db",
                   side_effect=Exception("DB error")):
            result = _persist_sensor_data(
                pd.DataFrame({"a": [1]}), mock_engine, logger
            )
        assert result is False
