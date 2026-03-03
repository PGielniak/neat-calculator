"""
Unit tests for data_pipeline/database_service.py

Tests cover:
  - initialize_tables
  - save_to_db
  - update_pipeline_run_status
  - get_pipeline_run_status
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from database.database_utils import SQLiteEngine, DatabaseEngine
from data_pipeline.database_service import (
    initialize_tables,
    save_to_db,
    update_pipeline_run_status,
    get_pipeline_run_status,
)
from data_pipeline.models import PipelineRun


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_engine(tmp_path):
    eng = SQLiteEngine(db_path=str(tmp_path / "test.db"))
    yield eng
    eng.close()


def _make_pipeline_run(run_id: str = "run-001") -> PipelineRun:
    return PipelineRun(
        run_id=run_id,
        status="STARTED",
        started_at=datetime(2024, 1, 1, 0, 0, 0),
        folder_path="/data",
        labels_csv_path="/labels.csv",
    )


# ---------------------------------------------------------------------------
# initialize_tables
# ---------------------------------------------------------------------------

class TestInitializeTables:
    def test_creates_pipeline_runs_table(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        assert sqlite_engine.table_exists("pipeline_runs")

    def test_creates_processed_files_table(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        assert sqlite_engine.table_exists("processed_files")

    def test_idempotent_when_called_twice(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        initialize_tables(sqlite_engine)  # no exception
        assert sqlite_engine.table_exists("pipeline_runs")


# ---------------------------------------------------------------------------
# save_to_db
# ---------------------------------------------------------------------------

class TestSaveToDb:
    def test_saves_pydantic_model(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        run = _make_pipeline_run("run-pydantic")
        save_to_db(run, table_name="pipeline_runs", db_engine=sqlite_engine)
        result = sqlite_engine.get_records("pipeline_runs", filters={"run_id": "run-pydantic"})
        assert len(result) == 1

    def test_saves_dataframe(self, sqlite_engine):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        save_to_db(df, table_name="my_df_table", db_engine=sqlite_engine)
        result = sqlite_engine.get_records("my_df_table")
        assert len(result) == 2

    def test_saves_dict(self, sqlite_engine):
        record = {"key": "k1", "value": "v1"}
        save_to_db(record, table_name="kv_store", db_engine=sqlite_engine)
        result = sqlite_engine.get_records("kv_store")
        assert len(result) == 1

    def test_saves_list_of_pydantic_models(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        runs = [_make_pipeline_run(f"run-{i}") for i in range(3)]
        save_to_db(runs, table_name="pipeline_runs", db_engine=sqlite_engine)
        result = sqlite_engine.get_records("pipeline_runs")
        assert len(result) == 3

    def test_saves_list_of_dicts(self, sqlite_engine):
        records = [{"name": "a"}, {"name": "b"}]
        save_to_db(records, table_name="names", db_engine=sqlite_engine)
        result = sqlite_engine.get_records("names")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# update_pipeline_run_status
# ---------------------------------------------------------------------------

class TestUpdatePipelineRunStatus:
    def test_updates_status(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        run = _make_pipeline_run("run-upd")
        save_to_db(run, table_name="pipeline_runs", db_engine=sqlite_engine)

        completed_at = datetime(2024, 1, 1, 1, 0, 0)
        update_pipeline_run_status(sqlite_engine, run_id="run-upd",
                                   status="COMPLETED", completed_at=completed_at)

        result = sqlite_engine.get_records("pipeline_runs", filters={"run_id": "run-upd"})
        assert result.iloc[0]["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# get_pipeline_run_status
# ---------------------------------------------------------------------------

class TestGetPipelineRunStatus:
    def test_returns_correct_status(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        run = _make_pipeline_run("run-get")
        save_to_db(run, table_name="pipeline_runs", db_engine=sqlite_engine)

        status = get_pipeline_run_status(sqlite_engine, run_id="run-get")
        assert status == "STARTED"

    def test_returns_not_found_for_unknown_id(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        status = get_pipeline_run_status(sqlite_engine, run_id="no-such-run")
        assert status == "RUN_ID_NOT_FOUND"

    def test_reflects_updated_status(self, sqlite_engine):
        initialize_tables(sqlite_engine)
        run = _make_pipeline_run("run-reflect")
        save_to_db(run, table_name="pipeline_runs", db_engine=sqlite_engine)
        update_pipeline_run_status(sqlite_engine, run_id="run-reflect",
                                   status="FAILED",
                                   completed_at=datetime(2024, 1, 1, 2, 0, 0))
        assert get_pipeline_run_status(sqlite_engine, run_id="run-reflect") == "FAILED"
