"""
Unit tests for database/database_utils.py

Tests cover:
  - SQLiteEngine  (save_dataframe, save_record, table_exists, get_records,
                   update_record, close)
  - DatabaseFactory.create_engine
  - DatabaseRepository  (delegates correctly to the engine)
"""
import tempfile
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from database.database_utils import (
    SQLiteEngine,
    DatabaseFactory,
    DatabaseRepository,
    DatabaseEngine,
)


# ---------------------------------------------------------------------------
# SQLiteEngine
# ---------------------------------------------------------------------------

class TestSQLiteEngine:
    """Use an in-memory (temp-file) SQLite database for all tests."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test.db")

    @pytest.fixture
    def engine(self, db_path):
        eng = SQLiteEngine(db_path=db_path)
        yield eng
        eng.close()

    # --- save_dataframe / table_exists ---

    def test_save_dataframe_creates_table(self, engine):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        engine.save_dataframe(df, "test_table")
        assert engine.table_exists("test_table")

    def test_save_dataframe_appends_rows(self, engine):
        df1 = pd.DataFrame({"val": [1, 2]})
        df2 = pd.DataFrame({"val": [3, 4]})
        engine.save_dataframe(df1, "t")
        engine.save_dataframe(df2, "t", if_exists="append")
        result = engine.get_records("t")
        assert len(result) == 4

    def test_table_not_exists_returns_false(self, engine):
        assert not engine.table_exists("nonexistent_table")

    # --- save_record ---

    def test_save_record_inserts_row(self, engine):
        engine.save_record({"name": "alice", "score": 42}, "people")
        result = engine.get_records("people")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "alice"

    # --- get_records ---

    def test_get_records_returns_all_rows(self, engine):
        df = pd.DataFrame({"x": [1, 2, 3]})
        engine.save_dataframe(df, "nums")
        result = engine.get_records("nums")
        assert len(result) == 3

    def test_get_records_with_filter(self, engine):
        df = pd.DataFrame({"city": ["London", "Paris", "London"], "pop": [9, 2, 9]})
        engine.save_dataframe(df, "cities")
        result = engine.get_records("cities", filters={"city": "London"})
        assert len(result) == 2
        assert (result["city"] == "London").all()

    def test_get_records_with_columns(self, engine):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        engine.save_dataframe(df, "abc")
        result = engine.get_records("abc", columns=["a", "b"])
        assert "c" not in result.columns

    # --- update_record ---

    def test_update_record_modifies_value(self, engine):
        engine.save_dataframe(pd.DataFrame({"id": ["r1"], "status": ["STARTED"]}), "runs")
        engine.update_record("runs", primary_key="id", key_value="r1",
                             updates={"status": "COMPLETED"})
        result = engine.get_records("runs", filters={"id": "r1"})
        assert result.iloc[0]["status"] == "COMPLETED"

    def test_update_nonexistent_record_raises(self, engine):
        engine.save_dataframe(pd.DataFrame({"id": ["r1"], "status": ["STARTED"]}), "runs")
        with pytest.raises(ValueError):
            engine.update_record("runs", primary_key="id", key_value="no_such_id",
                                 updates={"status": "DONE"})

    # --- close ---

    def test_close_does_not_raise(self, engine):
        engine.close()  # should not raise


# ---------------------------------------------------------------------------
# DatabaseFactory
# ---------------------------------------------------------------------------

class TestDatabaseFactory:
    def test_creates_sqlite_engine(self, tmp_path):
        eng = DatabaseFactory.create_engine("sqlite", db_path=str(tmp_path / "f.db"))
        assert isinstance(eng, SQLiteEngine)
        eng.close()

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DatabaseFactory.create_engine("mysql")

    def test_default_db_path_used_for_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        eng = DatabaseFactory.create_engine("sqlite")
        assert isinstance(eng, SQLiteEngine)
        eng.close()


# ---------------------------------------------------------------------------
# DatabaseRepository
# ---------------------------------------------------------------------------

class TestDatabaseRepository:
    """DatabaseRepository delegates to the engine — verify delegation."""

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock(spec=DatabaseEngine)
        engine.get_records.return_value = pd.DataFrame({"a": [1]})
        return engine

    @pytest.fixture
    def repo(self, mock_engine):
        return DatabaseRepository(engine=mock_engine)

    def test_save_dataframe_delegates(self, repo, mock_engine):
        df = pd.DataFrame({"x": [1]})
        repo.save_dataframe(df, "t")
        mock_engine.save_dataframe.assert_called_once_with(df, "t", "append")

    def test_save_record_delegates(self, repo, mock_engine):
        repo.save_record({"k": "v"}, "t")
        mock_engine.save_record.assert_called_once_with({"k": "v"}, "t")

    def test_save_records_delegates_as_dataframe(self, repo, mock_engine):
        records = [{"a": 1}, {"a": 2}]
        repo.save_records(records, "t")
        mock_engine.save_dataframe.assert_called_once()

    def test_get_records_delegates(self, repo, mock_engine):
        result = repo.get_records("t", filters={"id": "1"})
        mock_engine.get_records.assert_called_once_with("t", {"id": "1"})
        assert isinstance(result, pd.DataFrame)

    def test_update_record_delegates(self, repo, mock_engine):
        repo.update_record("t", "id", "val", {"status": "done"})
        mock_engine.update_record.assert_called_once_with("t", "id", "val", {"status": "done"})

    def test_table_exists_delegates(self, repo, mock_engine):
        mock_engine.table_exists.return_value = True
        assert repo.table_exists("some_table") is True

    def test_close_delegates(self, repo, mock_engine):
        repo.close()
        mock_engine.close.assert_called_once()
