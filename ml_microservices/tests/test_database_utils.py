"""
Unit tests for database/database_utils.py

Uses in-memory SQLite so no external services are required.
"""
import os
import tempfile

import pandas as pd
import pytest

from database.database_utils import (
    DatabaseFactory,
    DatabaseRepository,
    SQLiteEngine,
    get_postgres_db_engine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sqlite_engine(tmp_path):
    """A fresh SQLite engine backed by a temp file."""
    engine = SQLiteEngine(db_path=str(tmp_path / "test.db"))
    yield engine
    engine.close()


@pytest.fixture()
def repo(sqlite_engine):
    return DatabaseRepository(sqlite_engine)


# ---------------------------------------------------------------------------
# SQLiteEngine
# ---------------------------------------------------------------------------

class TestSQLiteEngine:
    def test_save_and_retrieve_dataframe(self, sqlite_engine):
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        sqlite_engine.save_dataframe(df, "test_table", if_exists="replace")
        result = sqlite_engine.get_records("test_table")
        assert len(result) == 3
        assert list(result["col_a"]) == [1, 2, 3]

    def test_table_exists_returns_true_after_creation(self, sqlite_engine):
        df = pd.DataFrame({"val": [10]})
        sqlite_engine.save_dataframe(df, "my_table", if_exists="replace")
        assert sqlite_engine.table_exists("my_table") is True

    def test_table_exists_returns_false_for_missing_table(self, sqlite_engine):
        assert sqlite_engine.table_exists("nonexistent") is False

    def test_save_record_appends_row(self, sqlite_engine):
        df = pd.DataFrame({"id": [1], "name": ["alice"]})
        sqlite_engine.save_dataframe(df, "people", if_exists="replace")
        sqlite_engine.save_record({"id": 2, "name": "bob"}, "people")
        result = sqlite_engine.get_records("people")
        assert len(result) == 2

    def test_get_records_with_filter(self, sqlite_engine):
        df = pd.DataFrame({"id": [1, 2, 3], "status": ["ok", "err", "ok"]})
        sqlite_engine.save_dataframe(df, "items", if_exists="replace")
        result = sqlite_engine.get_records("items", filters={"status": "ok"})
        assert len(result) == 2
        assert all(result["status"] == "ok")

    def test_get_records_with_columns(self, sqlite_engine):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"], "col_c": [0.1, 0.2]})
        sqlite_engine.save_dataframe(df, "cols_test", if_exists="replace")
        result = sqlite_engine.get_records("cols_test", columns=["col_a"])
        assert "col_a" in result.columns
        assert "col_b" not in result.columns

    def test_update_record(self, sqlite_engine):
        df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        sqlite_engine.save_dataframe(df, "update_test", if_exists="replace")
        sqlite_engine.update_record("update_test", primary_key="id", key_value=1, updates={"value": 99})
        result = sqlite_engine.get_records("update_test", filters={"id": 1})
        assert int(result["value"].iloc[0]) == 99

    def test_update_record_missing_key_raises(self, sqlite_engine):
        df = pd.DataFrame({"id": [1], "value": [10]})
        sqlite_engine.save_dataframe(df, "err_test", if_exists="replace")
        with pytest.raises(ValueError):
            sqlite_engine.update_record("err_test", primary_key="id", key_value=999, updates={"value": 0})

    def test_close_does_not_raise(self, sqlite_engine):
        sqlite_engine.close()  # should not raise

    def test_save_dataframe_replace_overwrites_table(self, sqlite_engine):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [9]})
        sqlite_engine.save_dataframe(df1, "replace_tbl", if_exists="replace")
        sqlite_engine.save_dataframe(df2, "replace_tbl", if_exists="replace")
        result = sqlite_engine.get_records("replace_tbl")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# DatabaseFactory
# ---------------------------------------------------------------------------

class TestDatabaseFactory:
    def test_create_sqlite_engine(self, tmp_path):
        engine = DatabaseFactory.create_engine("sqlite", db_path=str(tmp_path / "factory.db"))
        assert isinstance(engine, SQLiteEngine)
        engine.close()

    def test_unsupported_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DatabaseFactory.create_engine("mysql")


# ---------------------------------------------------------------------------
# DatabaseRepository
# ---------------------------------------------------------------------------

class TestDatabaseRepository:
    def test_save_and_get_records(self, repo):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        repo.save_dataframe(df, "repo_table", if_exists="replace")
        result = repo.get_records("repo_table")
        assert len(result) == 2

    def test_save_records_batch(self, repo):
        records = [{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}]
        repo.save_records(records, "batch_table")
        result = repo.get_records("batch_table")
        assert len(result) == 2

    def test_table_exists(self, repo):
        assert repo.table_exists("nonexistent") is False
        repo.save_dataframe(pd.DataFrame({"v": [1]}), "exists_table", if_exists="replace")
        assert repo.table_exists("exists_table") is True

    def test_update_record(self, repo):
        repo.save_dataframe(pd.DataFrame({"id": [1], "val": [5]}), "upd_tbl", if_exists="replace")
        repo.update_record("upd_tbl", primary_key="id", key_value=1, updates={"val": 42})
        result = repo.get_records("upd_tbl")
        assert int(result["val"].iloc[0]) == 42

    def test_close_does_not_raise(self, repo):
        repo.close()


# ---------------------------------------------------------------------------
# get_postgres_db_engine (smoke test – does not connect to real DB)
# ---------------------------------------------------------------------------

class TestGetPostgresDbEngine:
    def test_returns_engine_without_connecting(self, monkeypatch):
        """
        get_postgres_db_engine builds a PostgreSQLEngine object.
        We monkeypatch create_engine so it never tries to open a socket.
        """
        import database.database_utils as dbu
        from unittest.mock import MagicMock

        fake_engine = MagicMock()
        monkeypatch.setattr(dbu, "DatabaseFactory", MagicMock(
            create_engine=MagicMock(return_value=fake_engine)
        ))
        result = get_postgres_db_engine()
        assert result is fake_engine
