"""
Unit tests for model_training/train_model.py

MLflow and database calls are mocked so the tests run offline.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from model_training.train_model import (
    balance_classes,
    drop_columns_with_too_much_importance,
    get_required_env,
    prepare_data_for_training,
    save_ml_artifacts_to_file,
    find_best_model,
    prepare_variables,
    setup_and_test_mlflow_connection,
    load_data_fromdb,
    load_kaggle_data_fromdb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activity_df(n_per_class: int = 30) -> pd.DataFrame:
    """Return a small DataFrame mimicking training data."""
    activities = ["WALKING", "SITTING", "STANDING", "LAYING"]
    rng = np.random.default_rng(0)
    rows = []
    for act in activities:
        for _ in range(n_per_class):
            row = {f"feature_{i}": rng.standard_normal() for i in range(5)}
            row["Activity"] = act
            row["timestamp"] = 1_700_000_000_000
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_required_env
# ---------------------------------------------------------------------------

class TestGetRequiredEnv:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello")
        from model_training.train_model import get_required_env
        assert get_required_env("MY_TEST_VAR") == "hello"

    def test_raises_when_not_set(self, monkeypatch):
        monkeypatch.delenv("MY_MISSING_VAR", raising=False)
        with pytest.raises(ValueError):
            get_required_env("MY_MISSING_VAR")


# ---------------------------------------------------------------------------
# balance_classes
# ---------------------------------------------------------------------------

class TestBalanceClasses:
    def test_least_represented_balances_to_min_count(self):
        df = _make_activity_df()
        # Make one class smaller
        df = pd.concat([df, pd.DataFrame([{"feature_0": 0.0, "feature_1": 0.0,
                                            "feature_2": 0.0, "feature_3": 0.0,
                                            "feature_4": 0.0, "Activity": "WALKING",
                                            "timestamp": 1}] * 5)], ignore_index=True)
        result = asyncio.get_event_loop().run_until_complete(
            balance_classes(df, method="least_represented")
        )
        counts = result["Activity"].value_counts()
        assert counts.max() == counts.min()

    def test_cap_at_median(self):
        df = _make_activity_df(n_per_class=40)
        result = asyncio.get_event_loop().run_until_complete(
            balance_classes(df, method="cap_at_median")
        )
        assert len(result) > 0
        assert "Activity" in result.columns

    def test_undersample_highest(self):
        df = _make_activity_df(n_per_class=30)
        result = asyncio.get_event_loop().run_until_complete(
            balance_classes(df, method="undersample_highest")
        )
        assert len(result) > 0

    def test_unknown_method_returns_original_data(self):
        df = _make_activity_df()
        result = asyncio.get_event_loop().run_until_complete(
            balance_classes(df, method="unknown_method")
        )
        assert len(result) == len(df)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            asyncio.get_event_loop().run_until_complete(
                balance_classes(pd.DataFrame(), method="least_represented")
            )

    def test_missing_activity_column_raises(self):
        df = pd.DataFrame({"feature": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Activity"):
            asyncio.get_event_loop().run_until_complete(
                balance_classes(df, method="least_represented")
            )


# ---------------------------------------------------------------------------
# drop_columns_with_too_much_importance
# ---------------------------------------------------------------------------

class TestDropColumnsWithTooMuchImportance:
    def test_drops_angle_columns(self):
        df = pd.DataFrame({
            "angle(X,gravityMean)": [1.0],
            "tBodyAcc-X-something": [2.0],  # matches 'tBodyAcc-X' pattern
            "other_feature": [3.0],
        })
        result = asyncio.get_event_loop().run_until_complete(
            drop_columns_with_too_much_importance(df)
        )
        assert "angle(X,gravityMean)" not in result.columns
        assert "tBodyAcc-X-something" not in result.columns
        assert "other_feature" in result.columns

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            asyncio.get_event_loop().run_until_complete(
                drop_columns_with_too_much_importance(pd.DataFrame())
            )

    def test_no_matching_columns_unchanged(self):
        df = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0]})
        result = asyncio.get_event_loop().run_until_complete(
            drop_columns_with_too_much_importance(df)
        )
        assert set(result.columns) == {"feat_a", "feat_b"}


# ---------------------------------------------------------------------------
# prepare_data_for_training
# ---------------------------------------------------------------------------

class TestPrepareDataForTraining:
    def test_returns_splits_and_encoder(self):
        df = _make_activity_df(n_per_class=20)
        result = asyncio.get_event_loop().run_until_complete(
            prepare_data_for_training(df)
        )
        X_train, X_test, y_train, y_test, label_encoder, scaler = result
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            asyncio.get_event_loop().run_until_complete(
                prepare_data_for_training(pd.DataFrame())
            )

    def test_missing_activity_column_raises(self):
        df = pd.DataFrame({"feature": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Activity"):
            asyncio.get_event_loop().run_until_complete(
                prepare_data_for_training(df)
            )


# ---------------------------------------------------------------------------
# save_ml_artifacts_to_file
# ---------------------------------------------------------------------------

class TestSaveMLArtifactsToFile:
    def test_saves_artifacts_and_returns_paths(self, tmp_path, monkeypatch):
        # Run from tmp_path so temp_artifact_store is created there
        monkeypatch.chdir(tmp_path)
        # Use a real picklable object (list) instead of MagicMock
        paths, directory = asyncio.get_event_loop().run_until_complete(
            save_ml_artifacts_to_file(scaler=[1, 2, 3])
        )
        assert len(paths) == 1
        assert paths[0].name == "scaler.pkl"
        assert directory.exists()

    def test_multiple_artifacts_saved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        paths, _ = asyncio.get_event_loop().run_until_complete(
            save_ml_artifacts_to_file(scaler=[1.0], label_encoder={"a": 0})
        )
        names = [p.name for p in paths]
        assert "scaler.pkl" in names
        assert "label_encoder.pkl" in names


# ---------------------------------------------------------------------------
# find_best_model
# ---------------------------------------------------------------------------

class TestFindBestModel:
    def test_returns_model_and_params(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((60, 4)), columns=[f"f{i}" for i in range(4)])
        y = np.array([0, 1, 2] * 20)
        model, best_params, mean_cv, std_cv, col_names = find_best_model(X, y, random_search_cv=False)
        assert model is not None
        assert isinstance(best_params, dict)
        assert 0.0 <= mean_cv <= 1.0
        assert std_cv >= 0.0


# ---------------------------------------------------------------------------
# prepare_variables (async, mocked env)
# ---------------------------------------------------------------------------

class TestPrepareVariables:
    def test_returns_backend_and_artifact_uri(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "postgresql://user:pass@host/db")
        monkeypatch.setenv("MLFLOW_ARTIFACT_URI", "wasbs://container@account.blob.core.windows.net")
        backend, artifact = asyncio.get_event_loop().run_until_complete(prepare_variables())
        assert "postgresql" in backend
        assert "wasbs" in artifact

    def test_raises_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_BACKEND_STORE_URI", raising=False)
        monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(prepare_variables())


# ---------------------------------------------------------------------------
# setup_and_test_mlflow_connection
# ---------------------------------------------------------------------------

class TestSetupAndTestMlflowConnection:
    def test_raises_on_connection_failure(self):
        with patch("model_training.train_model.mlflow") as mock_mlflow:
            mock_mlflow.set_tracking_uri.return_value = None
            mock_mlflow.get_tracking_uri.return_value = "postgresql://..."
            mock_mlflow.search_experiments.side_effect = Exception("connection refused")
            with pytest.raises(ValueError, match="Connection failed"):
                asyncio.get_event_loop().run_until_complete(
                    setup_and_test_mlflow_connection("postgresql://host/db")
                )

    def test_succeeds_when_mlflow_responds(self):
        with patch("model_training.train_model.mlflow") as mock_mlflow:
            mock_mlflow.set_tracking_uri.return_value = None
            mock_mlflow.get_tracking_uri.return_value = "postgresql://..."
            mock_mlflow.search_experiments.return_value = []
            # Should not raise
            asyncio.get_event_loop().run_until_complete(
                setup_and_test_mlflow_connection("postgresql://host/db")
            )


# ---------------------------------------------------------------------------
# load_data_fromdb
# ---------------------------------------------------------------------------

class TestLoadDataFromdb:
    def test_returns_dataframe(self, monkeypatch):
        monkeypatch.setenv("LABELED_TRAINING_DATA_TABLE_NAME", "test_table")
        fake_engine = MagicMock()
        fake_engine.get_records.return_value = pd.DataFrame({"Activity": ["WALKING"]})
        result = asyncio.get_event_loop().run_until_complete(load_data_fromdb(fake_engine))
        assert isinstance(result, pd.DataFrame)

    def test_raises_when_table_name_missing(self, monkeypatch):
        monkeypatch.delenv("LABELED_TRAINING_DATA_TABLE_NAME", raising=False)
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(load_data_fromdb(MagicMock()))

    def test_raises_when_empty_data_returned(self, monkeypatch):
        monkeypatch.setenv("LABELED_TRAINING_DATA_TABLE_NAME", "empty_table")
        fake_engine = MagicMock()
        fake_engine.get_records.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            asyncio.get_event_loop().run_until_complete(load_data_fromdb(fake_engine))


# ---------------------------------------------------------------------------
# load_kaggle_data_fromdb
# ---------------------------------------------------------------------------

class TestLoadKaggleDataFromdb:
    def test_loads_with_column_subset(self):
        fake_engine = MagicMock()
        fake_engine.get_records.return_value = pd.DataFrame({
            "tBodyAcc-mean()-X": [0.1],
            "Activity": ["WALKING"],
        })
        result = asyncio.get_event_loop().run_until_complete(
            load_kaggle_data_fromdb(fake_engine, columns=["tBodyAcc-mean()-X", "Activity"])
        )
        assert "Activity" in result.columns
