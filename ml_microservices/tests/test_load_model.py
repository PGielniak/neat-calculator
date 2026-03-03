"""
Unit tests for prediction_api/load_model.py

MLflow is fully mocked so no real tracking server is needed.
"""
import os
from unittest.mock import MagicMock, patch, call
import tempfile

import pytest

from prediction_api.load_model import load_production_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlflow_mock(model=None, scaler_exists=True, le_exists=True):
    """Build a minimal MLflow mock that mimics a successful model load."""
    mlflow_mock = MagicMock()

    # Returned model
    loaded_model = model or MagicMock()
    mlflow_mock.xgboost.load_model.return_value = loaded_model

    # MlflowClient
    mv_mock = MagicMock()
    mv_mock.run_id = "run-abc123"
    client_mock = MagicMock()
    client_mock.get_model_version_by_alias.return_value = mv_mock
    mlflow_mock.MlflowClient.return_value = client_mock

    # download_artifacts returns a path that optionally exists
    def _download(run_id, artifact_path):
        fd, tmp = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        if not ((artifact_path == "scaler.pkl" and scaler_exists) or
                (artifact_path == "label_encoder.pkl" and le_exists)):
            os.unlink(tmp)  # Remove the file so os.path.exists returns False
        return tmp

    mlflow_mock.artifacts.download_artifacts.side_effect = _download

    return mlflow_mock, loaded_model


# ---------------------------------------------------------------------------
# load_production_model
# ---------------------------------------------------------------------------

class TestLoadProductionModel:
    def test_returns_model_scaler_and_encoder(self):
        import joblib
        mlflow_mock, loaded_model = _make_mlflow_mock()

        with (
            patch("prediction_api.load_model.mlflow", mlflow_mock),
            patch("prediction_api.load_model.MlflowClient",
                  mlflow_mock.MlflowClient),
            patch("prediction_api.load_model.joblib.load") as mock_joblib,
        ):
            mock_joblib.side_effect = lambda path: MagicMock()
            model, scaler, le = load_production_model("har-xgboost", "production")

        assert model is loaded_model
        assert scaler is not None
        assert le is not None

    def test_returns_none_when_artifacts_not_found(self):
        mlflow_mock, loaded_model = _make_mlflow_mock(scaler_exists=False, le_exists=False)

        with (
            patch("prediction_api.load_model.mlflow", mlflow_mock),
            patch("prediction_api.load_model.MlflowClient",
                  mlflow_mock.MlflowClient),
        ):
            model, scaler, le = load_production_model("har-xgboost", "production")

        assert model is loaded_model
        assert scaler is None
        assert le is None

    def test_model_is_loaded_from_correct_uri(self):
        mlflow_mock, _ = _make_mlflow_mock()

        with (
            patch("prediction_api.load_model.mlflow", mlflow_mock),
            patch("prediction_api.load_model.MlflowClient",
                  mlflow_mock.MlflowClient),
            patch("prediction_api.load_model.joblib.load", return_value=MagicMock()),
        ):
            load_production_model("my-model", "staging")

        mlflow_mock.xgboost.load_model.assert_called_once_with(
            "models:/my-model@staging"
        )

    def test_handles_client_exception_gracefully(self):
        mlflow_mock = MagicMock()
        loaded_model = MagicMock()
        mlflow_mock.xgboost.load_model.return_value = loaded_model
        mlflow_mock.MlflowClient.return_value.get_model_version_by_alias.side_effect = Exception(
            "registry error"
        )

        with (
            patch("prediction_api.load_model.mlflow", mlflow_mock),
            patch("prediction_api.load_model.MlflowClient",
                  mlflow_mock.MlflowClient),
        ):
            model, scaler, le = load_production_model("har-xgboost", "production")

        # Model loaded but artifacts fall back to None on exception
        assert model is loaded_model
        assert scaler is None
        assert le is None
