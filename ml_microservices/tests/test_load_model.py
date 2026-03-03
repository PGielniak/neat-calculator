"""
Unit tests for prediction_api/load_model.py

Tests cover:
  - load_production_model  (model loaded from MLflow, scaler/encoder artifacts)

All MLflow SDK calls are mocked.
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure mlflow is mocked before prediction_api.load_model is imported.
# ---------------------------------------------------------------------------

os.environ.setdefault("MLFLOW_BACKEND_STORE_URI", "sqlite:///test.db")

_mlflow_mock = MagicMock()
_mlflow_mock.get_tracking_uri.return_value = "sqlite:///test.db"
_mlflow_mock.exceptions = types.SimpleNamespace(MlflowException=Exception)

sys.modules.setdefault("mlflow", _mlflow_mock)
sys.modules.setdefault("mlflow.exceptions", _mlflow_mock.exceptions)

# Import here to ensure the module is in sys.modules before tests run
from prediction_api.load_model import load_production_model  # noqa: E402
import prediction_api.load_model as _lm  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadProductionModel:
    """Tests for load_production_model using mocked MLflow SDK."""

    def _mock_client(self, run_id: str = "run-abc"):
        mock_client = MagicMock()
        mock_mv = MagicMock()
        mock_mv.run_id = run_id
        mock_client.get_model_version_by_alias.return_value = mock_mv
        return mock_client

    def test_returns_model_scaler_encoder_tuple(self):
        mock_model = MagicMock()
        mock_client = self._mock_client()

        with patch.object(_lm, "mlflow") as mock_mlflow, \
             patch.object(_lm, "MlflowClient", return_value=mock_client), \
             patch.object(_lm.os.path, "exists", return_value=True), \
             patch.object(_lm, "joblib") as mock_joblib:
            mock_mlflow.sklearn.load_model.return_value = mock_model
            mock_mlflow.artifacts.download_artifacts.side_effect = [
                "/tmp/scaler.pkl", "/tmp/label_encoder.pkl"
            ]
            mock_joblib.load.side_effect = [MagicMock(), MagicMock()]

            model, scaler, encoder = _lm.load_production_model("har-model", "production")

        assert model is not None

    def test_scaler_none_when_artifact_file_missing(self):
        mock_client = self._mock_client()

        with patch.object(_lm, "mlflow") as mock_mlflow, \
             patch.object(_lm, "MlflowClient", return_value=mock_client), \
             patch.object(_lm.os.path, "exists", return_value=False):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/file.pkl"

            model, scaler, encoder = _lm.load_production_model("har-model", "production")

        assert scaler is None
        assert encoder is None

    def test_random_forest_uses_sklearn_loader(self):
        mock_client = self._mock_client()

        with patch.object(_lm, "mlflow") as mock_mlflow, \
             patch.object(_lm, "MlflowClient", return_value=mock_client), \
             patch.object(_lm.os.path, "exists", return_value=False):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/file.pkl"

            _lm.load_production_model("model", "production", algorithm="random_forest")
            mock_mlflow.sklearn.load_model.assert_called_once()

    def test_xgboost_uses_xgboost_loader(self):
        mock_client = self._mock_client()

        with patch.object(_lm, "mlflow") as mock_mlflow, \
             patch.object(_lm, "MlflowClient", return_value=mock_client), \
             patch.object(_lm.os.path, "exists", return_value=False):
            mock_mlflow.xgboost.load_model.return_value = MagicMock()
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/file.pkl"

            _lm.load_production_model("model", "production", algorithm="xgboost")
            mock_mlflow.xgboost.load_model.assert_called_once()

    def test_artifact_download_failure_returns_none_for_artifacts(self):
        mock_client = self._mock_client()

        with patch.object(_lm, "mlflow") as mock_mlflow, \
             patch.object(_lm, "MlflowClient", return_value=mock_client):
            mock_mlflow.sklearn.load_model.return_value = MagicMock()
            mock_mlflow.artifacts.download_artifacts.side_effect = Exception("not found")

            model, scaler, encoder = _lm.load_production_model("model", "production")
            assert scaler is None
            assert encoder is None

    def test_model_uri_format(self):
        mock_client = self._mock_client()
        captured_uri = []

        with patch.object(_lm, "mlflow") as mock_mlflow, \
             patch.object(_lm, "MlflowClient", return_value=mock_client), \
             patch.object(_lm.os.path, "exists", return_value=False):
            def capture_uri(uri):
                captured_uri.append(uri)
                return MagicMock()
            mock_mlflow.sklearn.load_model.side_effect = capture_uri
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/file.pkl"

            _lm.load_production_model("my-model", "staging", algorithm="random_forest")

        assert len(captured_uri) == 1
        assert captured_uri[0] == "models:/my-model@staging"
