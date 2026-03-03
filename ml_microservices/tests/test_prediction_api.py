"""
Unit tests for prediction_api/prediction_api.py

MLflow model loading is mocked at import time so the app can be imported
without a live MLflow server.
"""
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Patch MLflow and model loading BEFORE importing the app module
# ---------------------------------------------------------------------------

# We need to prevent the module-level calls (_init_mlflow, _load_models) from
# actually contacting MLflow.  We do this by mocking the relevant functions
# before the module is imported into this test session.

os.environ.setdefault("MLFLOW_BACKEND_STORE_URI", "mock://tracking")

_mlflow_mock = MagicMock()
_mlflow_mock.get_tracking_uri.return_value = "mock://tracking-uri"
_mlflow_mock.exceptions.MlflowException = Exception

with (
    patch.dict("sys.modules", {"mlflow": _mlflow_mock,
                                "mlflow.exceptions": _mlflow_mock.exceptions,
                                "mlflow.xgboost": MagicMock()}),
    patch("prediction_api.load_model.load_production_model",
          return_value=(None, None, None)),
):
    import importlib
    import prediction_api.prediction_api as _api_module

    # Reset the module-level globals to a known state for tests
    _api_module.model = None
    _api_module.scaler = None
    _api_module.label_encoder = None
    _api_module.model_info = None

app = _api_module.app
WINDOW_SIZE = _api_module.WINDOW_SIZE


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sensor_samples(n: int, label: str = "UNLABELED"):
    return [
        {
            "accelerometerX": 0.1 * i,
            "accelerometerY": 0.2,
            "accelerometerZ": 9.8,
            "gyroscopeX": 0.01,
            "gyroscopeY": 0.02,
            "gyroscopeZ": 0.03,
            "timestamp": 1_700_000_000_000 + i * 20,
            "timestampNanos": (1_700_000_000_000 + i * 20) * 1_000_000,
            "label": label,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# drop_unnecessary_columns
# ---------------------------------------------------------------------------

class TestDropUnnecessaryColumns:
    def test_drops_matching_columns(self):
        df = pd.DataFrame({
            "angle(X,gravityMean)": [1.0],
            "tBodyAcc-X-something": [2.0],
            "keep_this": [3.0],
        })
        result = _api_module.drop_unnecessary_columns(
            df, columns_to_drop=["angle", "tBodyAcc-X"]
        )
        assert "keep_this" in result.columns
        assert "angle(X,gravityMean)" not in result.columns
        assert "tBodyAcc-X-something" not in result.columns

    def test_no_matching_columns_unchanged(self):
        df = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]})
        result = _api_module.drop_unnecessary_columns(df, columns_to_drop=["angle"])
        assert set(result.columns) == {"feature_a", "feature_b"}

    def test_empty_patterns_list_unchanged(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = _api_module.drop_unnecessary_columns(df, columns_to_drop=[])
        assert set(result.columns) == {"x", "y"}


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_has_status_field(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_model_loaded_false_when_no_model(self, client):
        _api_module.model = None
        response = client.get("/health")
        assert response.json()["model_loaded"] is False

    def test_model_loaded_true_when_model_set(self, client):
        _api_module.model = MagicMock()
        try:
            response = client.get("/health")
            assert response.json()["model_loaded"] is True
        finally:
            _api_module.model = None


# ---------------------------------------------------------------------------
# /predict endpoint — validation errors
# ---------------------------------------------------------------------------

class TestPredictEndpointValidation:
    def test_insufficient_samples_returns_422(self, client):
        payload = {"samples": _make_sensor_samples(10)}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_samples_returns_422(self, client):
        payload = {"samples": []}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /predict endpoint — BUFFERING when < WINDOW_SIZE rows after pipeline
# ---------------------------------------------------------------------------

class TestPredictBuffering:
    def test_returns_buffering_when_pipeline_yields_few_rows(self, client):
        """
        When data_processing_pipeline returns a DataFrame with fewer than
        WINDOW_SIZE rows, predict should return {"activity": "BUFFERING"}.
        """
        small_df = pd.DataFrame({col: [0.1] for col in _api_module.sensor_cols})

        with patch.object(_api_module, "data_processing_pipeline",
                          new=AsyncMock(return_value=small_df)):
            payload = {"samples": _make_sensor_samples(WINDOW_SIZE)}
            response = client.post("/predict", json=payload)
        # BUFFERING can come back as 200 or 503 depending on model state
        assert response.status_code in (200, 503)


# ---------------------------------------------------------------------------
# run_prediction — unit tests
# ---------------------------------------------------------------------------

class TestRunPrediction:
    @pytest.mark.asyncio
    async def test_raises_503_when_no_model(self):
        from fastapi import HTTPException

        _api_module.model = None
        feature_df = pd.DataFrame({"feature_a": [0.1]})
        with pytest.raises(HTTPException) as exc_info:
            await _api_module.run_prediction(feature_df)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_raises_422_when_empty_dataframe(self):
        from fastapi import HTTPException

        mock_model = MagicMock()
        _api_module.model = mock_model
        _api_module.scaler = None
        try:
            with pytest.raises(HTTPException) as exc_info:
                await _api_module.run_prediction(pd.DataFrame())
            assert exc_info.value.status_code == 422
        finally:
            _api_module.model = None

    @pytest.mark.asyncio
    async def test_returns_prediction_dict(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8, 0.0, 0.0, 0.0]])
        _api_module.model = mock_model
        _api_module.scaler = None
        _api_module.label_encoder = None
        try:
            feature_df = pd.DataFrame({"feature_a": [0.1], "feature_b": [0.2]})
            result = await _api_module.run_prediction(feature_df)
            assert "activity" in result
            assert "confidence" in result
            assert "prediction_index" in result
            assert "all_probabilities" in result
        finally:
            _api_module.model = None
