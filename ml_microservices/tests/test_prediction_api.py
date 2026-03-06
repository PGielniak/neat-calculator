"""
Unit tests for prediction_api/prediction_api.py

The module executes _init_mlflow() and _load_models() at import time.
Both are neutralised by patching mlflow in sys.modules before the import.

Tests cover:
  - _apply_activity_taxes
  - drop_unnecessary_columns
  - _fetch_model_info
  - _init_mlflow
  - validate_and_cache_api_key (cache-down 503)
  - check_rate_limit (cache-down 503)
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Patch mlflow in sys.modules so prediction_api can be imported safely.
# ---------------------------------------------------------------------------

os.environ.setdefault("MLFLOW_BACKEND_STORE_URI", "sqlite:///test.db")

_mlflow_mock = MagicMock()
_mlflow_mock.get_tracking_uri.return_value = "sqlite:///test.db"
_mlflow_mock.exceptions = types.SimpleNamespace(MlflowException=Exception)

# Inject the mock BEFORE any import of prediction_api
# Only remove prediction_api.prediction_api (not load_model, to avoid stale references)
for _key in list(sys.modules):
    if _key == "prediction_api.prediction_api":
        del sys.modules[_key]

sys.modules.setdefault("mlflow", _mlflow_mock)
sys.modules.setdefault("mlflow.exceptions", _mlflow_mock.exceptions)

import prediction_api.prediction_api as _pa

# Reset module-level globals so tests start clean
_pa.model = None
_pa.scaler = None
_pa.label_encoder = None
_pa.model_info = None


# ---------------------------------------------------------------------------
# _apply_activity_taxes
# ---------------------------------------------------------------------------

class TestApplyActivityTaxes:
    _thresholds = {
        "WALKING_UPSTAIRS": 0.65,
        "WALKING_DOWNSTAIRS": 0.40,
    }

    def test_below_threshold_returns_walking(self):
        result = _pa._apply_activity_taxes("WALKING_UPSTAIRS", 0.50, self._thresholds)
        assert result == "WALKING"

    def test_above_threshold_returns_original(self):
        result = _pa._apply_activity_taxes("WALKING_UPSTAIRS", 0.70, self._thresholds)
        assert result == "WALKING_UPSTAIRS"

    def test_exact_threshold_returns_original(self):
        # confidence == threshold is NOT < threshold, so original is returned
        result = _pa._apply_activity_taxes("WALKING_UPSTAIRS", 0.65, self._thresholds)
        assert result == "WALKING_UPSTAIRS"

    def test_activity_not_in_thresholds_returns_original(self):
        result = _pa._apply_activity_taxes("WALKING", 0.10, self._thresholds)
        assert result == "WALKING"

    def test_downstairs_below_threshold_returns_walking(self):
        result = _pa._apply_activity_taxes("WALKING_DOWNSTAIRS", 0.30, self._thresholds)
        assert result == "WALKING"

    def test_downstairs_above_threshold_returns_original(self):
        result = _pa._apply_activity_taxes("WALKING_DOWNSTAIRS", 0.50, self._thresholds)
        assert result == "WALKING_DOWNSTAIRS"

    def test_standing_with_any_confidence_returns_standing(self):
        result = _pa._apply_activity_taxes("STANDING", 0.01, self._thresholds)
        assert result == "STANDING"

    def test_empty_thresholds_returns_prediction_unchanged(self):
        result = _pa._apply_activity_taxes("LAYING", 0.0, {})
        assert result == "LAYING"


# ---------------------------------------------------------------------------
# drop_unnecessary_columns
# ---------------------------------------------------------------------------

class TestDropUnnecessaryColumns:
    def test_drops_matching_columns(self):
        df = pd.DataFrame({"angle_x": [1], "tBodyAcc-X": [2], "other": [3]})
        result = _pa.drop_unnecessary_columns(df, ["angle", "tBodyAcc-X"])
        assert "angle_x" not in result.columns
        assert "tBodyAcc-X" not in result.columns
        assert "other" in result.columns

    def test_no_match_returns_all_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = _pa.drop_unnecessary_columns(df, ["xyz"])
        assert list(result.columns) == ["a", "b"]

    def test_empty_patterns_returns_all_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = _pa.drop_unnecessary_columns(df, [])
        assert list(result.columns) == ["a", "b"]

    def test_returns_dataframe(self):
        df = pd.DataFrame({"x": [1]})
        result = _pa.drop_unnecessary_columns(df, ["x"])
        assert isinstance(result, pd.DataFrame)

    def test_partial_pattern_match(self):
        df = pd.DataFrame({"tBodyAcc-mean()-X": [1], "fBodyAcc-mean()-X": [2], "label": [3]})
        result = _pa.drop_unnecessary_columns(df, ["tBodyAcc"])
        assert "tBodyAcc-mean()-X" not in result.columns
        assert "fBodyAcc-mean()-X" in result.columns

    def test_errors_ignored_for_missing_columns(self):
        df = pd.DataFrame({"keep": [1]})
        result = _pa.drop_unnecessary_columns(df, ["nonexistent"])
        assert "keep" in result.columns


# ---------------------------------------------------------------------------
# _fetch_model_info
# ---------------------------------------------------------------------------

class TestFetchModelInfo:
    def test_returns_dict_on_success(self):
        mock_client = MagicMock()
        mock_mv = MagicMock()
        mock_mv.name = "model"
        mock_mv.version = "1"
        mock_mv.current_stage = "Production"
        mock_mv.run_id = "abc123"
        mock_mv.description = "test"
        mock_mv.creation_timestamp = 0
        mock_mv.last_updated_timestamp = 0
        mock_client.get_model_version_by_alias.return_value = mock_mv
        mock_client.get_run.return_value = MagicMock(
            info=MagicMock(run_name="run1", status="FINISHED"),
            data=MagicMock(metrics={})
        )

        with patch("prediction_api.prediction_api.MlflowClient", return_value=mock_client):
            result = _pa._fetch_model_info()

        assert isinstance(result, dict)
        assert result["version"] == "1"

    def test_returns_none_on_exception(self):
        with patch("prediction_api.prediction_api.MlflowClient",
                   side_effect=Exception("connection error")):
            result = _pa._fetch_model_info()
        assert result is None


# ---------------------------------------------------------------------------
# _init_mlflow
# ---------------------------------------------------------------------------

class TestInitMlflow:
    def test_raises_when_uri_not_set(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_BACKEND_STORE_URI", raising=False)
        with pytest.raises(ValueError, match="MLFLOW_BACKEND_STORE_URI"):
            _pa._init_mlflow()

    def test_sets_tracking_uri(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///test.db")
        with patch("prediction_api.prediction_api.mlflow") as mock_mlflow:
            _pa._init_mlflow()
            mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///test.db")


# ---------------------------------------------------------------------------
# validate_and_cache_api_key – cache-down 503
# ---------------------------------------------------------------------------

class TestValidateAndCacheApiKey:
    def test_raises_503_on_cache_lookup_error(self):
        mock_cache = MagicMock()
        mock_cache.hgetall.side_effect = _pa.redis.exceptions.RedisError("connection refused")
        with patch("prediction_api.prediction_api.get_cache", return_value=mock_cache):
            from fastapi import HTTPException as _HTTPException
            with pytest.raises(_HTTPException) as exc_info:
                _pa.validate_and_cache_api_key("test-key")
        assert exc_info.value.status_code == 503

    def test_raises_503_on_cache_write_error(self):
        mock_cache = MagicMock()
        mock_cache.hgetall.return_value = {}  # cache miss → will call the key service
        mock_cache.hset.side_effect = _pa.redis.exceptions.RedisError("connection refused")
        with patch("prediction_api.prediction_api.get_cache", return_value=mock_cache), \
             patch("prediction_api.prediction_api.requests") as mock_requests:
            mock_requests.post.return_value.json.return_value = {"valid": True, "rate_limit_req_no": 30, "rate_limit_interval_minutes": 1}
            from fastapi import HTTPException as _HTTPException
            with pytest.raises(_HTTPException) as exc_info:
                _pa.validate_and_cache_api_key("test-key")
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# check_rate_limit – cache-down 503
# ---------------------------------------------------------------------------

class TestCheckRateLimit:
    def test_raises_503_on_cache_error(self):
        mock_cache = MagicMock()
        mock_cache.hgetall.side_effect = _pa.redis.exceptions.RedisError("connection refused")
        with patch("prediction_api.prediction_api.get_cache", return_value=mock_cache):
            from fastapi import HTTPException as _HTTPException
            with pytest.raises(_HTTPException) as exc_info:
                _pa.check_rate_limit("prefix", "keyhash")
        assert exc_info.value.status_code == 503

    def test_raises_503_on_incr_error(self):
        mock_cache = MagicMock()
        mock_cache.hgetall.return_value = {"rate_limit_req_no": "30", "rate_limit_interval_minutes": "1"}
        mock_cache.incr.side_effect = _pa.redis.exceptions.RedisError("READONLY")
        with patch("prediction_api.prediction_api.get_cache", return_value=mock_cache):
            from fastapi import HTTPException as _HTTPException
            with pytest.raises(_HTTPException) as exc_info:
                _pa.check_rate_limit("prefix", "keyhash")
        assert exc_info.value.status_code == 503

    def test_returns_true_when_within_limit(self):
        mock_cache = MagicMock()
        mock_cache.hgetall.return_value = {"rate_limit_req_no": "30", "rate_limit_interval_minutes": "1"}
        mock_cache.incr.return_value = 1
        with patch("prediction_api.prediction_api.get_cache", return_value=mock_cache):
            result = _pa.check_rate_limit("prefix", "keyhash")
        assert result is True

    def test_returns_false_when_limit_exceeded(self):
        mock_cache = MagicMock()
        mock_cache.hgetall.return_value = {"rate_limit_req_no": "5", "rate_limit_interval_minutes": "1"}
        mock_cache.incr.return_value = 6
        with patch("prediction_api.prediction_api.get_cache", return_value=mock_cache):
            result = _pa.check_rate_limit("prefix", "keyhash")
        assert result is False
