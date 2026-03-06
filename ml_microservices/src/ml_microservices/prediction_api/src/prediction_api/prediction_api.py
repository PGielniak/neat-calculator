from json import load
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Annotated
import pandas as pd
import mlflow
from mlflow import MlflowClient
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
from shared.process_raw_data import remove_duplicates, resample_data, create_sliding_windows, extract_features_from_windows, rename_features, filter_features_to_match_kaggle
load_dotenv()
from prediction_api.load_model import load_production_model
import importlib.resources
from yarl import URL
import requests
import hashlib
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import redis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_NAME = os.getenv("MODEL_NAME", "har-randomforest01")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

API_KEY_SERVICE_URL = os.getenv("API_KEY_SERVICE_URL", "http://localhost:8007/api-keys/validate")
DRAGONFLY_URL = os.getenv("DRAGONFLY_URL", "redis://127.0.0.1:6379")
CACHE_TTL_SECONDS = int(os.getenv("API_KEY_CACHE_TTL", 300))  # 5 min default

model = None
scaler = None
label_encoder = None
model_info = None

_cache: redis.Redis | None = None

def get_cache() -> redis.Redis:
    global _cache
    if _cache is None:
        _cache = redis.from_url(DRAGONFLY_URL, decode_responses=True)
    return _cache

def _init_mlflow() -> None:
    uri = os.getenv("MLFLOW_BACKEND_STORE_URI")
    if not uri:
        raise ValueError("MLFLOW_BACKEND_STORE_URI environment variable is not set.")
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI set to: {uri}")


def _fetch_model_info() -> dict | None:
    """Fetch model version and run metadata from MLflow."""
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run = client.get_run(mv.run_id)
        info = {
            "name": mv.name,
            "version": mv.version,
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "description": mv.description or "No description available",
            "created": mv.creation_timestamp,
            "last_updated": mv.last_updated_timestamp,
        }
        logger.info(
            f"Model '{info['name']}' v{info['version']} | stage={info['stage']} | run={info['run_id']}"
        )
        logger.info(f"Run '{run.info.run_name}' status={run.info.status}")
        if run.data.metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in run.data.metrics.items())
            logger.info(f"Training metrics: {metrics_str}")
        return info
    except Exception as e:
        logger.warning(f"Could not fetch model metadata: {e}")
        return None


def _log_artifact_status() -> None:
    if model:
        logger.info("Model loaded and ready for predictions.")
    else:
        logger.warning("No model loaded. Predictions will fail until a model is available.")
    if scaler:
        logger.info("Scaler is available for data transformation.")
    else:
        logger.warning("No scaler loaded. Predictions on raw data may be incorrect.")
    if label_encoder is not None:
        logger.info(
            f"Label encoder loaded with {len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}"
        )
    else:
        logger.warning("Label encoder not loaded. Will use fallback activity labels.")


def _load_models() -> None:
    global model, scaler, label_encoder, model_info
    try:
        model, scaler, label_encoder = load_production_model(MODEL_NAME, MODEL_ALIAS)
        logger.info(f"Loaded model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'")
        model_info = _fetch_model_info()
    except mlflow.exceptions.MlflowException as e:
        if "not found" in str(e).lower():
            logger.warning(
                f"Model '{MODEL_NAME}' with alias '{MODEL_ALIAS}' not found in MLflow. "
                f"The API will start but predictions will fail until a model is registered."
            )
        else:
            logger.error(f"MLflow error while loading model: {e}")
    except Exception as e:
        logger.error(f"Could not load model: {e}")
    finally:
        _log_artifact_status()

def _apply_activity_taxes(prediction, confidence, thresholds):
    """
    thresholds = {
        'WALKING_UPSTAIRS': 0.65,
        'WALKING_DOWNSTAIRS': 0.55
    }
    """
    # Get the specific threshold for this prediction, default to 0.0
    required_confidence = thresholds.get(prediction, 0.0)
    
    if confidence < required_confidence:
        # If it fails the 'Stair Tax', it's almost certainly just Walking
        return 'WALKING'
    
    return prediction

def validate_and_cache_api_key(raw_key: str) -> bool:
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    cache_key = f"apikey:{key_hash}"
    cached = get_cache().hgetall(cache_key)
    if cached:
        return cached.get("valid") == "1"
    try:
        response = requests.post(API_KEY_SERVICE_URL, json={"raw_key": raw_key}, timeout=5)
        data = response.json()
    except Exception as exc:
        logger.error(f"API key validation failed: {exc}")
        return False
    get_cache().hset(cache_key, mapping={
        "valid": "1" if data.get("valid") else "0",
        "rate_limit_req_no": data.get("rate_limit_req_no", 30),
        "rate_limit_interval_minutes": data.get("rate_limit_interval_minutes", 1),
    })
    get_cache().expire(cache_key, CACHE_TTL_SECONDS)
    return bool(data.get("valid", False))


def check_rate_limit(prefix: str, key_hash: str) -> bool:
    """Fixed-window rate limit check using Dragonfly. Returns False if limit exceeded."""
    import time
    cached = get_cache().hgetall(f"apikey:{key_hash}")
    limit = int(cached.get("rate_limit_req_no", 30))
    interval_minutes = int(cached.get("rate_limit_interval_minutes", 1))
    interval_seconds = interval_minutes * 60
    window = int(time.time() / interval_seconds)
    key = f"ratelimit:{prefix}:{window}"
    count = get_cache().incr(key)
    if count == 1:
        get_cache().expire(key, interval_seconds + 1)
    return count <= limit


_init_mlflow()
_load_models()

limiter = Limiter(key_func=get_remote_address, storage_uri=DRAGONFLY_URL)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

WINDOW_SIZE = 128

# Activity class mappings based on UCI HAR dataset (used in training)
ACTIVITY_LABELS = {
    0: "LAYING",
    1: "SITTING", 
    2: "STANDING",
    3: "WALKING",
    4: "WALKING_DOWNSTAIRS",
    5: "WALKING_UPSTAIRS"
}

@app.get("/health")
@limiter.limit("1/minute")
async def health(request: Request):
    """Health check endpoint that shows model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_info": model_info
    }

class SensorRecording(BaseModel):
    accelerometerX: float
    accelerometerY: float
    accelerometerZ: float
    gyroscopeX: float
    gyroscopeY: float
    gyroscopeZ: float
    timestamp: int
    timestampNanos: int
    label: str = "UNLABELED"
    
sensor_cols = [
    "accelerometerX","accelerometerY","accelerometerZ",
    "gyroscopeX","gyroscopeY","gyroscopeZ",
    ]

class PredictionRequest(BaseModel):
    samples: List[SensorRecording]

    # Walidator Pydantic - sprawdzamy długość PRZED uruchomieniem logiki ML
    @field_validator('samples')
    def validate_length(cls, v):
        if len(v) < WINDOW_SIZE:
            raise ValueError(f"Not enough data! Expected at least {WINDOW_SIZE}, got {len(v)}")
        return v

@app.post("/predict")
@limiter.limit("120/minute")
async def predict(request: Request, payload: PredictionRequest, x_api_key_header: Annotated[str | None, Header(alias="X-Api-Key")] = None):

    if not x_api_key_header:
        raise HTTPException(status_code=401, detail="Failed to authenticate the request.")

    if not validate_and_cache_api_key(x_api_key_header):
        raise HTTPException(status_code=401, detail="Failed to authenticate the request.")

    x_api_key_prefix = x_api_key_header.split("_")[0]
    key_hash = hashlib.sha256(x_api_key_header.encode()).hexdigest()

    if not check_rate_limit(x_api_key_prefix, key_hash):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    logger.info(f"Proceeding with prediction for api key prefix {x_api_key_prefix}")
    
    samples_dict = [sample.model_dump() for sample in payload.samples]
    df = pd.DataFrame(samples_dict)

    logger.info(f"Received {df.shape[0]} samples with columns: {df.columns.tolist()}")

    if len(df) > WINDOW_SIZE:
        df = df.iloc[-WINDOW_SIZE:].reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        return {"activity": "BUFFERING", "confidence": 0.0}

    feature_df = await data_processing_pipeline(df)

    predictions = await run_prediction(feature_df)

    logger.info(f"Prediction: activity={predictions.get('activity')}, confidence={predictions.get('confidence'):.4f}")

    return predictions

async def data_processing_pipeline(window_df: pd.DataFrame):
    deduped_data = remove_duplicates(window_df, sensor_cols=sensor_cols)
    resampled_data = resample_data(deduped_data, target_freq=50, sensor_cols=sensor_cols)
    del deduped_data
    sliding_windows, feature_array, labels_array, timestamp_array = create_sliding_windows(resampled_data, window_size=128, step_size=64, sensor_cols=sensor_cols)
    del resampled_data
    extracted_features = extract_features_from_windows(sliding_windows, feature_array, labels_array, timestamp_array)
    del sliding_windows, feature_array, labels_array  # free up memory
    renamed_features = rename_features(extracted_features)
    del extracted_features  # free up memory
    kaggle_csv_path = str(importlib.resources.files("shared").joinpath("kaggle.csv"))
    intersect_with_kaggle = filter_features_to_match_kaggle(renamed_features, kaggle_csv_path=kaggle_csv_path)   
    del renamed_features  # free up memory
    feature_df = pd.DataFrame(intersect_with_kaggle)
    # feature_df_without_columns = drop_unnecessary_columns(feature_df, columns_to_drop=['angle', 'tGravityAcc-X', 'tGravityAcc-Y', 'tGravityAcc-Z', 
    #                     'tBodyAcc-X', 'tBodyAcc-Y', 'tBodyAcc-Z',
    #                     'fBodyAcc-X', 'fBodyAcc-Y', 'fBodyAcc-Z'])
    return feature_df

def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    cols_to_drop = [col for col in df.columns if any(pat in col for pat in columns_to_drop)]
    logger.debug(f"Dropping {len(cols_to_drop)} columns matching patterns. Shape before: {df.shape}")
    df = df.drop(columns=cols_to_drop, errors='ignore')
    logger.debug(f"Shape after dropping columns: {df.shape}")
    return df

async def run_prediction(feature_df: pd.DataFrame):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Please train and register a model with name '{MODEL_NAME}' and alias '{MODEL_ALIAS}' in MLflow."
        )

    if feature_df.empty:
        raise HTTPException(status_code=422, detail="Feature extraction produced an empty DataFrame.")

    columns_to_drop = ["Activity", "timestamp"]
    feature_columns = [col for col in feature_df.columns if col not in columns_to_drop]
    X = feature_df[feature_columns]
    logger.debug(f"Running prediction on shape {X.shape}")

    # APPLY SCALER IF AVAILABLE
    if scaler:
        try:
            X_scaled = scaler.transform(X)
            logger.debug(f"Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
        except Exception as e:
            logger.warning(f"Scaling failed: {e}. Falling back to raw data (results may be poor).")
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
    else:
        logger.warning("Using raw unscaled data.")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    
    # Convert numpy types to native Python types for JSON serialization
    predicted_label = int(predictions[0])
    predicted_proba = float(probabilities[0][predicted_label])
    
    # Get activity name from loaded label encoder or fallback to hardcoded labels
    if label_encoder is not None:
        activity_name = label_encoder.classes_[predicted_label]
        activity_labels = {i: name for i, name in enumerate(label_encoder.classes_)}
    else:
        activity_labels = ACTIVITY_LABELS
        activity_name = activity_labels.get(predicted_label, f"Activity_{predicted_label}")

    stair_taxes = {
    'WALKING_UPSTAIRS': 0.65, 
    'WALKING_DOWNSTAIRS': 0.40
    }   
    activity_name_after_staircase_tax = _apply_activity_taxes(activity_name, predicted_proba, stair_taxes)

    return {
        "activity": activity_name_after_staircase_tax,
        "confidence": predicted_proba,
        "prediction_index": predicted_label,
        "all_probabilities": {
            activity_labels.get(i, f"Activity_{i}"): float(prob) 
            for i, prob in enumerate(probabilities[0])
        }
    }