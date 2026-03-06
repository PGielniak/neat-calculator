from json import load
from fastapi import FastAPI, HTTPException, Header
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_NAME = os.getenv("MODEL_NAME", "har-randomforest01")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

API_KEY_SERVICE_URL = os.getenv("API_KEY_SERVICE_URL", "http://localhost:8007/api-keys/validate")

model = None
scaler = None
label_encoder = None
model_info = None

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


_init_mlflow()
_load_models()


app = FastAPI()

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
async def health():
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
async def predict(payload: PredictionRequest, x_api_key_header: Annotated[str | None, Header(alias="X-Api-Key")] = None):

    if not x_api_key_header:
        raise HTTPException(status_code=401, detail="Failed to authenticate the request.")
    
    if not validate_api_key_header(x_api_key_header):
        raise HTTPException(status_code=401, detail="Failed to authenticate the request.")
    # Convert Pydantic models to dictionaries before creating DataFrame
    
    x_api_key_prefix = x_api_key_header.split("_")[0]

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

async def validate_api_key_header(x_api_key_header: str) -> bool:
    
    body = {'raw_key': x_api_key_header}
    try:
        # Add a timeout so auth failures don't hang the request
        response = requests.post(url=API_KEY_SERVICE_URL, data=body, timeout=5)
    except requests.RequestException as exc:
        logger.error(f"API key validation request failed: {exc}")
        return False

    logger.info(f"Validation Response status code: {response.status_code}")

    # Treat non-200 responses as failed validation
    if response.status_code != 200:
        return False

    try:
        json_payload = response.json()
    except ValueError:
        logger.error("API key validation response is not valid JSON.")
        return False

    # Expecting a JSON object with a 'valid' field, e.g. {'valid': true}
    if isinstance(json_payload, dict):
        return bool(json_payload.get("valid", False))

    # Fallback: if the payload itself is a boolean
    if isinstance(json_payload, bool):
        return json_payload

    # Any other unexpected format is treated as invalid
    return False
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