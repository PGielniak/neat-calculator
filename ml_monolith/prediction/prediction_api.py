from json import load
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, ValidationError
from typing import List
import pandas as pd
import mlflow
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
project_root = Path().resolve().parent  # from neat_dashboard/ to repo root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

ml_monolith_root = project_root / "ml_monolith"
if str(ml_monolith_root) not in sys.path:
    sys.path.insert(0, str(ml_monolith_root))

from ml_monolith.data_pipeline.process_raw_data import remove_duplicates, resample_data, create_sliding_windows, extract_features_from_windows, rename_features, filter_features_to_match_kaggle
load_dotenv()
from load_model import load_production_model

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_BACKEND_STORE_URI", None)
if MLFLOW_TRACKING_URI is None:
    raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")
MODEL_NAME = os.getenv("MODEL_NAME", "HAR_xgboost")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
model, scaler = load_production_model(MODEL_NAME, MODEL_ALIAS)
print(f"Loaded model: {MODEL_NAME} with alias: {MODEL_ALIAS}")

try:
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    
    print(f"\n=== MODEL DETAILS ===")
    print(f"Model Name: {model_version.name}")
    print(f"Version: {model_version.version}")
    print(f"Stage: {model_version.current_stage}")
    print(f"Description: {model_version.description or 'No description available'}")
    print(f"Created: {model_version.creation_timestamp}")
    print(f"Last Updated: {model_version.last_updated_timestamp}")
    print(f"Run ID: {model_version.run_id}")
    
    # Get run information for additional details
    run = client.get_run(model_version.run_id)
    print(f"\n=== TRAINING RUN DETAILS ===")
    print(f"Run Name: {run.info.run_name}")
    print(f"Start Time: {run.info.start_time}")
    print(f"End Time: {run.info.end_time}")
    print(f"Status: {run.info.status}")
    
    # Show key metrics if available
    if run.data.metrics:
        print(f"\n=== TRAINING METRICS ===")
        for metric_name, metric_value in run.data.metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Show parameters
    if run.data.params:
        print(f"\n=== MODEL PARAMETERS ===")
        for param_name, param_value in run.data.params.items():
            print(f"  {param_name}: {param_value}")

except Exception as e:
    print(f"Could not fetch detailed model info: {e}")

if scaler:
    print("\n✅ Scaler is available for data transformation.")
else:
    print("\n⚠️  WARNING: No scaler loaded. Predictions on raw data may be incorrect.")


app = FastAPI()

WINDOW_SIZE = 128

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
            raise ValidationError(f"Not enough data! Expected at least {WINDOW_SIZE}, got {len(v)}")
        return v

@app.post("/predict")
async def predict(payload: PredictionRequest):
    df = pd.DataFrame(payload.samples)
    
    if len(df) > WINDOW_SIZE:
        df = df.iloc[-WINDOW_SIZE:].reset_index(drop=True)
    
    if len(df) < WINDOW_SIZE:
        return {"activity": "BUFFERING", "confidence": 0.0}

    feature_df = await data_processing_pipeline(df)
    
    return {"activity": "WALKING", "confidence": 0.95}


async def data_processing_pipeline(window_df: pd.DataFrame):
    deduped_data = remove_duplicates(window_df)
    resampled_data = resample_data(deduped_data, target_freq_hz=50)
    del deduped_data
    sliding_windows, feature_array, labels_array, timestamp_array = create_sliding_windows(resampled_data, window_size=128, step_size=64, sensor_cols=sensor_cols)
    del resampled_data
    extracted_features = extract_features_from_windows(sliding_windows, feature_array, labels_array, timestamp_array)
    del sliding_windows, feature_array, labels_array  # free up memory
    renamed_features = rename_features(extracted_features)
    del extracted_features  # free up memory
    intersect_with_kaggle = filter_features_to_match_kaggle(renamed_features, kaggle_csv_path=kaggle_csv_path)   
    del renamed_features  # free up memory
    feature_df = pd.DataFrame(intersect_with_kaggle)
    feature_df_without_columns = drop_unnecessary_columns(feature_df, columns_to_drop=['angle', 'tGravityAcc-X', 'tGravityAcc-Y', 'tGravityAcc-Z', 
                        'tBodyAcc-X', 'tBodyAcc-Y', 'tBodyAcc-Z',
                        'fBodyAcc-X', 'fBodyAcc-Y', 'fBodyAcc-Z'])
    return feature_df_without_columns

def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    return df.drop(columns=columns_to_drop, errors='ignore')

async def run_prediction(feature_df: pd.DataFrame):
    original_df = feature_df.copy()
    columns_to_drop = ["Activity", "timestamp"]
    feature_columns = [col for col in feature_df.columns if col not in columns_to_drop]
    for col in feature_columns:
        print(f"Feature column: {col}")
        
        
    X= feature_df[feature_columns]
    
    # APPLY SCALER IF AVAILABLE
    if scaler:
        print("Applying scaler to input data...")
        # Filter columns that scaler expects
        # (Assuming scaler was fit on same feature set. If mismatch, we might need intersection)
        try:
            X_scaled = scaler.transform(X)
            print(f"Scaled Data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
            # Make predictions using the SCALED data
            predictions = model.predict(X_scaled)          # Returns integer labels
            probabilities = model.predict_proba(X_scaled)  # Probability matrix
        except Exception as e:
            print(f"Scaling failed: {e}")
            print("Falling back to raw data (results may be poor)")
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
    else:
        print("WARNING: Using raw unscaled data.")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
    return zip(predictions, probabilities)