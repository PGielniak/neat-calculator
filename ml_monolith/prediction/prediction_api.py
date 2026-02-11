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
from data_pipeline.process_raw_data import remove_duplicates, resample_data, create_sliding_windows, extract_features_from_windows, rename_features, filter_features_to_match_kaggle
load_dotenv()
from prediction.load_model import load_production_model
import joblib
import tempfile

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_BACKEND_STORE_URI", None)
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
if MLFLOW_TRACKING_URI is None:
    raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

# Set MLflow tracking URI before loading model
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"✓ MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

MODEL_NAME = os.getenv("MODEL_NAME", "HAR_xgboost")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

model = None
scaler = None
label_encoder = None
model_info = None

try:
    model, scaler = load_production_model(MODEL_NAME, MODEL_ALIAS)
    print(f"✓ Loaded model: {MODEL_NAME} with alias: {MODEL_ALIAS}")
    
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    
    # Download and load label encoder from model artifacts
    try:
        artifact_path = client.download_artifacts(model_version.run_id, "label_encoder.pkl")
        label_encoder = joblib.load(os.path.join(artifact_path, "label_encoder.pkl"))
        print(f"✓ Loaded label encoder with classes: {label_encoder.classes_.tolist()}")
    except Exception as e:
        print(f"⚠️  Could not load label encoder from artifacts: {e}")
        print("   Will fall back to default activity labels")
    
    model_info = {
        "name": model_version.name,
        "version": model_version.version,
        "stage": model_version.current_stage,
        "run_id": model_version.run_id,
        "description": model_version.description or "No description available",
        "created": model_version.creation_timestamp,
        "last_updated": model_version.last_updated_timestamp
    }
    
    print(f"\n=== MODEL DETAILS ===")
    print(f"Model Name: {model_info['name']}")
    print(f"Version: {model_info['version']}")
    print(f"Stage: {model_info['stage']}")
    print(f"Description: {model_info['description']}")
    print(f"Created: {model_info['created']}")
    print(f"Last Updated: {model_info['last_updated']}")
    print(f"Run ID: {model_info['run_id']}")
    
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

except mlflow.exceptions.MlflowException as e:
    if "not found" in str(e).lower():
        print(f"\n⚠️  WARNING: Model '{MODEL_NAME}' with alias '{MODEL_ALIAS}' not found in MLflow.")
        print(f"   The API will start but predictions will fail until a model is registered.")
        print(f"   Please train and register a model with alias '{MODEL_ALIAS}'.")
    else:
        print(f"⚠️  MLflow error while loading model: {e}")
except Exception as e:
    print(f"⚠️  Could not load model: {e}")

if model:
    print("\n✅ Model loaded successfully and ready for predictions.")
else:
    print("\n⚠️  No model loaded. Predictions will fail until a model is available.")
    
if scaler:
    print("\n✅ Scaler is available for data transformation.")
else:
    print("\n⚠️  WARNING: No scaler loaded. Predictions on raw data may be incorrect.")

if label_encoder is not None:
    print(f"\n✅ Label encoder loaded with {len(label_encoder.classes_)} activity classes: {list(label_encoder.classes_)}")
else:
    print("\n⚠️  Label encoder not loaded. Will use fallback activity labels.")


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
            raise ValidationError(f"Not enough data! Expected at least {WINDOW_SIZE}, got {len(v)}")
        return v

@app.post("/predict")
async def predict(payload: PredictionRequest):
    # Convert Pydantic models to dictionaries before creating DataFrame
    samples_dict = [sample.model_dump() for sample in payload.samples]
    df = pd.DataFrame(samples_dict)
    
    print(f"Received DataFrame with columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    
    if len(df) > WINDOW_SIZE:
        df = df.iloc[-WINDOW_SIZE:].reset_index(drop=True)
    
    if len(df) < WINDOW_SIZE:
        return {"activity": "BUFFERING", "confidence": 0.0}

    feature_df = await data_processing_pipeline(df)
    
    predictions = await run_prediction(feature_df)
    
    print(f"Predictions: {predictions}")
    
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
    kaggle_csv_path = "/app/kaggle.csv"
    intersect_with_kaggle = filter_features_to_match_kaggle(renamed_features, kaggle_csv_path=kaggle_csv_path)   
    del renamed_features  # free up memory
    feature_df = pd.DataFrame(intersect_with_kaggle)
    feature_df_without_columns = drop_unnecessary_columns(feature_df, columns_to_drop=['angle', 'tGravityAcc-X', 'tGravityAcc-Y', 'tGravityAcc-Z', 
                        'tBodyAcc-X', 'tBodyAcc-Y', 'tBodyAcc-Z',
                        'fBodyAcc-X', 'fBodyAcc-Y', 'fBodyAcc-Z'])
    return feature_df_without_columns

def drop_unnecessary_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    to_drop_patterns = ['angle', 'tGravityAcc-X', 'tGravityAcc-Y', 'tGravityAcc-Z', 
                    'tBodyAcc-X', 'tBodyAcc-Y', 'tBodyAcc-Z',
                    'fBodyAcc-X', 'fBodyAcc-Y', 'fBodyAcc-Z']
    
    cols_to_drop = [col for col in df.columns if any(pat in col for pat in to_drop_patterns)]
    print(f"Found {len(cols_to_drop)} columns to drop: {cols_to_drop}")
    print(f"Original shape: {df.shape}")
    
    # Actually drop the columns (assign result back to df)
    df = df.drop(columns=cols_to_drop)
    print(f"New shape after dropping columns: {df.shape}")
    return df

async def run_prediction(feature_df: pd.DataFrame):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Please train and register a model with name '{MODEL_NAME}' and alias '{MODEL_ALIAS}' in MLflow."
        )
    
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
    
    return {
        "activity": activity_name,
        "confidence": predicted_proba,
        "prediction_index": predicted_label,
        "all_probabilities": {
            activity_labels.get(i, f"Activity_{i}"): float(prob) 
            for i, prob in enumerate(probabilities[0])
        }
    }