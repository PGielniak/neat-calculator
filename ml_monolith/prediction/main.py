import os
from typing import List

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from har_pipeline import preprocess_and_extract_features


# ---- Config ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # e.g. "file:/path/to/mlruns" or "sqlite:///mlflow.db"
MODEL_NAME = os.getenv("MODEL_NAME", "har-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")    # "Production", "Staging", or a number like "1"


if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI env var must be set")


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_production_model():
    """
    Load the current Production version of the registered model.
    """
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"Loading model from MLflow URI: {model_uri}")
    # pyfunc works with sklearn, xgboost, etc.
    model = mlflow.pyfunc.load_model(model_uri)
    return model


# ---- FastAPI setup ----

app = FastAPI(title="HAR Prediction Service")

# load once at startup
model = load_production_model()


# ---- Request schemas ----

class SensorSample(BaseModel):
    timestamp: int     # or float (ms/ns since epoch)
    accelerometerX: float
    accelerometerY: float
    accelerometerZ: float
    gyroscopeX: float
    gyroscopeY: float
    gyroscopeZ: float


class SensorBatch(BaseModel):
    device_id: str
    session_id: str
    samples: List[SensorSample]


# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status": "ok", "model_name": MODEL_NAME, "stage": MODEL_STAGE}


@app.post("/predict")
def predict(batch: SensorBatch):
    # convert samples to DataFrame
    df_raw = pd.DataFrame([s.dict() for s in batch.samples])

    # run your preprocessing + feature extraction
    X_features = preprocess_and_extract_features(df_raw)

    # IMPORTANT: ensure feature columns match the model's training schema
    # If you saved the feature list somewhere, align here:
    # X_features = X_features[feature_names]

    preds = model.predict(X_features)

    # you can also get probabilities if your model supports it:
    # probs = model.predict_proba(X_features)

    return {
        "device_id": batch.device_id,
        "session_id": batch.session_id,
        "n_windows": len(X_features),
        "predictions": [str(p) for p in preds],
    }


# Optional: endpoint to reload latest Production model without restarting app
@app.post("/reload_model")
def reload_model():
    global model
    model = load_production_model()
    return {"status": "reloaded", "model_name": MODEL_NAME, "stage": MODEL_STAGE}
