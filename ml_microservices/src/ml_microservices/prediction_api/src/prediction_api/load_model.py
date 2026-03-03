import mlflow
from mlflow import MlflowClient
import joblib
import logging
import os

logger = logging.getLogger(__name__)

def load_production_model(model_name, model_alias):
    """
    Load the registered model by alias (e.g., 'production').
    Returns tuple: (model, scaler, label_encoder)
    """
    current_uri = mlflow.get_tracking_uri()
    logger.info(f"Current MLflow tracking URI: {current_uri}")

    model_uri = f"models:/{model_name}@{model_alias}"
    logger.info(f"Loading model from MLflow URI: {model_uri}")

    model = mlflow.xgboost.load_model(model_uri)

    scaler = None
    label_encoder = None
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(model_name, model_alias)
        run_id = mv.run_id

        logger.info(f"Downloading scaler artifact from run ID: {run_id}")
        scaler_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
        if os.path.exists(scaler_local_path):
            scaler = joblib.load(scaler_local_path)
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("Scaler artifact not found")

        logger.info(f"Downloading label encoder artifact from run ID: {run_id}")
        le_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="label_encoder.pkl")
        if os.path.exists(le_local_path):
            label_encoder = joblib.load(le_local_path)
            logger.info("Label encoder loaded successfully")
        else:
            logger.warning("Label encoder artifact not found")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")

    return model, scaler, label_encoder