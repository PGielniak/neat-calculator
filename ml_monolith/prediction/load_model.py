import mlflow
from mlflow import MlflowClient
import joblib
import os

def load_production_model(model_name, model_alias):
    """
    Load the registered model by alias (e.g., 'production').
    Returns tuple: (model, scaler)
    """
    # Use @ syntax for aliases (newer MLflow) instead of / for stages
    model_uri = f"models:/{model_name}@{model_alias}"
    print(f"Loading model from MLflow URI: {model_uri}")
    
    # Load model
    model = mlflow.xgboost.load_model(model_uri)
    
    # Load associated scaler artifact
    scaler = None
    try:
        client = MlflowClient()
        # Get run ID associated with this model version
        mv = client.get_model_version_by_alias(model_name, model_alias)
        run_id = mv.run_id
        
        print(f"Downloading scaler artifact from Run ID: {run_id}")
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
        
        if os.path.exists(local_path):
            scaler = joblib.load(local_path)
            print("✓ Scaler loaded successfully")
        else:
            print("⚠ Scaler artifact not found")
            
    except Exception as e:
        print(f"⚠ Failed to load scaler: {e}")
        
    return model, scaler