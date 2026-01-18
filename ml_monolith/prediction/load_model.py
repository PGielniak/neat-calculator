import mlflow

def load_production_model(model_name, model_alias):
    """
    Load the registered model by alias (e.g., 'production').
    """
    # Use @ syntax for aliases (newer MLflow) instead of / for stages
    model_uri = f"models:/{model_name}@{model_alias}"
    print(f"Loading model from MLflow URI: {model_uri}")
    # Load with xgboost flavor to get predict_proba support
    model = mlflow.xgboost.load_model(model_uri)
    return model