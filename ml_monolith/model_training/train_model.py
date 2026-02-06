# Configure MLflow with PostgreSQL + Azure Blob Storage
from multiprocessing.spawn import prepare
import os
from venv import logger
from dotenv import load_dotenv
import mlflow
import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.xgboost
import logging
from infra.db.database_utils import DatabaseEngine, get_postgres_db_engine

#TODO refactor into smaller parametrzed functions

logger =   logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def prepare_variables():
    load_dotenv('../.env')  # Load environment variables from .env file
    
    logger.info(os.getenv("ENV_PATH"))
    backend_store_uri = os.getenv('MLFLOW_BACKEND_STORE_URI', 'sqlite:///mlflow.db')  # Fallback to local
    artifact_uri = os.getenv('MLFLOW_ARTIFACT_URI', './mlruns')  # Fallback to local

    logger.info(f"Backend Store: {backend_store_uri.split('@')[0]}...")  # Don't print password
    logger.info(f"Artifact Store: {artifact_uri}")
    
    return backend_store_uri, artifact_uri
    
async def setup_and_test_mlflow_connection(backend_store_uri: str):
    # Set tracking URI
    mlflow.set_tracking_uri(backend_store_uri)
    logger.info(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")
    # Test connection
    try:
        mlflow.search_experiments()
        logger.info("✓ Successfully connected to MLflow backend")
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        logger.info("Falling back to local SQLite")
        
async def load_data_fromdb(database_engine) -> pd.DataFrame:
    data = database_engine.get_records(table_name="training_data_labeled")
    logger.info(f"Loaded {len(data)} rows from database")
    logger.info(f"Total columns: {len(data.columns)}")
    
    return data

async def prepare_data_for_training(X: pd.DataFrame, y: pd.Series):
    logger.info(f"\nFeature count: {X.shape[1]}")
    logger.info(f"Classes: {y.unique()}")
    logger.info(f"Class distribution:\n{y.value_counts()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder

async def run_ml_flow_experiment(artifact_uri: str, data: pd.DataFrame):
    
    # Train XGBoost and register with MLflow
    print(f"Using artifact URI: {artifact_uri}")
    X = data.drop(["Activity", "timestamp"], axis=1)
    y = data["Activity"]

    X_train, X_test, y_train_encoded, y_test_encoded, label_encoder = await prepare_data_for_training(X, y)
    # Set experiment with artifact location
    experiment = mlflow.set_experiment("Merito_HAR_Production_Models")
    logger.info(f"Experiment artifact location: {experiment.artifact_location}")

    # If experiment artifact location is local, we need to create a new experiment or update it
    if not experiment.artifact_location.startswith(('wasbs://', 'wasb://', 's3://', 'gs://')):
        logger.warning(f"⚠️  Experiment is using local storage: {experiment.artifact_location}")
        logger.info("Creating a new experiment with Azure Blob Storage...")
        
        # Create a new experiment with the correct artifact location
        try:
            experiment_id = mlflow.create_experiment(
                "HAR_Production_Models",
                artifact_location=artifact_uri
            )
            mlflow.set_experiment("HAR_Production_Models")
            print(f"✓ Created new experiment with artifact location: {artifact_uri}")
        except:
            # If experiment already exists, just set it
            mlflow.set_experiment("HAR_Production_Models")
    with mlflow.start_run(run_name="XGBoost_180_features") as run:
        # Train model
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3
        )
        model.fit(X_train, y_train_encoded)
        
        # Evaluate
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        y_test = label_encoder.inverse_transform(y_test_encoded)
        
        accuracy = model.score(X_test, y_test_encoded)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(label_encoder.classes_))
        
        # Log model with signature (will be saved to Azure Blob)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="HAR_xgboost"
        )
        
        print(f"\n=== XGBoost Model Training Complete ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Model registered as: HAR_xgboost")
        print(f"Artifacts stored at: {run.info.artifact_uri}")

async def load_kaggle_data_fromdb(database_engine: DatabaseEngine, columns: list) -> pd.DataFrame:
    data = database_engine.get_records(table_name="kaggle_train_data", columns=columns)
    print(f"Loaded {len(data)} rows from database")
    print(f"Total columns: {len(data.columns)}")
    
    return data

async def train_model_async(db_engine: DatabaseEngine):
    backend_store_uri, artifact_uri = await prepare_variables()
    await setup_and_test_mlflow_connection(backend_store_uri)
    data = await load_data_fromdb(db_engine)
    
    columns = list(data.columns.drop(["timestamp"]))
    kaggle_data = await load_kaggle_data_fromdb(db_engine, columns)
    
    combined_data = pd.concat([data, kaggle_data], ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")
    await run_ml_flow_experiment(artifact_uri, combined_data)






