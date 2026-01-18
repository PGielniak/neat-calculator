# Configure MLflow with PostgreSQL + Azure Blob Storage
import os
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

from infra.db.database_utils import DatabaseFactory

#TODO refactor into smaller parametrzed functions

# Load data from database (same source as prediction pipeline)

# Load environment variables
load_dotenv()

# Add parent directory to path
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


database_engine = DatabaseFactory.create_engine(
    db_type='sqlite',
    db_path=str(project_root / 'sensor_features.db')
)


# Get URIs from environment
backend_store_uri = os.getenv('MLFLOW_BACKEND_STORE_URI', 'sqlite:///mlflow.db')  # Fallback to local
artifact_uri = os.getenv('MLFLOW_ARTIFACT_URI', './mlruns')  # Fallback to local

print(f"Backend Store: {backend_store_uri.split('@')[0]}...")  # Don't print password
print(f"Artifact Store: {artifact_uri}")

# Set tracking URI
mlflow.set_tracking_uri(backend_store_uri)

# Test connection
try:
    mlflow.search_experiments()
    print("✓ Successfully connected to MLflow backend")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("Falling back to local SQLite")
    
    
data = database_engine.get_records(table_name="training_data_labeled")
print(f"Loaded {len(data)} rows from database")
print(f"Total columns: {len(data.columns)}")

# Drop non-feature columns
X = data.drop(["Activity", "timestamp"], axis=1)
y = data["Activity"]

print(f"\nFeature count: {X.shape[1]}")
print(f"Classes: {y.unique()}")
print(f"Class distribution:\n{y.value_counts()}")

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


# Train XGBoost and register with MLflow
artifact_uri = os.getenv('MLFLOW_ARTIFACT_URI', './mlruns')
print(f"Using artifact URI: {artifact_uri}")

# Set experiment with artifact location
experiment = mlflow.set_experiment("Merito_HAR_Production_Models")
print(f"Experiment artifact location: {experiment.artifact_location}")

# If experiment artifact location is local, we need to create a new experiment or update it
if not experiment.artifact_location.startswith(('wasbs://', 'wasb://', 's3://', 'gs://')):
    print(f"⚠️  Experiment is using local storage: {experiment.artifact_location}")
    print("Creating a new experiment with Azure Blob Storage...")
    
    # Create a new experiment with the correct artifact location
    try:
        experiment_id = mlflow.create_experiment(
            "HAR_Production_Models3",
            artifact_location=artifact_uri
        )
        mlflow.set_experiment("HAR_Production_Models3")
        print(f"✓ Created new experiment with artifact location: {artifact_uri}")
    except:
        # If experiment already exists, just set it
        mlflow.set_experiment("HAR_Production_Models3")

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

