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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import tempfile
import shutil
import os
import mlflow.xgboost
import logging
from infra.db.database_utils import DatabaseEngine, get_postgres_db_engine
import numpy as np

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

async def run_ml_flow_experiment(artifact_uri: str, data: pd.DataFrame, scaler_path: str = None):
    
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
        # Log scaler if provided
        if scaler_path and os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path)

        # Save and log label encoder
        label_encoder_path = os.path.join(os.path.dirname(scaler_path), "label_encoder.pkl")
        joblib.dump(label_encoder, label_encoder_path)
        mlflow.log_artifact(label_encoder_path)
        logger.info(f"Label encoder saved and logged to MLflow")

        # Train model
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3
        )
        model.fit(X_train, y_train_encoded)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        xgb_cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=cv, scoring='accuracy')
        logger.info("XGBoost 5-fold CV accuracy scores:", xgb_cv_scores)
        logger.info("Mean accuracy: {:.4f} (+/- {:.4f})".format(np.mean(xgb_cv_scores), np.std(xgb_cv_scores)))
        # Evaluate
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        y_test = label_encoder.inverse_transform(y_test_encoded)
        
        accuracy = model.score(X_test, y_test_encoded)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        classification_rep = classification_report(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_classes", len(label_encoder.classes_))
        mlflow.log_param("confusion matrix", conf_matrix.tolist())
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=label_encoder.classes_,
            columns=label_encoder.classes_
        )
        conf_matrix_path = "confusion_matrix.csv"
        conf_matrix_df.to_csv(conf_matrix_path)
        mlflow.log_artifact(conf_matrix_path)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=label_encoder.classes_
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation="vertical")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        mlflow.log_param("classification report", classification_rep)
        mlflow.log_param("cv_accuracy_mean", np.mean(xgb_cv_scores))
        mlflow.log_param("cv_accuracy_std", np.std(xgb_cv_scores))
        
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

async def balance_classes(data: pd.DataFrame, method: str = 'least_represented') -> pd.DataFrame:
    logger.info("Balancing classes by undersampling the majority class...")
    class_counts = data["Activity"].value_counts()
    logger.info(f"Original class distribution:\n{class_counts}")
    
    # Undersample the majority class (e.g., 'STANDING')
    majority_class = class_counts.idxmax()
    minority_classes = class_counts[class_counts.index != majority_class].index
    
    if method == 'least_represented':
        # get the least represented class
        least_represented_class = class_counts.idxmin()
        logger.info(f"Majority class: {majority_class} with {class_counts[majority_class]} samples")
        logger.info(f"Minority classes: {minority_classes.tolist()} with counts:\n{class_counts[minority_classes]}")
        
        # cut all classess to the least represented class count
        target_count = class_counts[least_represented_class]
        
        # display the target count for each class
        logger.info(f"Target count for each class after balancing: {target_count}")
        
        # undersample all classess to the target count
        balanced_data = data.groupby("Activity").apply(lambda x: x.sample(target_count, random_state=42)).reset_index(drop=True)
    elif method == 'cap_at_median':
        median_count = int(class_counts.median())
        logger.info(f"Majority class: {majority_class} with {class_counts[majority_class]} samples")
        logger.info(f"Minority classes: {minority_classes.tolist()} with counts:\n{class_counts[minority_classes]}")
        logger.info(f"Target count for each class after balancing: {median_count}")
        
        classes_above_median = class_counts[class_counts > median_count].index
        classes_below_median = class_counts[class_counts <= median_count].index
        
        logger.info(f"Classes above median: {classes_above_median.tolist()} with counts:\n{class_counts[classes_above_median]}")
        logger.info(f"Classes at or below median: {classes_below_median.tolist()}")
                    
        above_median_sampled = data[data["Activity"].isin(classes_above_median)].groupby("Activity").apply(lambda x: x.sample(median_count, random_state=42)).reset_index(drop=True)
        
        below_median_unsampled = data[data["Activity"].isin(classes_below_median)]
        
        combined_data = pd.concat([above_median_sampled, below_median_unsampled], ignore_index=True)
        balanced_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        logger.error(f"Unknown balancing method: {method}. No balancing applied.")
        balanced_data = data
    return balanced_data



async def drop_columns_with_too_much_importance(df: pd.DataFrame):
    to_drop_patterns = ['angle', 'tGravityAcc-X', 'tGravityAcc-Y', 'tGravityAcc-Z', 
                    'tBodyAcc-X', 'tBodyAcc-Y', 'tBodyAcc-Z',
                    'fBodyAcc-X', 'fBodyAcc-Y', 'fBodyAcc-Z']
    
    cols_to_drop = [col for col in df.columns if any(pat in col for pat in to_drop_patterns)]
    logger.info(f"Found {len(cols_to_drop)} columns to drop: {cols_to_drop}")
    logger.info(f"Original shape: {df.shape}")
    
    # Actually drop the columns (assign result back to df)
    df = df.drop(columns=cols_to_drop)
    logger.info(f"New shape after dropping: {df.shape}")
    
    return df
    
    
async def train_model_async(db_engine: DatabaseEngine):
    backend_store_uri, artifact_uri = await prepare_variables()
    await setup_and_test_mlflow_connection(backend_store_uri)
    data = await load_data_fromdb(db_engine)
    remove_problematic_columns = await drop_columns_with_too_much_importance(data)
    balanced_data = await balance_classes(data=remove_problematic_columns,method='cap_at_median')
    # Scale user data to match Kaggle range [-1, 1]
    logger.info("Scaling user data to [-1, 1] range to match Kaggle distribution...")
    
    # Create a temporary directory for artifacts
    temp_dir = tempfile.mkdtemp()
    scaler_path = os.path.join(temp_dir, "scaler.pkl")
    
    try:
        numeric_cols = balanced_data.select_dtypes(include=['number']).columns
        # cols_to_exclude = ['timestamp', 'Activity', 'label', 'subject']
        # flexible exclusion
        cols_to_scale = [c for c in numeric_cols if c not in ['timestamp', 'Activity', 'label', 'subject', 'Subject']]
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        balanced_data[cols_to_scale] = scaler.fit_transform(balanced_data[cols_to_scale])
        
        # Save scaler to temp file
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler temporarily saved to {scaler_path}")
    except Exception as e:
        logger.error(f"Scaling failed: {e}")
        shutil.rmtree(temp_dir)
        raise e

    columns = list(balanced_data.columns.drop(["timestamp"]))
    # kaggle_data = await load_kaggle_data_fromdb(db_engine, columns)
    
    combined_data = balanced_data  # pd.concat([data, kaggle_data], ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")
    
    try:
        await run_ml_flow_experiment(artifact_uri, combined_data, scaler_path=scaler_path)
    finally:
        # Cleanup temp dir
        shutil.rmtree(temp_dir)






