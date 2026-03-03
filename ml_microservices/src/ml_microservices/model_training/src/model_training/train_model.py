# Configure MLflow with PostgreSQL + Azure Blob Storage
import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from database.database_utils import DatabaseEngine, get_postgres_db_engine

#TODO refactor into smaller parametrzed functions

logger =   logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_required_env(key: str) -> str:
    if not (value := os.getenv(key)):
        raise ValueError(f"{key} not set in environment")
    return value


def find_best_model(X_train, y_train, random_search_cv: bool=True):

    if random_search_cv:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.1, 0.3, 0.5],
            "min_child_weight": [1, 3, 5],
        }
    else:
        param_dist = {
        "n_estimators": [300],
        "max_depth": [4],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.6],
        "gamma": [0.3],
        "min_child_weight": [5],
        }
    base_model = xgb.XGBClassifier(eval_metric="mlogloss", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="accuracy",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_index = search.best_index_
    # Collect per-fold scores for the best parameter combination
    n_splits = cv.get_n_splits()
    fold_scores = np.array([search.cv_results_[f"split{i}_test_score"][best_index] for i in range(n_splits)])
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"XGBoost 5-fold CV accuracy scores: {fold_scores}")
    mean_cv = np.mean(fold_scores)
    std_cv = np.std(fold_scores)
    logger.info("Mean accuracy: {:.4f} (+/- {:.4f})".format(np.mean(fold_scores), np.std(fold_scores)))
    column_names = X_train.columns
    return search.best_estimator_, search.best_params_, mean_cv, std_cv, column_names


async def prepare_variables():
    load_dotenv()  # Load environment variables from .env file
    
    logger.info(f"ENV_PATH: {os.getenv('ENV_PATH') or '(not set)'}")
    backend_store_uri = get_required_env('MLFLOW_BACKEND_STORE_URI')
    artifact_uri = get_required_env('MLFLOW_ARTIFACT_URI')

    logger.info(f"Backend Store: {backend_store_uri.split('@')[0]}...")  # Don't print password
    logger.info(f"Artifact Store: {artifact_uri}")
    
    return backend_store_uri, artifact_uri
    
async def setup_and_test_mlflow_connection(backend_store_uri: str):
    mlflow.set_tracking_uri(backend_store_uri)
    logger.info(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")
    # Test connection
    try:
        mlflow.search_experiments()
        logger.info("✓ Successfully connected to MLflow backend")
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        raise ValueError(f"✗ Connection failed: {e}")
        
async def load_data_fromdb(database_engine) -> pd.DataFrame:
    if not (training_table_name := os.getenv("LABELED_TRAINING_DATA_TABLE_NAME")):
        raise ValueError("Training Table name not set in the variable LABELED_TRAINING_DATA_TABLE_NAME")
    data = database_engine.get_records(table_name=training_table_name)
    if data.empty:
        raise ValueError(f"No data returned from table '{training_table_name}' — cannot train on an empty dataset")
    logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns from table '{training_table_name}'")
    
    return data

async def prepare_data_for_training(data: pd.DataFrame):
    if data.empty:
        raise ValueError("Cannot prepare training data: input DataFrame is empty")
    if "Activity" not in data.columns:
        raise ValueError(f"'Activity' column not found. Available columns: {data.columns.tolist()}")

    logger.info(f"Feature count: {data.shape[1] - 1}")
    logger.info(f"Classes: {data["Activity"].unique()}")
    logger.info(f"Class distribution:\n{data["Activity"].value_counts()}")
    
    # Remove problematic columns
    remove_problematic_columns = await drop_columns_with_too_much_importance(data)

    # Balance classes to avoid class inbalance lol
    balanced_data = await balance_classes(data=remove_problematic_columns,method='undersample_highest')
    # Scale user data to match Kaggle range [-1, 1]
    logger.info("Scaling user data to [-1, 1] range to match Kaggle distribution...")
    # # Create a temporary directory for artifacts
    # temp_dir = tempfile.mkdtemp()
    # scaler_path = os.path.join(temp_dir, "scaler.pkl")
    try:
        numeric_cols = balanced_data.select_dtypes(include=['number']).columns
        cols_to_scale = [c for c in numeric_cols if c not in ['timestamp', 'Activity', 'label', 'subject', 'Subject']]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        balanced_data[cols_to_scale] = scaler.fit_transform(balanced_data[cols_to_scale])

    except Exception as e:
        logger.error(f"Scaling failed: {e}")
        raise e

    X = balanced_data.drop(["Activity", "timestamp"], axis=1)
    y = balanced_data["Activity"]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    #Encoding Labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder, scaler

async def save_ml_artifacts_to_file(**artifacts):
    """
    Saves arbitrary ML objects to a specified directory.
    Example: save_ml_artifacts(scaler=my_scaler, encoder=my_le)
    """
    temp_dir = Path.cwd() / "temp_artifact_store"
    temp_dir.mkdir(exist_ok=True)

    logger.info(f"Saving to {temp_dir}")

    saved_files = []
    for name, obj in artifacts.items():
        # Construct filename: e.g., "scaler.joblib"
        file_path = temp_dir / f"{name}.pkl"
        try:
            joblib.dump(obj, file_path)
            logger.info(f"Successfully saved artifact: {file_path}")
            saved_files.append(file_path)
        except Exception as e:
            logger.error(f"Failed to save {name}: {e}")
            raise

    return saved_files, temp_dir


async def run_ml_flow_experiment(artifact_uri: str,
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    training_labels_df: pd.DataFrame,
    test_labels_df: pd.DataFrame,
    scaler = None,
    label_encoder= None):
    
    # Train XGBoost and register with MLflow
    logger.info(f"Using artifact URI: {artifact_uri}")
    EXPERIMENT_NAME = "har-xgboost"
    try:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_uri
        )
        experiment = mlflow.get_experiment(experiment_id)
        logger.info(f"Created new experiment '{EXPERIMENT_NAME}' with artifact location: {artifact_uri}")
    except mlflow.exceptions.MlflowException:
        try:
            # If experiment already exists, just set it
            experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        except mlflow.exceptions.MlflowException as e:
            message = f"Failed to create or set MLflow experiment '{EXPERIMENT_NAME}': {e}"
            logger.error(message)
            raise ValueError(message) from e
    logger.info(f"Experiment artifact location: {experiment.artifact_location}")

    if not experiment.artifact_location.startswith(('wasbs://', 'wasb://', 's3://', 'gs://')):
        message = "Unsupported artifact location"
        logger.error(message)
        raise ValueError(message)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_NAME}-{timestamp}"
    with mlflow.start_run(run_name=run_name) as run:

        ml_artifacts_file_paths, artifacts_directory = await save_ml_artifacts_to_file(
            scaler=scaler,
            label_encoder=label_encoder
        )
        for artifact_file_path in ml_artifacts_file_paths:
            mlflow.log_artifact(artifact_file_path)
        shutil.rmtree(artifacts_directory)
        logger.info("Folder and all contents deleted.")
        # Train model with hyperparameters tuning
        model, best_params, mean_cv, std_cv, column_names = find_best_model(training_df, training_labels_df, False)
        # Evaluate
        y_pred_encoded = model.predict(test_df)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_test = label_encoder.inverse_transform(test_labels_df)

        accuracy = model.score(test_df, test_labels_df)        
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        data_with_labels = training_df.copy()
        data_with_labels["Activity"] = label_encoder.inverse_transform(training_labels_df)
        labels_value_counts = data_with_labels["Activity"].value_counts()
        
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({'Feature': column_names, 'Importance': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(20)
        logger.info(f"Feature importances: {importances}")
        fig_feature_imp = plt.figure(figsize=(10, 10))
        plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='skyblue')
        plt.xlabel('Importance Score')
        plt.title('Top 20 XGBoost Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        mlflow.log_figure(fig_feature_imp, "feature_importance.png")
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_features", training_df.shape[1])
        mlflow.log_param("n_classes", len(label_encoder.classes_))
        mlflow.log_param("confusion matrix", conf_matrix.tolist())
        mlflow.log_param("best parameters", best_params)
        mlflow.log_param("value_counts", labels_value_counts)
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
        mlflow.log_param("cv_accuracy_mean", mean_cv)
        mlflow.log_param("cv_accuracy_std", std_cv)
        
        # Log model with signature (will be saved to Azure Blob)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(training_df, model.predict(training_df))
        
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=EXPERIMENT_NAME
        )
        
        logger.info(f"\n=== XGBoost Model Training Complete ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Model registered as: {EXPERIMENT_NAME}")
        logger.info(f"Artifacts stored at: {run.info.artifact_uri}")

async def load_kaggle_data_fromdb(database_engine: DatabaseEngine, columns: list) -> pd.DataFrame:
    data = database_engine.get_records(table_name="kaggle_train_data", columns=columns)
    logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns from 'kaggle_train_data'")
    
    return data

async def balance_classes(data: pd.DataFrame, method: str = 'least_represented') -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot balance classes: input DataFrame is empty")
    if "Activity" not in data.columns:
        raise ValueError(f"'Activity' column not found — cannot balance classes. Available: {data.columns.tolist()}")
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
    elif method == 'undersample_highest':
        data_without_highest = data[data["Activity"] != majority_class]
        dwh_class_counts = data_without_highest["Activity"].value_counts()
        new_majority_class = dwh_class_counts.idxmax()
        majority = data[data["Activity"] == majority_class]
        df_majority_capped = majority.sample(n=dwh_class_counts[new_majority_class], random_state=42)
        balanced_data = pd.concat([data_without_highest, df_majority_capped], ignore_index=True)
    else:
        logger.error(f"Unknown balancing method: {method}. No balancing applied.")
        balanced_data = data
    return balanced_data



async def drop_columns_with_too_much_importance(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Cannot drop columns: input DataFrame is empty")
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
    X_train, X_test, y_train_encoded, y_test_encoded, label_encoder, scaler = await prepare_data_for_training(data)
    try:
        await run_ml_flow_experiment(artifact_uri=artifact_uri,
         training_df=X_train,
         test_df=X_test,
         training_labels_df=y_train_encoded,
         test_labels_df=y_test_encoded,
         scaler=scaler,
         label_encoder=label_encoder)
    finally:
        # Cleanup temp dir
        logger.info("Finished MLflow experiment run")






