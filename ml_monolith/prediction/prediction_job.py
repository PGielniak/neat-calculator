from data_pipeline.database import save_to_db, initialize_tables, update_pipeline_run_status
from infra.db.database_utils import DatabaseFactory
import os

import logging
from pathlib import Path
logger = logging.getLogger(__name__)
import mlflow
from load_model import load_production_model

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# e.g. "file:/path/to/mlruns" or "sqlite:///mlflow.db"
MODEL_NAME = os.getenv("MODEL_NAME", "HAR_xgboost")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production") 



database_engine = DatabaseFactory.create_engine(
    db_type='sqlite',
    db_path='sensor_features.db'
)

data_to_predict = database_engine.get_records(table_name="predictions", filters={"label": "UNLABELED"})

# load model
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = load_production_model(MODEL_NAME, MODEL_STAGE)

data_to_predict = data_to_predict.drop(columns=["label", "id"])  # drop non-feature columns
X = data_to_predict.to_numpy()
predictions = model.predict(X)  # class labels
probabilities = model.predict(X, params={"predict_method": "predict_proba"}) 




# run prediction on the whole dataset and apply labels


