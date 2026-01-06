import logging
import uuid
import os
import pandas as pd
import argparse
import hashlib
from data_pipeline.models import PipelineRun, ProcessedFile
from data_pipeline.database import save_to_db, initialize_tables, update_pipeline_run_status
from infra.db.database_utils import DatabaseFactory
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from data_pipeline.process_raw_data import process_raw_sensor_data

argparser = argparse.ArgumentParser(description="Process raw sensor data files.")
argparser.add_argument("--raw_data_file_dir", type=str, required=True, help="Directory containing raw sensor data files.")
argparser.add_argument("--labels_csv_path", type=str, required=True, help="Path to the labels CSV file.")
argparser.add_argument("--kaggle_csv_path", type=str, required=True, help="Path to the Kaggle CSV file.")
args = argparser.parse_args()

logger = logging.getLogger(__name__)

database_engine = DatabaseFactory.create_engine(
    db_type='sqlite',
    db_path='sensor_features.db'
)

logger.info("Initializing database tables if they do not exist.")
initialize_tables(database_engine)

pipeline_run = PipelineRun(
    run_id=str(uuid.uuid4()),
    status="STARTED",
    started_at=datetime.now(),
    folder_path=args.raw_data_file_dir,
    labels_csv_path=args.labels_csv_path)


logger.info(f"Pipeline Run ID: {pipeline_run.run_id}")
logger.info(f"Saving Pipeline Run info to database.")

save_to_db(pipeline_run, table_name="pipeline_runs", db_engine=database_engine)

sensor_data_files = os.listdir(args.raw_data_file_dir)
sensor_data_files.sort()
sensor_data_files = pd.DataFrame(sensor_data_files, columns=["file_name"])
sensor_data_files["pipeline_run_id"] = pipeline_run.run_id

sensor_data_files["checksum"] = sensor_data_files["file_name"].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

processed_files = []
skipped_files = []
for _, row in sensor_data_files.iterrows():
    processed_file = ProcessedFile(
        file_id=str(uuid.uuid4()),
        file_name=row["file_name"],
        pipeline_run_id=row["pipeline_run_id"],
        processed_at=datetime.now(),
        checksum=row["checksum"]
    )
    try:
        save_to_db(processed_file, table_name="processed_files", db_engine=database_engine)
        logger.info(f"Saved processed file info for {row['file_name']} to database.")
        processed_files.append(processed_file)
    except IntegrityError as e:
        if "UNIQUE constraint failed: processed_files.checksum" in str(e):
            logger.warning(f"File {row['file_name']} already processed (duplicate checksum). Skipping.")
            skipped_files.append(row['file_name'])
        else:
            logger.error(f"Integrity error saving processed file info for {row['file_name']}: {e}")
            raise
    except Exception as e:
        logger.error(f"Error saving processed file info for {row['file_name']} to database: {e}")
        raise


logger.info(f"Saving Processed Files info to database.")


sensor_data = process_raw_sensor_data(
    raw_data_file_dir=args.raw_data_file_dir,
    labels_csv_path=args.labels_csv_path,
    kaggle_csv_path=args.kaggle_csv_path,
    skipped_files=skipped_files
)

try:
    logger.info(f"Saving processed sensor data to database.")
    save_to_db(sensor_data, table_name="sensor_features", db_engine=database_engine)
    pipeline_run.status = "COMPLETED"
except Exception as e:
    logger.error(f"Error saving processed sensor data to database: {e}")
    pipeline_run.status = "FAILED"
finally:
    pipeline_run.completed_at = datetime.now()
    logger.info(f"Updating Pipeline Run status to {pipeline_run.status}.")
    update_pipeline_run_status(database_engine, run_id=pipeline_run.run_id, status=pipeline_run.status, completed_at=pipeline_run.completed_at)