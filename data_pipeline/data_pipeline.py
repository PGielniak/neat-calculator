import logging
import uuid
import os
import pandas as pd
import argparse
import hashlib
from data_pipeline.models import PipelineRun, ProcessedFile
from data_pipeline.database import save_to_db, initialize_tables, update_pipeline_run_status
from infra.db.database_utils import SQLiteEngine, DatabaseFactory
from datetime import datetime

argparser = argparse.ArgumentParser(description="Process raw sensor data files.")
argparser.add_argument("--raw_data_file_dir", type=str, required=True, help="Directory containing raw sensor data files.")
argparser.add_argument("--labels_csv_path", type=str, required=True, help="Path to the labels CSV file.")
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
for _, row in sensor_data_files.iterrows():
    processed_file = ProcessedFile(
        file_id=str(uuid.uuid4()),
        file_name=row["file_name"],
        pipeline_run_id=row["pipeline_run_id"],
        processed_at=datetime.now(),
        checksum=row["checksum"]
    )
    processed_files.append(processed_file)
logger.info(f"Saving Processed Files info to database.")

save_to_db(processed_files, table_name="processed_files", db_engine=database_engine)

pipeline_run.status = "COMPLETED"
pipeline_run.completed_at = datetime.now()
logger.info(f"Updating Pipeline Run status to COMPLETED.")
update_pipeline_run_status(database_engine, run_id=pipeline_run.run_id, status="COMPLETED", completed_at=pipeline_run.completed_at)