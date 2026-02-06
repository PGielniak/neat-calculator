import logging
import uuid
import os
import shutil
import pandas as pd
import argparse
import hashlib
from data_pipeline.models import PipelineRun, ProcessedFile
from data_pipeline.database import save_to_db, initialize_tables, update_pipeline_run_status
from infra.db.database_utils import get_postgres_db_engine
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from data_pipeline.storage_account_helpers import download_blob_to_dir
import asyncio

from data_pipeline.process_raw_data import process_raw_sensor_data

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process raw sensor data files.")
    argparser.add_argument("--raw_data_file_dir", type=str, required=False, help="Directory containing raw sensor data files.")
    argparser.add_argument("--raw_data_storage_account_container_uri", type=str, required=False, help="Storage account container uri containing raw sensor data files. eg. wasbs://<container>@<storage_account>.blob.core.windows.net/")
    argparser.add_argument("--labels_csv_path", type=str, required=False, help="Path to the labels CSV file.")
    argparser.add_argument("--labels_storage_account_blob_uri", type=str, required=False, help="Storage account blob uri for the labels CSV file. eg. wasbs://<container>@<storage_account>.blob.core.windows.net/path/to/labels.csv")
    argparser.add_argument("--kaggle_csv_path", type=str, required=False, help="Path to the Kaggle CSV file.", default="kaggle.csv")
    argparser.add_argument("--pipeline_run_id", type=str, required=False, help="Optionally supply your own pipeline run ID.")
    args = argparser.parse_args()

    raw_data_file_dir = args.raw_data_file_dir
    labels_csv_path = args.labels_csv_path
    raw_data_storage_account_container_uri = args.raw_data_storage_account_container_uri
    labels_storage_account_blob_uri = args.labels_storage_account_blob_uri
    kaggle_csv_path = args.kaggle_csv_path
    pipeline_run_id = args.pipeline_run_id



async def run_data_pipeline_async(pipeline_run_id: str = "",
                            raw_data_file_dir: str = "",
                            raw_data_storage_account_container_uri: str = "",
                            labels_csv_path: str = "",
                            labels_storage_account_blob_uri: str = "",
                            kaggle_csv_path: str = "",
                            db_engine = None
                            
                            ):
    logger = logging.getLogger(__name__)

    if db_engine is None:
        database_engine = get_postgres_db_engine()
    else:
        database_engine = db_engine

    logger.info("Initializing database tables if they do not exist.")
    initialize_tables(database_engine)

    if not pipeline_run_id:
        pipeline_run_id = str(uuid.uuid4())
        
    pipeline_run = PipelineRun(
        run_id=pipeline_run_id,
        status="STARTED",
        started_at=datetime.now(),
        folder_path=raw_data_file_dir,
        labels_csv_path=labels_csv_path)


    logger.info(f"Pipeline Run ID: {pipeline_run.run_id}")
    logger.info(f"Saving Pipeline Run info to database.")

    save_to_db(pipeline_run, table_name="pipeline_runs", db_engine=database_engine)

    if not raw_data_file_dir and not raw_data_storage_account_container_uri:
        logger.error("Either --raw_data_file_dir or --raw_data_storage_account_container_uri must be provided.")
        raise ValueError("Either --raw_data_file_dir or --raw_data_storage_account_container_uri must be provided.")

    raw_data_dir = raw_data_file_dir if raw_data_file_dir else "raw_sensor_data_files"
    if raw_data_storage_account_container_uri:
        logger.info("Downloading raw data files from storage account container URI")
        # storage account name wasbs://<container>@<storage_account>.blob.core.windows.net/
        download_blob_to_dir(
            storage_account_blob_uri=raw_data_storage_account_container_uri,
            download_dir="raw_sensor_data_files",
            logger=logger
        )
        
    if not labels_csv_path and not labels_storage_account_blob_uri:
        logger.error("Either --labels_csv_path or --labels_storage_account_blob_uri must be provided.")
        raise ValueError("Either --labels_csv_path or --labels_storage_account_blob_uri must be provided.")

    labels_csv_path = labels_csv_path if labels_csv_path else "labels.csv"
    if labels_storage_account_blob_uri:
        logger.info("Downloading labels CSV from storage account blob URI")
        # storage account name wasbs://<container>@<storage_account>.blob.core.windows.net/path/to/labels.csv
        download_blob_to_dir(
            storage_account_blob_uri=labels_storage_account_blob_uri,
            download_dir=".",
            logger=logger
        )

    sensor_data_files = os.listdir(raw_data_dir)
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
            if "duplicate key value violates unique constraint \"processed_files_checksum_key\"" in str(e):
                logger.warning(f"File {row['file_name']} already processed (duplicate checksum). Skipping.")
                skipped_files.append(row['file_name'])
            else:
                logger.error(f"Integrity error saving processed file info for {row['file_name']}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error saving processed file info for {row['file_name']} to database: {e}")
            raise
    logger.info(f"Saving Processed Files info to database.")

    try:
        sensor_data = process_raw_sensor_data(
            raw_data_file_dir=raw_data_dir,
            labels_csv_path=labels_csv_path,
            kaggle_csv_path=kaggle_csv_path,
            skipped_files=skipped_files
        )
    except Exception as e:
        logger.error(f"Error processing raw sensor data: {e}")
        pipeline_run.status = "FAILED"
        pipeline_run.completed_at = datetime.now()
        update_pipeline_run_status(database_engine, run_id=pipeline_run.run_id, status=pipeline_run.status, completed_at=pipeline_run.completed_at)
        raise e

    if raw_data_storage_account_container_uri:
        shutil.rmtree(raw_data_dir, ignore_errors=True)
    if labels_storage_account_blob_uri:
        os.remove(labels_csv_path)

    try:
        logger.info(f"Saving processed sensor data to database.")
        save_to_db(sensor_data, table_name="training_data_labeled", db_engine=database_engine)
        pipeline_run.status = "COMPLETED"
    except Exception as e:
        logger.error(f"Error saving processed sensor data to database: {e}")
        pipeline_run.status = "FAILED"
    finally:
        pipeline_run.completed_at = datetime.now()
        logger.info(f"Updating Pipeline Run status to {pipeline_run.status}.")
        update_pipeline_run_status(database_engine, run_id=pipeline_run.run_id, status=pipeline_run.status, completed_at=pipeline_run.completed_at)


if __name__ == "__main__":
    asyncio.run(run_data_pipeline_async(
        pipeline_run_id=pipeline_run_id,
        raw_data_file_dir=raw_data_file_dir,
        raw_data_storage_account_container_uri=raw_data_storage_account_container_uri,
        labels_csv_path=labels_csv_path,
        labels_storage_account_blob_uri=labels_storage_account_blob_uri,
        kaggle_csv_path=kaggle_csv_path
    ))