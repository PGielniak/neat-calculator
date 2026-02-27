import logging
import uuid
import os
import pandas as pd
import argparse
import hashlib
from datetime import datetime
from typing import List, Tuple, Optional
from sqlalchemy.exc import IntegrityError
import asyncio

from data_pipeline.models import PipelineRun, ProcessedFile
from data_pipeline.database_service import save_to_db, initialize_tables, update_pipeline_run_status, get_db_engine
from shared.storage_account_helpers import download_blob_to_dir
from shared.process_raw_data import process_raw_sensor_data


def _resolve_raw_data_source(
    raw_data_file_dir: str,
    raw_data_storage_account_container_uri: str,
    logger: logging.Logger,
) -> str:
    """
    Ensures raw data is available locally and returns the local directory path.
    Downloads from blob storage if a container URI is provided.
    Raises ValueError if neither source is specified.
    """
    if not raw_data_file_dir and not raw_data_storage_account_container_uri:
        raise ValueError(
            "Either raw_data_file_dir or raw_data_storage_account_container_uri must be provided."
        )

    raw_data_dir = raw_data_file_dir if raw_data_file_dir else "raw_sensor_data_files"

    if raw_data_storage_account_container_uri:
        logger.info(
            "Downloading raw data files from blob storage | container_uri=%s | target_dir=%s",
            raw_data_storage_account_container_uri,
            raw_data_dir,
        )
        download_blob_to_dir(
            storage_account_blob_uri=raw_data_storage_account_container_uri,
            download_dir=raw_data_dir,
            logger=logger,
        )
        logger.info("Raw data download complete | target_dir=%s", raw_data_dir)

    return raw_data_dir


def _resolve_labels_source(
    labels_csv_path: str,
    labels_storage_account_blob_uri: str,
    logger: logging.Logger,
) -> str:
    """
    Ensures the labels CSV is available locally and returns its path.
    Downloads from blob storage if a blob URI is provided.
    Raises ValueError if neither source is specified.
    """
    if not labels_csv_path and not labels_storage_account_blob_uri:
        raise ValueError(
            "Either labels_csv_path or labels_storage_account_blob_uri must be provided."
        )

    if labels_storage_account_blob_uri:
        logger.info(
            "Downloading labels CSV from blob storage | blob_uri=%s",
            labels_storage_account_blob_uri,
        )
        download_blob_to_dir(
            storage_account_blob_uri=labels_storage_account_blob_uri,
            download_dir=".",
            logger=logger,
        )
        resolved_path = os.path.join(".", os.path.basename(labels_storage_account_blob_uri))
        logger.info("Labels CSV download complete | local_path=%s", resolved_path)
        return resolved_path

    return labels_csv_path


def _build_file_manifest(raw_data_dir: str, pipeline_run_id: str) -> pd.DataFrame:
    """
    Scans raw_data_dir and returns a sorted DataFrame with columns:
    file_name, pipeline_run_id, checksum (MD5 of file_name).
    """
    file_names = sorted(os.listdir(raw_data_dir))
    manifest = pd.DataFrame(file_names, columns=["file_name"])
    manifest["pipeline_run_id"] = pipeline_run_id
    manifest["checksum"] = manifest["file_name"].apply(
        lambda name: hashlib.md5(name.encode()).hexdigest()
    )
    return manifest


def _register_files(
    manifest: pd.DataFrame,
    db_engine,
    logger: logging.Logger,
) -> Tuple[List[ProcessedFile], List[str]]:
    """
    Persists each file in the manifest to the processed_files table.
    Deduplicates by checksum — already-seen files are added to skipped_files.
    Returns (processed_files, skipped_files).
    """
    processed_files: List[ProcessedFile] = []
    skipped_files: List[str] = []

    for _, row in manifest.iterrows():
        processed_file = ProcessedFile(
            file_id=str(uuid.uuid4()),
            file_name=row["file_name"],
            pipeline_run_id=row["pipeline_run_id"],
            processed_at=datetime.now(),
            checksum=row["checksum"],
        )
        try:
            save_to_db(processed_file, table_name="processed_files", db_engine=db_engine)
            logger.debug(
                "Registered file | file_name=%s | checksum=%s",
                row["file_name"],
                row["checksum"],
            )
            processed_files.append(processed_file)
        except IntegrityError as exc:
            if 'duplicate key value violates unique constraint "processed_files_checksum_key"' in str(exc):
                logger.warning(
                    "Skipping duplicate file | file_name=%s | checksum=%s",
                    row["file_name"],
                    row["checksum"],
                )
                skipped_files.append(row["file_name"])
            else:
                logger.error(
                    "Unexpected integrity error registering file | file_name=%s | error=%s",
                    row["file_name"],
                    exc,
                )
                raise
        except Exception as exc:
            logger.error(
                "Failed to register file | file_name=%s | error=%s",
                row["file_name"],
                exc,
            )
            raise

    logger.info(
        "File registration complete | new=%d | skipped=%d",
        len(processed_files),
        len(skipped_files),
    )
    return processed_files, skipped_files


def _process_sensor_data(
    raw_data_dir: str,
    labels_csv_path: str,
    kaggle_csv_path: str,
    skipped_files: List[str],
    use_v2_features: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Runs feature extraction / processing on the raw sensor files.
    Returns a DataFrame of processed sensor data ready for persistence.
    """
    version = "2" if use_v2_features else "1"
    logger.info(
        "Starting sensor data processing | version=%s | raw_data_dir=%s | skipped=%d files",
        version,
        raw_data_dir,
        len(skipped_files),
    )
    sensor_data = process_raw_sensor_data(
        raw_data_file_dir=raw_data_dir,
        labels_csv_path=labels_csv_path,
        kaggle_csv_path=kaggle_csv_path,
        skipped_files=skipped_files,
        version=version,
    )
    logger.info(
        "Sensor data processing complete | rows=%d | columns=%d",
        len(sensor_data),
        len(sensor_data.columns),
    )
    return sensor_data


def _persist_sensor_data(
    sensor_data: pd.DataFrame,
    db_engine,
    logger: logging.Logger,
) -> bool:
    """
    Saves processed sensor data to the training_data_labeled table.
    Returns True on success, False on failure (logs the error but does not raise).
    """
    logger.info(
        "Saving processed sensor data | rows=%d | table=training_data_labeled",
        len(sensor_data),
    )
    try:
        save_to_db(sensor_data, table_name="training_data_labeled", db_engine=db_engine)
        logger.info("Sensor data saved successfully")
        return True
    except Exception as exc:
        logger.error("Failed to save sensor data to database | error=%s", exc)
        return False


async def run_data_pipeline_async(
    pipeline_run_id: str = "",
    raw_data_file_dir: str = "",
    raw_data_storage_account_container_uri: str = "",
    labels_csv_path: str = "",
    labels_storage_account_blob_uri: str = "",
    kaggle_csv_path: str = "",
    db_engine=None,
    use_v2_features: bool = False,
):
    """
    Orchestrates the full data pipeline:
      1. Resolve / download raw data source
      2. Resolve / download labels source
      3. Build file manifest and register files (dedup by checksum)
      4. Process raw sensor data into features
      5. Persist results; update pipeline run status
    """
    logger = logging.getLogger(__name__)

    database_engine = db_engine if db_engine is not None else get_db_engine()

    logger.info("Initializing database tables if they do not exist.")
    initialize_tables(database_engine)

    if not pipeline_run_id:
        pipeline_run_id = str(uuid.uuid4())

    pipeline_run = PipelineRun(
        run_id=pipeline_run_id,
        status="STARTED",
        started_at=datetime.now(),
        folder_path=raw_data_file_dir,
        labels_csv_path=labels_csv_path,
    )
    logger.info("Pipeline run created | run_id=%s", pipeline_run.run_id)
    save_to_db(pipeline_run, table_name="pipeline_runs", db_engine=database_engine)

    try:
        raw_data_dir = _resolve_raw_data_source(
            raw_data_file_dir, raw_data_storage_account_container_uri, logger
        )

        resolved_labels_path = _resolve_labels_source(
            labels_csv_path, labels_storage_account_blob_uri, logger
        )

        manifest = _build_file_manifest(raw_data_dir, pipeline_run_id)
        logger.info("File manifest built | total_files=%d | run_id=%s", len(manifest), pipeline_run_id)

        _processed_files, skipped_files = _register_files(manifest, database_engine, logger)

        sensor_data = _process_sensor_data(
            raw_data_dir=raw_data_dir,
            labels_csv_path=resolved_labels_path,
            kaggle_csv_path=kaggle_csv_path,
            skipped_files=skipped_files,
            use_v2_features=use_v2_features,
            logger=logger,
        )

        success = _persist_sensor_data(sensor_data, database_engine, logger)
        pipeline_run.status = "COMPLETED" if success else "FAILED"

    except Exception as exc:
        logger.error(
            "Pipeline run failed | run_id=%s | error=%s",
            pipeline_run.run_id,
            exc,
            exc_info=True,
        )
        pipeline_run.status = "FAILED"
        raise
    finally:
        pipeline_run.completed_at = datetime.now()
        logger.info(
            "Finalising pipeline run | run_id=%s | status=%s | duration_s=%.1f",
            pipeline_run.run_id,
            pipeline_run.status,
            (pipeline_run.completed_at - pipeline_run.started_at).total_seconds(),
        )
        update_pipeline_run_status(
            database_engine,
            run_id=pipeline_run.run_id,
            status=pipeline_run.status,
            completed_at=pipeline_run.completed_at,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process raw sensor data files.")
    argparser.add_argument("--raw_data_file_dir", type=str, required=False)
    argparser.add_argument("--raw_data_storage_account_container_uri", type=str, required=False)
    argparser.add_argument("--labels_csv_path", type=str, required=False)
    argparser.add_argument("--labels_storage_account_blob_uri", type=str, required=False)
    argparser.add_argument("--kaggle_csv_path", type=str, required=False, default="kaggle.csv")
    argparser.add_argument("--pipeline_run_id", type=str, required=False, default="")
    argparser.add_argument("--use_v2_features", action="store_true", default=False)
    args = argparser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(
        run_data_pipeline_async(
            pipeline_run_id=args.pipeline_run_id,
            raw_data_file_dir=args.raw_data_file_dir or "",
            raw_data_storage_account_container_uri=args.raw_data_storage_account_container_uri or "",
            labels_csv_path=args.labels_csv_path or "",
            labels_storage_account_blob_uri=args.labels_storage_account_blob_uri or "",
            kaggle_csv_path=args.kaggle_csv_path,
            use_v2_features=args.use_v2_features,
        )
    )