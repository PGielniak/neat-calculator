from json import load
from fastapi import FastAPI, Depends, Query, HTTPException, status, Response
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
import logging
# from fastapi.middleware.cors import CORSMiddleware
import asyncio
import subprocess
import platform
from data_pipeline.database_service import get_pipeline_run_status
from dataclasses import dataclass
from data_pipeline.data_pipeline import run_data_pipeline_async
from data_pipeline.database_service import get_db_engine

app = FastAPI()
executor = ThreadPoolExecutor()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set debug level for all data_pipeline loggers
logging.getLogger('data_pipeline').setLevel(logging.DEBUG)
logging.getLogger('azure').setLevel(logging.WARNING)  # Reduce Azure SDK noise

# Global database engine (initialized at startup)
_db_engine = None

@app.on_event("startup")
async def startup_event():
    global _db_engine
    logger.info("Initializing database engine...")
    _db_engine = get_db_engine()
    logger.info("Database engine initialized successfully")

def get_cached_db_engine():
    """Dependency that returns the cached database engine"""
    if _db_engine is None:
        raise RuntimeError("Database engine not initialized")
    return _db_engine
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],  # Explicitly list allowed methods
#     allow_headers=["*"],  # Allow all headers
#     expose_headers=["*"],  # Expose all headers
#     max_age=3600,  # Cache preflight requests for 1 hour
# )

@dataclass
class WebhookPayload:
    kaggle_csv_path: str = None
    raw_data_file_dir: str = None
    raw_data_storage_account_container_uri: str = None
    labels_csv_path: str = None
    labels_storage_account_blob_uri: str = None
    use_v2_features: bool = False  # New parameter for v2
    


def run_pipeline_sync(*args, **kwargs):
    asyncio.run(run_data_pipeline_async(*args, **kwargs))
    
@app.post("/data_pipeline_webhook2", status_code=status.HTTP_201_CREATED)
async def handle_webhook_v2(
    payload: WebhookPayload,
    db = Depends(get_cached_db_engine),
    debug: bool = Query(False, description="Run pipeline synchronously for debugging")
):
    # V2 automatically enables new features
    payload.use_v2_features = True
    return await handle_webhook(payload, db, debug)


@app.post("/data_pipeline_webhook", status_code=status.HTTP_201_CREATED)
async def handle_webhook(
    payload: WebhookPayload,
    db = Depends(get_cached_db_engine),
    debug: bool = Query(False, description="Run pipeline synchronously for debugging")
):
    pipeline_run_id = str(uuid.uuid4())
    raw_data_file_dir = payload.raw_data_file_dir if payload.raw_data_file_dir else ""
    raw_data_storage_account_container_uri = payload.raw_data_storage_account_container_uri if payload.raw_data_storage_account_container_uri else ""
    labels_csv_path = payload.labels_csv_path if payload.labels_csv_path else ""
    labels_storage_account_blob_uri = payload.labels_storage_account_blob_uri if payload.labels_storage_account_blob_uri else ""
    kaggle_csv_path = payload.kaggle_csv_path if payload.kaggle_csv_path else "kaggle.csv"
    logger.info(f"Triggering data pipeline run: {pipeline_run_id} with payload: {payload} (debug={debug})")

    if debug:
        logger.info(f"Running data pipeline {pipeline_run_id} synchronously in DEBUG MODE")
        try:
            await run_data_pipeline_async(
                pipeline_run_id=pipeline_run_id,
                raw_data_file_dir=raw_data_file_dir,
                raw_data_storage_account_container_uri=raw_data_storage_account_container_uri,
                labels_csv_path=labels_csv_path,
                labels_storage_account_blob_uri=labels_storage_account_blob_uri,
                kaggle_csv_path=kaggle_csv_path,
                db_engine=db,
                use_v2_features=payload.use_v2_features
            )
            return {"message": "data pipeline completed (debug mode).", "pipeline_run_id": pipeline_run_id, "data": payload}
        except Exception as e:
            logger.error(f"Pipeline failed in debug mode: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    else:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            executor,
            run_pipeline_sync,
            pipeline_run_id,
            raw_data_file_dir,
            raw_data_storage_account_container_uri,
            labels_csv_path,
            labels_storage_account_blob_uri,
            kaggle_csv_path,
            db,
            payload.use_v2_features  # Pass new parameter
        )
        logger.info(f"Scheduled data pipeline run: {pipeline_run_id}")
        return {"message": "data pipeline triggered. use the pipeline run id to track progress.", "pipeline_run_id": pipeline_run_id, "data": payload}


@app.get("/v2/data_pipeline_status/{pipeline_run_id}")
async def get_pipeline_status(pipeline_run_id: str, db=Depends(get_cached_db_engine)):
    status_result = get_pipeline_run_status(db_engine=db, run_id=pipeline_run_id)
    if status_result == "RUN_ID_NOT_FOUND":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline run ID not found")
    return {"pipeline_run_id": pipeline_run_id, "status": status_result}