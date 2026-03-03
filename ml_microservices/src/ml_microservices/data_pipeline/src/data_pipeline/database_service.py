# database.py
from datetime import datetime
from typing import Any, List, Union
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

from database.database_utils import DatabaseEngine, DatabaseRepository, DatabaseFactory, get_postgres_db_engine

def get_db_engine() -> DatabaseEngine:
    return get_postgres_db_engine()

def initialize_tables(db_engine: DatabaseEngine) -> None:
    """
    Initializes the required tables in the database if they do not exist.
    Creates pipeline_runs and processed_files tables with proper schema.
    """
    from sqlalchemy import text
    
    # Get the underlying SQLAlchemy engine
    engine = db_engine.engine
    
    with engine.connect() as conn:
        # Create pipeline_runs table with run_id as PRIMARY KEY
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                folder_path TEXT,
                labels_csv_path TEXT
            )
        """))
        
        # Create processed_files table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS processed_files (
                file_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                pipeline_run_id TEXT NOT NULL,
                processed_at TIMESTAMP NOT NULL,
                checksum TEXT UNIQUE,
                FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(run_id)
            )
        """))
        
        conn.commit()
    
    print("Database tables initialized successfully")

def save_to_db(data: Union[BaseModel, List[BaseModel], dict, List[dict], pd.DataFrame], table_name: str, db_engine: DatabaseEngine) -> None:
    """
    Saves the given data to the specified database table.
    Handles both Pydantic models and dictionaries.
    """
    from database.database_utils import DatabaseRepository
    
    repository = DatabaseRepository(db_engine)
    
    try:
        if isinstance(data, pd.DataFrame):
            repository.save_dataframe(data, table_name)
        # Convert Pydantic model(s) to dict(s)
        elif isinstance(data, list):
            # List of records
            records = []
            for item in data:
                if isinstance(item, BaseModel):
                    records.append(item.model_dump())  # or item.dict() for older pydantic
                else:
                    records.append(item)
            repository.save_records(records, table_name)
        else:
            # Single record
            if isinstance(data, BaseModel):
                record = data.model_dump()  # or data.dict() for older pydantic
            else:
                record = data
            repository.save_record(record, table_name)
            
    except Exception as e:
        print(f"Error saving to database: {e}")
        raise e
    finally:
        repository.close()
        
def update_pipeline_run_status(db_engine: DatabaseEngine, run_id: str, status: str, completed_at: datetime) -> None:
    """
    Updates the status of a pipeline run in the database.
    """
    repository = DatabaseRepository(engine=db_engine)
    
    updates = {
        "status": status,
        "completed_at": completed_at
    }
    
    try:
        repository.update_record(
            table_name="pipeline_runs",
            primary_key="run_id",
            key_value=run_id,
            updates=updates
        )
    except Exception as e:
        # Handle or log the exception as needed
        raise e
    finally:
        repository.close()
        
        
def get_pipeline_run_status(db_engine: DatabaseEngine, run_id: str) -> str:
    """
    Updates the status of a pipeline run in the database.
    """
    repository = DatabaseRepository(engine=db_engine)
    
    try:
        pipeline_status = repository.get_records(
            table_name="pipeline_runs", filters={"run_id": run_id}
        )
    except Exception as e:
        # Handle or log the exception as needed
        raise e
    finally:
        repository.close()
    if pipeline_status.empty:
        return "RUN_ID_NOT_FOUND"
    pipeline_status = pipeline_status['status'][0]
    return pipeline_status