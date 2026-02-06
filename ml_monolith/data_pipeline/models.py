from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class PipelineRun(BaseModel):
    run_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    folder_path: str
    labels_csv_path: str

class ProcessedFile(BaseModel):
    file_id: str
    file_name: str
    pipeline_run_id: str
    processed_at: datetime
    checksum: str = Field(..., description="MD5 checksum of the processed file")