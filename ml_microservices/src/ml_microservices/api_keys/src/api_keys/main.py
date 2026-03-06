from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
import logging
from database.database_utils import get_postgres_db_engine, DatabaseRepository
from api_keys.api_key_service import ApiKeyService
from api_keys.api_key_models import Base

app = FastAPI()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global database repository (initialized at startup)
_db_repo: Optional[DatabaseRepository] = None

@app.on_event("startup")
async def startup_event():
    global _db_repo
    logger.info("Initializing database engine...")
    engine = get_postgres_db_engine()
    Base.metadata.create_all(bind=engine.engine)
    _db_repo = DatabaseRepository(engine)
    logger.info("Database engine initialized successfully")


def get_service() -> ApiKeyService:
    if _db_repo is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not ready")
    return ApiKeyService(_db_repo)


# --- Schemas ---

class GenerateApiKeyRequest(BaseModel):
    prefix: str = Field(default="ak", max_length=10)
    rate_limit_req_no: int = 30
    rate_limit_interval_minutes: int = 1
    comment: Optional[str] = None

class GenerateApiKeyResponse(BaseModel):
    raw_key: str
    prefix: str

class ValidateApiKeyRequest(BaseModel):
    raw_key: str

class ToggleApiKeyRequest(BaseModel):
    prefix: str

class ValidateApiKeyResponse(BaseModel):
    valid: bool
    rate_limit_req_no: int = 30
    rate_limit_interval_minutes: int = 1
   

# --- Endpoints ---

@app.post("/api-keys/generate", response_model=GenerateApiKeyResponse)
def generate_api_key(
    request: GenerateApiKeyRequest,
    service: ApiKeyService = Depends(get_service)
):
    raw_key, prefix = service.generate_api_key(request.prefix, request.rate_limit_req_no, request.rate_limit_interval_minutes, request.comment)
    return GenerateApiKeyResponse(raw_key=raw_key, prefix=prefix)


@app.post("/api-keys/validate", response_model=ValidateApiKeyResponse)
def validate_api_key(request: ValidateApiKeyRequest, service: ApiKeyService = 
  Depends(get_service)):
    api_key = service.get_api_key_details(request.raw_key)  # returns None or ApiKey
    if api_key is None:
        return ValidateApiKeyResponse(valid=False)
    return ValidateApiKeyResponse(
        valid=True,
        rate_limit_req_no=api_key.rate_limit_req_no,
        rate_limit_interval_minutes=api_key.rate_limit_interval_minutes,
    )


@app.patch("/api-keys/disable")
def disable_api_key(request: ToggleApiKeyRequest, service: ApiKeyService = Depends(get_service)):
    found = service.disable_api_key(request.prefix)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"API key with prefix '{request.prefix}' not found")
    return {"detail": f"API key '{request.prefix}' disabled"}


@app.patch("/api-keys/enable")
def enable_api_key(
    request: ToggleApiKeyRequest,
    service: ApiKeyService = Depends(get_service)
):
    found = service.enable_api_key(request.prefix)
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"API key with prefix '{request.prefix}' not found")
    return {"detail": f"API key '{request.prefix}' enabled"}