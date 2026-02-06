from json import load
from fastapi import FastAPI, Depends, Query, HTTPException, status, Response
import logging
# from fastapi.middleware.cors import CORSMiddleware
from infra.db.database_utils import get_postgres_db_engine
from dataclasses import dataclass

from model_training.train_model import train_model_async

app = FastAPI()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set debug level for all data_pipeline loggers
logging.getLogger('data_pipeline').setLevel(logging.DEBUG)
logging.getLogger('azure').setLevel(logging.WARNING)  # Reduce Azure SDK noise
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],  # Explicitly list allowed methods
#     allow_headers=["*"],  # Allow all headers
#     expose_headers=["*"],  # Expose all headers
#     max_age=3600,  # Cache preflight requests for 1 hour
# )

# @dataclass
# class WebhookPayload:


    

@app.post("/train_model_webhook", status_code=status.HTTP_201_CREATED)
async def handle_webhook(db_engine=Depends(get_postgres_db_engine)):
    try:
        logger.info(f"Triggering model training")
        await train_model_async(db_engine=db_engine)
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model training failed.")
    return {"message": "Model training started in background."}


