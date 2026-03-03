

uv run --package model-training uvicorn model_training.webhook:app --port 8001

uv run --package prediction-api uvicorn prediction_api.prediction_api:app --port 8006