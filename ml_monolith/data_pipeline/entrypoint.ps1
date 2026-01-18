# ml_monolith/data_pipeline/entrypoint.ps1

# Set PYTHONPATH to include ml_monolith directory
$env:PYTHONPATH = "E:\src\neat-calculator\ml_monolith"

# Activate virtual environment and run server
& E:\src\neat-calculator\ml_monolith\.venv\Scripts\python.exe -m uvicorn data_pipeline.webhook:app --host 0.0.0.0 --port 8000 --reload