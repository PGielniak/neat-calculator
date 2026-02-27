
# Set PYTHONPATH 
export PYTHONPATH="/home/pitoleo/src/neat-calculator/ml_microservices/data_pipeline"

# Activate virtual environment and run server
/home/pitoleo/src/neat-calculator/ml_microservices/data_pipeline/.venv/bin/python3 -m uvicorn webhook:app --host 0.0.0.0 --port 8022 --reload