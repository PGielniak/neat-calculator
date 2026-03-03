"""
Shared pytest configuration and fixtures for ml_microservices tests.
Adds all microservice source packages to sys.path so they can be imported.
"""
import sys
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add every sub-package source tree to sys.path so plain imports work
_SRC_PATHS = [
    os.path.join(_BASE_DIR, "src", "ml_microservices", "shared", "src"),
    os.path.join(_BASE_DIR, "src", "ml_microservices", "database", "src"),
    os.path.join(_BASE_DIR, "src", "ml_microservices", "data_pipeline", "src"),
    os.path.join(_BASE_DIR, "src", "ml_microservices", "prediction_api", "src"),
    os.path.join(_BASE_DIR, "src", "ml_microservices", "model_training", "src"),
]

for path in _SRC_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)
