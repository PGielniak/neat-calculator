"""
Shared pytest fixtures and path configuration for ml_microservices tests.
"""
import sys
import os

# ---------------------------------------------------------------------------
# Add each micro-service source tree to the Python path so that tests can
# import packages (shared, database, data_pipeline, model_training,
# prediction_api) without installing them.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SOURCE_TREES = [
    "src/ml_microservices/shared/src",
    "src/ml_microservices/database/src",
    "src/ml_microservices/data_pipeline/src",
    "src/ml_microservices/model_training/src",
    "src/ml_microservices/prediction_api/src",
]

for _tree in _SOURCE_TREES:
    _path = os.path.join(_REPO_ROOT, _tree)
    if _path not in sys.path:
        sys.path.insert(0, _path)
