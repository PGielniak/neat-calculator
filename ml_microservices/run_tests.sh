#!/usr/bin/env bash
# run_tests.sh
#
# Runs all unit tests for the ml_microservices folder and generates:
#   - An HTML coverage report  → htmlcov/index.html
#   - A JUnit XML report       → test-report.xml
#
# Usage:
#   cd ml_microservices
#   bash run_tests.sh
#
# Options (passed straight through to pytest):
#   bash run_tests.sh -v           verbose output
#   bash run_tests.sh -k "shared"  run only tests matching "shared"
#   bash run_tests.sh --no-cov     skip coverage (faster)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Source paths (added to PYTHONPATH so imports resolve without installation) ─
export PYTHONPATH="\
${SCRIPT_DIR}/src/ml_microservices/shared/src:\
${SCRIPT_DIR}/src/ml_microservices/database/src:\
${SCRIPT_DIR}/src/ml_microservices/data_pipeline/src:\
${SCRIPT_DIR}/src/ml_microservices/prediction_api/src:\
${SCRIPT_DIR}/src/ml_microservices/model_training/src:\
${PYTHONPATH:-}"

# ── Verify pytest is available ─────────────────────────────────────────────
if ! command -v pytest &>/dev/null; then
    echo "pytest not found. Installing test dependencies..."
    pip install \
        pytest \
        pytest-cov \
        pytest-asyncio \
        numpy \
        scipy \
        pandas \
        scikit-learn \
        pydantic \
        sqlalchemy \
        python-dotenv \
        azure-storage-blob \
        azure-identity \
        fastapi \
        matplotlib
fi

echo "========================================"
echo "  Running ml_microservices unit tests   "
echo "========================================"

pytest tests/ \
    --cov=src/ml_microservices/shared/src/shared \
    --cov=src/ml_microservices/database/src/database \
    --cov=src/ml_microservices/data_pipeline/src/data_pipeline \
    --cov=src/ml_microservices/prediction_api/src/prediction_api \
    --cov=src/ml_microservices/model_training/src/model_training \
    --cov-report=html:htmlcov \
    --cov-report=term-missing \
    --junitxml=test-report.xml \
    "$@"

echo ""
echo "========================================"
echo "  Test run complete"
echo "  HTML coverage report : htmlcov/index.html"
echo "  JUnit XML report     : test-report.xml"
echo "========================================"
