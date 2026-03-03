#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_tests.sh
#
# Runs all unit tests for the ml_microservices project and generates an HTML
# coverage report plus a JUnit XML file (for CI integration).
#
# Usage:
#   ./run_tests.sh               # run all tests with default options
#   ./run_tests.sh --no-report   # skip the HTML report (faster)
#   ./run_tests.sh -k "helper"   # run only tests matching "helper"
#
# Any extra arguments are forwarded directly to pytest.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Python interpreter
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# PYTHONPATH: expose every micro-service source tree
# ---------------------------------------------------------------------------
SRC_TREES=(
    "src/ml_microservices/shared/src"
    "src/ml_microservices/database/src"
    "src/ml_microservices/data_pipeline/src"
    "src/ml_microservices/model_training/src"
    "src/ml_microservices/prediction_api/src"
)

PYTHONPATH_EXTRA=""
for tree in "${SRC_TREES[@]}"; do
    PYTHONPATH_EXTRA="${SCRIPT_DIR}/${tree}:${PYTHONPATH_EXTRA}"
done
export PYTHONPATH="${PYTHONPATH_EXTRA}${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
echo "Checking required packages..."
REQUIRED_PACKAGES=(
    pytest
    pytest_cov
    pytest_asyncio
    httpx
    numpy
    scipy
    pandas
    sqlalchemy
    fastapi
    pydantic
    azure.identity
    azure.storage.blob
    mlflow
    xgboost
    sklearn
    joblib
    dotenv
)

MISSING=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! "$PYTHON" -c "import ${pkg}" &>/dev/null; then
        MISSING+=("${pkg//_/-}")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "The following packages are missing and will be installed:"
    for pkg in "${MISSING[@]}"; do echo "  - $pkg"; done
    echo ""
    pip install \
        pytest pytest-cov pytest-asyncio httpx \
        numpy scipy pandas sqlalchemy \
        "fastapi>=0.100" pydantic \
        "azure-identity>=1.0" "azure-storage-blob>=12.0" \
        mlflow xgboost scikit-learn joblib python-dotenv \
        psycopg2-binary
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
GENERATE_REPORT=true
PYTEST_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--no-report" ]]; then
        GENERATE_REPORT=false
    else
        PYTEST_ARGS+=("$arg")
    fi
done

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
REPORT_DIR="${SCRIPT_DIR}/test_reports"
mkdir -p "$REPORT_DIR"

COMMON_FLAGS=(
    --tb=short
    --strict-markers
    --junit-xml="${REPORT_DIR}/results.xml"
)

if $GENERATE_REPORT; then
    COV_FLAGS=(
        --cov=src
        --cov-report=html:"${REPORT_DIR}/coverage_html"
        --cov-report=term-missing
        --cov-report=xml:"${REPORT_DIR}/coverage.xml"
    )
else
    COV_FLAGS=()
fi

echo ""
echo "=========================================="
echo " Running ml_microservices unit tests"
echo "=========================================="
echo ""

"$PYTHON" -m pytest \
    tests/ \
    "${COMMON_FLAGS[@]}" \
    "${COV_FLAGS[@]}" \
    "${PYTEST_ARGS[@]}"

EXIT_CODE=$?

if $GENERATE_REPORT; then
    echo ""
    echo "=========================================="
    echo " Reports saved to: ${REPORT_DIR}/"
    echo "   HTML coverage : coverage_html/index.html"
    echo "   XML coverage  : coverage.xml"
    echo "   JUnit XML     : results.xml"
    echo "=========================================="
fi

exit $EXIT_CODE
