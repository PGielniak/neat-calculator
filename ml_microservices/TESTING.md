# Testing the ml_microservices package

This document explains how to run the unit tests for every module inside the
`ml_microservices` folder and how to read the generated test report.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Python | 3.12 | Exact version required by the workspace (3.12.x) |
| pip | any | Bundled with Python |
| pytest | 8.0 | Installed automatically if missing |
| pytest-cov | 5.0 | Coverage plug-in |
| pytest-asyncio | 0.23 | Async test support |

Install the test dependencies with:

```bash
pip install pytest pytest-cov pytest-asyncio
```

> **Tip** – If you use `uv`, the dependencies are declared in the
> `[dependency-groups] dev` section of `ml_microservices/pyproject.toml` and
> can be installed with:
> ```bash
> uv sync --group dev
> ```

---

## Quick start – run all tests at once

```bash
cd ml_microservices          # must be in the ml_microservices directory
bash run_tests.sh
```

The script:
1. Adds all microservice source trees to `PYTHONPATH` automatically.
2. Runs every test file inside `ml_microservices/tests/`.
3. Prints a coverage summary to the terminal.
4. Generates two artefacts:
   - **`htmlcov/index.html`** – interactive HTML coverage report
   - **`test-report.xml`** – JUnit-compatible XML (for CI systems)

---

## Running tests manually with pytest

If you prefer to call pytest directly, export `PYTHONPATH` first:

```bash
cd ml_microservices

export PYTHONPATH="\
src/ml_microservices/shared/src:\
src/ml_microservices/database/src:\
src/ml_microservices/data_pipeline/src:\
src/ml_microservices/prediction_api/src:\
src/ml_microservices/model_training/src"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_helper_functions.py

# Run tests whose name matches a keyword
pytest tests/ -k "entropy"

# Verbose output
pytest tests/ -v

# Generate coverage report
pytest tests/ --cov=src --cov-report=html:htmlcov --cov-report=term-missing
```

---

## Test file overview

| Test file | Module under test | What is tested |
|-----------|------------------|----------------|
| `tests/test_helper_functions.py` | `shared/helper_functions.py` | `correlation`, `energy`, `entropy`, `sma`, `mean_freq`, `lowpass_filter`, `angle_between`, `extract_features` |
| `tests/test_process_raw_data.py` | `shared/process_raw_data.py` | `validate_directory`, `validate_labels_csv`, `merge_json_files`, `label_data`, `label_data_v2`, `remove_duplicates`, `resample_data`, `create_sliding_windows`, `extract_features_from_windows`, `rename_features`, `filter_features_to_match_kaggle` |
| `tests/test_storage_account_helpers.py` | `shared/storage_account_helpers.py` | `get_blob_service_client`, `download_blob_to_dir`, `list_blobs_in_prefix` |
| `tests/test_database_utils.py` | `database/database_utils.py` | `SQLiteEngine`, `DatabaseFactory`, `DatabaseRepository` |
| `tests/test_database_service.py` | `data_pipeline/database_service.py` | `initialize_tables`, `save_to_db`, `update_pipeline_run_status`, `get_pipeline_run_status` |
| `tests/test_data_pipeline.py` | `data_pipeline/data_pipeline.py` | `_resolve_raw_data_source`, `_resolve_labels_source`, `_build_file_manifest`, `_register_files`, `_process_sensor_data`, `_persist_sensor_data` |
| `tests/test_prediction_api.py` | `prediction_api/prediction_api.py` | `_apply_activity_taxes`, `drop_unnecessary_columns`, `_fetch_model_info`, `_init_mlflow` |
| `tests/test_load_model.py` | `prediction_api/load_model.py` | `load_production_model` |
| `tests/test_train_model.py` | `model_training/train_model.py` | `get_required_env`, `find_best_model`, `balance_classes`, `prepare_data_for_training`, `drop_columns_with_too_much_importance`, `save_ml_artifacts_to_file`, `load_data_fromdb` |

---

## Notes on external dependencies

The tests for modules that interact with external services (Azure Blob
Storage, MLflow, PostgreSQL) use `unittest.mock` to patch all network calls.
**No real credentials or running services are required.**

| Module | External service | How it is mocked |
|--------|-----------------|-----------------|
| `storage_account_helpers` | Azure Blob Storage | `BlobServiceClient` patched |
| `load_model` | MLflow | `mlflow` SDK patched |
| `prediction_api` | MLflow | Module-level `_init_mlflow` / `_load_models` patched |
| `train_model` | MLflow, MLflow DB | `mlflow` module patched at import |
| `data_pipeline` | Azure Blob, DB | `download_blob_to_dir`, `save_to_db` patched |
| `database_service` | PostgreSQL | In-memory SQLite used instead |

---

## Interpreting the HTML coverage report

After running `bash run_tests.sh` (or the equivalent pytest command with
`--cov-report=html`):

1. Open `ml_microservices/htmlcov/index.html` in a browser.
2. Each source file shows the percentage of lines executed by the tests.
3. Lines highlighted in **red** were not reached – consider adding tests for
   those code paths.

---

## Running tests in CI

Add a step like the following to your pipeline:

```yaml
- name: Run ml_microservices unit tests
  working-directory: ml_microservices
  run: bash run_tests.sh
```

The JUnit XML output (`test-report.xml`) can be consumed by most CI systems
(GitHub Actions, Azure Pipelines, Jenkins, etc.) to display test results
inline.
