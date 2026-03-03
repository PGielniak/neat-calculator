# ml_microservices

A collection of Python micro-services that handle the full machine-learning
lifecycle for the NEAT Calculator project: raw-sensor data ingestion, feature
engineering, model training and a prediction REST API.

## Service layout

| Service | Package | Purpose |
|---------|---------|---------|
| `data_pipeline` | `data_pipeline` | Ingest raw JSON sensor files, extract HAR features, store in PostgreSQL |
| `database` | `database` | Shared database abstraction (SQLite / PostgreSQL) |
| `model_training` | `model_training` | Train an XGBoost HAR classifier and track experiments with MLflow |
| `prediction_api` | `prediction_api` | FastAPI endpoint that performs real-time activity classification |
| `shared` | `shared` | Shared signal-processing helpers and raw-data processing utilities |

---

## Running the unit tests

### Prerequisites

- **Python 3.12** (required; other versions are untested)
- **pip** (comes with Python)
- No external services (PostgreSQL, MLflow, Azure Storage) are required — all
  I/O is mocked in the test suite.

### Quick start

```bash
cd ml_microservices
./run_tests.sh
```

The script will:
1. Verify that every required Python package is installed (and `pip install`
   any that are missing).
2. Configure `PYTHONPATH` so that every micro-service package is importable.
3. Run the full test suite with `pytest`.
4. Write a **JUnit XML** result file and an **HTML coverage report** to
   `test_reports/`.

### Running without the coverage report

```bash
./run_tests.sh --no-report
```

### Running a single test file

```bash
./run_tests.sh tests/test_helper_functions.py
```

### Running tests that match a keyword

```bash
./run_tests.sh -k "entropy or angle"
```

Any arguments after the (optional) `--no-report` flag are forwarded directly
to `pytest`.

### Running with pytest directly

If you prefer to call pytest yourself, export `PYTHONPATH` first:

```bash
export PYTHONPATH=\
src/ml_microservices/shared/src:\
src/ml_microservices/database/src:\
src/ml_microservices/data_pipeline/src:\
src/ml_microservices/model_training/src:\
src/ml_microservices/prediction_api/src

python3 -m pytest tests/ -v
```

---

## Test reports

After a successful run the `test_reports/` directory contains:

| File | Description |
|------|-------------|
| `results.xml` | JUnit-compatible XML — usable by GitHub Actions, Jenkins, etc. |
| `coverage.xml` | Cobertura XML coverage report |
| `coverage_html/index.html` | Interactive HTML coverage report — open in a browser |

---

## Test suite overview

| Test file | Module under test | Tests |
|-----------|-------------------|-------|
| `tests/test_helper_functions.py` | `shared/helper_functions.py` | `correlation`, `energy`, `entropy`, `sma`, `mean_freq`, `lowpass_filter`, `angle_between`, `extract_features` |
| `tests/test_process_raw_data.py` | `shared/process_raw_data.py` | `validate_directory`, `validate_labels_csv`, `remove_duplicates`, `resample_data`, `create_sliding_windows`, `extract_features_from_windows`, `rename_features`, `filter_features_to_match_kaggle` |
| `tests/test_database_utils.py` | `database/database_utils.py` | `SQLiteEngine`, `DatabaseFactory`, `DatabaseRepository`, `get_postgres_db_engine` |
| `tests/test_data_pipeline.py` | `data_pipeline/data_pipeline.py` | `_resolve_raw_data_source`, `_resolve_labels_source`, `_build_file_manifest`, `_register_files`, `_persist_sensor_data`, `_process_sensor_data` |
| `tests/test_train_model.py` | `model_training/train_model.py` | `get_required_env`, `balance_classes`, `drop_columns_with_too_much_importance`, `prepare_data_for_training`, `find_best_model`, `save_ml_artifacts_to_file`, `prepare_variables`, `setup_and_test_mlflow_connection`, `load_data_fromdb`, `load_kaggle_data_fromdb` |
| `tests/test_prediction_api.py` | `prediction_api/prediction_api.py` | `drop_unnecessary_columns`, `health` endpoint, `predict` endpoint, `run_prediction` |
| `tests/test_load_model.py` | `prediction_api/load_model.py` | `load_production_model` |
| `tests/test_storage_account_helpers.py` | `shared/storage_account_helpers.py` | `get_blob_service_client`, `list_blobs_in_prefix`, `download_blob_to_dir` |

### Design principles

- **No external services needed** — PostgreSQL, MLflow, and Azure Blob Storage
  are all mocked using `unittest.mock`.
- **SQLite in-memory** — database tests use an on-disk SQLite file in a
  `tmp_path` pytest fixture that is cleaned up automatically after each test.
- **Deterministic** — all random number generators are seeded so results are
  reproducible across runs.
