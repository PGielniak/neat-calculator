# data_pipeline.py — Refactor Notes

## What Changed

### 1. Monolith → Small, Named Functions

`run_data_pipeline_async` was a single ~110-line function doing everything.
It is now an orchestrator that delegates to five focused helpers:

| Function | Responsibility |
|---|---|
| `_resolve_raw_data_source(...)` | Validates that a raw-data source was given; downloads from blob if needed; returns local dir path |
| `_resolve_labels_source(...)` | Validates that a labels source was given; downloads from blob if needed; returns local CSV path |
| `_build_file_manifest(...)` | Scans the local raw-data dir and returns a sorted DataFrame of `(file_name, pipeline_run_id, checksum)` |
| `_register_files(...)` | Persists each file record to `processed_files`, handles duplicate-checksum dedup, returns `(processed_files, skipped_files)` |
| `_process_sensor_data(...)` | Wraps `process_raw_sensor_data` with logging context; returns the processed DataFrame |
| `_persist_sensor_data(...)` | Saves the DataFrame to `training_data_labeled`; returns `True/False` instead of raising so the `finally` block can always run |

### 2. Fixed Broken `__main__` Block

The original `__main__` block called `asyncio.run(run_data_pipeline_async(...))` with bare variable names (`pipeline_run_id`, `raw_data_file_dir`, …) that were parsed earlier by argparse but **never assigned correctly** — they would raise `NameError` at runtime.

Fixed: all `args.*` values are now wired through to the async call, including the new `--use_v2_features` flag and proper `or ""` guards for `Optional` string args.

### 3. Logging Improvements

- **Structured key=value pairs** in every log message (`run_id=…`, `file_name=…`, `rows=…`) so logs can be searched/filtered without regex gymnastics.
- `logger.debug` for per-file DB writes (avoids flooding INFO in production with hundreds of lines per run).
- `logger.info` with **counts** at the end of each stage: `new=42 | skipped=3`.
- **Duration** logged at finalisation: `duration_s=12.4`.
- `exc_info=True` on the top-level exception catch so the full traceback appears in the log, not just the message string.
- Removed leftover `logger.info(f"Saving Processed Files info to database.")` that appeared *after* the loop had already saved them.
- `logging.basicConfig(...)` moved into the `__main__` block — it only makes sense when running as a script, not when imported as a module.

### 4. Removed `shutil` Import

`shutil` was imported but only used in commented-out cleanup code. Removed to keep imports honest.

---

## Suggestions for Future Improvements

### Checksum on File *Contents*, Not Filename
`hashlib.md5(file_name.encode())` hashes the name, not the bytes.  
Two different files with the same name will falsely collide; the same file renamed will be reprocessed.  
**Fix:** read the file in chunks and hash its content:
```python
def _file_checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

### Replace `asyncio.get_event_loop().run_in_executor` in webhook.py
`asyncio.get_event_loop()` is deprecated in Python ≥ 3.10 when called from a non-async context.  
Use `asyncio.get_running_loop()` inside an `async` function, or restructure to use `BackgroundTasks` (FastAPI built-in) which handles thread-pool dispatch cleanly.

### Idempotent Pipeline Runs
Currently, if a run fails mid-way, re-triggering with the same `pipeline_run_id` will hit the `PRIMARY KEY` constraint on `pipeline_runs`.  
Consider an **upsert** (`INSERT … ON CONFLICT DO UPDATE`) or a status check + resume strategy.

### Async File Downloads
`download_blob_to_dir` is called synchronously inside `run_data_pipeline_async`.  
If the function is genuinely async-capable it should be awaited; if it is blocking I/O it should run in an executor to avoid blocking the event loop:
```python
loop = asyncio.get_running_loop()
await loop.run_in_executor(None, download_blob_to_dir, ...)
```

### Parallel File Registration
`_register_files` inserts files one-by-one in a Python loop.  
For runs with many files this is slow. A bulk insert (`repository.save_dataframe`) would be faster, with a post-insert query to identify which checksums were already present.

### `pipeline_run.completed_at` Type Safety
`PipelineRun.started_at` is set at construction time but `completed_at` is set later.  
The duration calculation in the `finally` block assumes both are always set — add a guard or make `started_at` a `datetime` field with a default factory.

### Typed Return for `_persist_sensor_data`
`bool` is a weak signal. Consider returning a `dataclasses.dataclass` result object (status, rows_written, error) so callers don't need out-of-band logging to understand what happened.
