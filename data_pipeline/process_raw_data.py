import hashlib
import uuid
import pandas as pd
import os
import json
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Tuple
import logging
import numpy as np
from data_pipeline.helper_functions import extract_features
from infra.db import database_utils
# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_raw_data.log'),
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)

class SensorRecording(BaseModel):
    accelerometerX: float
    accelerometerY: float
    accelerometerZ: float
    gyroscopeX: float
    gyroscopeY: float
    gyroscopeZ: float
    timestamp: int
    timestampNanos: int
    label: str = "UNLABELED"
    
sensor_cols = [
    "accelerometerX","accelerometerY","accelerometerZ",
    "gyroscopeX","gyroscopeY","gyroscopeZ",
    ]

def process_raw_sensor_data(raw_data_file_dir: str, kaggle_csv_path: str, skipped_files: List[str]=[], labels_csv_path = None) -> pd.DataFrame:
    """
    Process raw sensor data files from the specified directory and return a DataFrame
    containing the extracted features and labels.
    """

    
    logger.info(f"Starting processing of raw data in directory: {raw_data_file_dir}")
    logger.info(f"Using labels from: {labels_csv_path}")

    
    merged_data = merge_json_files(raw_data_file_dir, skipped_files=skipped_files)
    if labels_csv_path is None:
        logger.warning("No labels CSV path provided. Proceeding without labeling.")
        labels_csv_path = ""
    labeled_data = label_data(merged_data, labels_csv_path=labels_csv_path)
    del merged_data  # free up memory
    deduped_data = remove_duplicates(labeled_data,sensor_cols=sensor_cols)
    del labeled_data  # free up memory
    merged_data50Hz = resample_data(deduped_data, target_freq=50, sensor_cols=sensor_cols)
    del deduped_data  # free up memory
    sliding_windows, feature_array, labels_array, timestamp_array = create_sliding_windows(merged_data50Hz, window_size=128, step_size=64, sensor_cols=sensor_cols)
    del merged_data50Hz  # free up memory
    
    extracted_features = extract_features_from_windows(sliding_windows, feature_array, labels_array, timestamp_array)
    del sliding_windows, feature_array, labels_array  # free up memory
    renamed_features = rename_features(extracted_features)
    del extracted_features  # free up memory
    intersect_with_kaggle = filter_features_to_match_kaggle(renamed_features, kaggle_csv_path=kaggle_csv_path)   
    del renamed_features  # free up memory
    
    feature_df = pd.DataFrame(intersect_with_kaggle)
    return feature_df

def validate_directory(directory: str, skipped_files: list[str]=[]) -> None:
    logger.info(f"Validating directory: {directory}")
    if not os.path.exists(directory):
        logger.error(f"The directory {directory} does not exist.")
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    logger.info(f"Directory {directory} exists.")
    if not os.path.isdir(directory):
        logger.error(f"The path {directory} is not a directory.")
        raise NotADirectoryError(f"The path {directory} is not a directory.")
    logger.info(f"Path {directory} is a directory.")    
    if not os.listdir(directory):
        logger.error(f"The directory {directory} is empty.")
        raise ValueError(f"The directory {directory} is empty.")
    logger.info(f"Directory {directory} is not empty.")
    sensor_data_files = os.listdir(directory)
    
    if len(skipped_files) == len(sensor_data_files):
        logger.error("All files in the directory are marked to be skipped. No files to validate.")
        raise ValueError("All files in the directory are marked to be skipped. No files to validate.")
    
    for file_name in sensor_data_files:
        if file_name in skipped_files:
            logger.info(f"Skipping validation for file: {file_name}")
            continue
        logger.debug(f"Validating file: {file_name}")
        if not file_name.endswith('.json'):
            logger.error(f"The file {file_name} is not a JSON file.")
            raise ValueError(f"The file {file_name} is not a JSON file.")
        try:
            with open(os.path.join(directory, file_name), 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            logger.error(f"The file {file_name} contains invalid JSON.")
            raise ValueError(f"The file {file_name} contains invalid JSON.")
        
        with open(os.path.join(directory, file_name), 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.error(f"The file {file_name} does not contain a list of sensor recordings.")
                raise ValueError(f"The file {file_name} does not contain a list of sensor recordings.")
            for item in data:
                try:
                    SensorRecording(**item)
                except ValidationError as e:
                    logger.error(f"Validation error in file {file_name}: {e}")
                    logger.debug(f"Invalid item: {item}")
                    logger.debug(f"Validation errors: {e.errors()}")
                    logger.debug(f"Item type: {type(item)}")
                    logger.debug(f"Item keys: {item.keys() if isinstance(item, dict) else 'N/A'}")
                    logger.debug(f"Sample item: {data[0] if data else 'N/A'}")
                    logger.debug(f"Sample item type: {type(data[0]) if data else 'N/A'}")
                    logger.debug(f"Sample item keys: {data[0].keys() if data and isinstance(data[0], dict) else 'N/A'}")
                    logger.debug(f"Full data length: {len(data)}")
                    logger.debug(f"Full data type: {type(data)}")
                    raise ValueError(f"Validation error in file {file_name}: {e}")
        
    logger.info(f"All files in directory {directory} are valid.")

def merge_json_files(directory: str, skipped_files: list[str]=[]) -> list[SensorRecording]:
    
    if len(skipped_files) > 0:
        logger.info(f"Skipping files: {skipped_files}")
        
    validate_directory(directory, skipped_files)
    
    logger.info(f"Merging JSON files from directory: {directory}")
    sensor_data_files = os.listdir(directory)
    sensor_data_files.sort()
    merged_data = []
    for file_name in sensor_data_files:
        if file_name in skipped_files:
            logger.info(f"Skipping file: {file_name}")
            continue
        file_path = os.path.join(directory, file_name)
        logger.info(f"Merging sensor data file: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)

    return merged_data

def label_data(data: list[SensorRecording], labels_csv_path: str) -> pd.DataFrame:
    
    if labels_csv_path == "":
        logger.info("No labels CSV path provided. Labelling with default value 'UNLABELED'.")
        data_df = pd.DataFrame(data)
        data_df['label'] = 'UNLABELED'
        return data_df
    logger.info(f"Labeling data using labels from: {labels_csv_path}")
    data_df = pd.DataFrame(data)
    data_df.sort_values(by=['timestamp'], inplace=True)
    
    labels_df = pd.read_csv(labels_csv_path)
    
    for index, row in labels_df.iterrows():
        start_idx = (data_df['timestamp'] - int(row['timestamp'])).abs().idxmin()
        logger.info(f"Start Index for row with index {index} found: {start_idx}")
        data_df.loc[start_idx:, 'label'] = row['label']
    logger.info("Labeling completed.")
    return data_df
    
def remove_duplicates(data_df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    logger.info("Removing duplicate timestamps and aggregating sensor data.")
    data_df = data_df.sort_values("timestampNanos")
    data_df["ts"] = data_df["timestampNanos"] / 1e9
    data_df["dt"] = data_df["ts"].diff()
        
    agg_dict = {col: "mean" for col in sensor_cols} # this line creates a dictionary mapping each sensor column to the "mean" aggregation function
    print(agg_dict)
    agg_dict["label"] = "first"
    agg_dict["timestamp"] = "first"
    print(agg_dict)
    data_df = (
        data_df
        .groupby("timestampNanos", as_index=False)
        .agg(agg_dict)
    )
    # we grouped the data by timestampNanos and averaged the sensor readings for duplicate timestamps, while keeping the first label.
    data_df["t"] = pd.to_timedelta(data_df["timestampNanos"], unit="ns") # we convert nanoseconds to a timedelta object (pandas Timedelta) because it's easier to work with time intervals
    data_df = data_df.set_index("t") # set the time column as the index of the dataframe for easier time-based slicing and analysis
    data_df = data_df.sort_index() # ensure the data is sorted by time index
    logger.info("Duplicates removed and data aggregated.")
    logger.info(f"Columns after removing duplicates: {data_df.columns.tolist()}")
    return data_df

def resample_data(data_df: pd.DataFrame, target_freq: int, sensor_cols: list[str]) -> pd.DataFrame:
    logger.info(f"Resampling data to target frequency: {target_freq} Hz")
    # 1) put data on 20ms bins, averaging any points that fall in each bin
    resample_miliseconds = int(1000 / target_freq)
    resampled = (
        data_df[sensor_cols]
            .resample(f"{resample_miliseconds}ms")
            .mean()              # <--- this is the key, NOT asfreq()
    )
    resampled = resampled.interpolate("time", limit=500)
    # 2) forward fill labels to match the resampled sensor data    
    resampled["label"] = data_df["label"].resample(f"{resample_miliseconds}ms").first()
    resampled["timestamp"] = data_df["timestamp"].resample(f"{resample_miliseconds}ms").first()
    resampled = resampled.dropna(how="all", subset=sensor_cols)
    data_after_resample = resampled.sort_values("t")
    data_after_resample["dt"] = data_after_resample.index.diff() # with this line we substract each timestamp from the previous one so we get the time gaps.
    data_after_resample["dt"] = data_after_resample["dt"].fillna(pd.Timedelta(0))
    data_after_resample.head()
    # Convert timedelta to seconds, then calculate frequency
    data_after_resample["dt_seconds"] = data_after_resample["dt"].dt.total_seconds()
    # Calculate frequency (Hz) - drop the first row to avoid division by zero
    freq = 1 / data_after_resample["dt_seconds"].iloc[1:]
    median_dt = data_after_resample["dt_seconds"].median()
    logger.info(f"Median delta time: {median_dt} seconds")
    # # to calculate frequency in Hz (samples per second), we take the inverse of median_dt so how many samples fit in
    median_freq_hz = 1 / median_dt
    logger.info(f"Estimated frequency: {median_freq_hz} Hz")
    logger.info("Resampling completed.")
    
    logger.info(f"Columns after resampling: {data_after_resample.columns.tolist()}")
    return data_after_resample
        
    
def create_sliding_windows(data_df: pd.DataFrame, window_size: int, step_size: int, sensor_cols: list[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    logger.info(f"Creating sliding windows with window size: {window_size}, step size: {step_size}")
    data_df = data_df.sort_index()
    # compute gaps between samples (in seconds)
    logger.info("Identifying session breaks based on time gaps. A session means a singular recording session without big time gaps.")
    data_df["dt"] = data_df.index.to_series().diff()
    data_df["dt_seconds"] = data_df["dt"].dt.total_seconds().fillna(0)
    
    # anything bigger than, say, 1s is a "break" between sessions
    GAP_THRESHOLD = 1.0  # 10x your normal 0.02s step

    data_df["session_break"] = data_df["dt_seconds"] > GAP_THRESHOLD
    # session_id increments every time we hit a break
    data_df["session_id"] = data_df["session_break"].cumsum().astype(int)

    print(data_df[["dt_seconds", "session_id"]].head(10))
    data_df["label"] = data_df["label"].ffill()
    logger.info(f"Number of sessions: {data_df['session_id'].nunique()}")
    

    X_windows = []   # list of (128, num_features) arrays
    y_windows = []   # list of labels
    timestamp_windows = [] 
    for sid, group in data_df.groupby("session_id"):
        # make sure there are enough samples in this session
        if len(group) < window_size:
            continue

        # ensure we use only rows with valid sensor values
        group = group.dropna(subset=sensor_cols)

        # if after dropping NaN we don't have enough, skip
        if len(group) < window_size:
            continue

        # convert to numpy for fast slicing
        values = group[sensor_cols].to_numpy()
        labels = group["label"].to_numpy()
        timestamps = group["timestamp"].to_numpy() 
        
        # sliding windows
        for start in range(0, len(group) - window_size + 1, step_size):
            end = start + window_size

            window_vals = values[start:end]
            window_labels = labels[start:end]
            window_timestamps = timestamps[start:end]

            window_label = pd.Series(window_labels).mode().iloc[0]
            window_timestamp = timestamps[start] 

            X_windows.append(window_vals)
            y_windows.append(window_label)
            timestamp_windows.append(window_timestamp)  

    X = np.stack(X_windows)      # shape: (n_windows, 128, 6)
    y = np.array(y_windows)      # shape: (n_windows,)
    ts = np.array(timestamp_windows)

    print("Number of windows:", X.shape[0])
    print("Window shape:", X.shape[1:])
    print("Example labels:", np.unique(y))
    data_df = data_df[data_df["label"] != "CUTOUT"]
    logger.info("Sliding windows created.")
    logger.info(f"Columns after creating sliding windows: {data_df.columns.tolist()}")
    return data_df, X, y, ts
    

def extract_features_from_windows(data_df: pd.DataFrame, X_array: np.ndarray, y_array: np.ndarray, timestamp_array: np.ndarray) -> pd.DataFrame:
    logger.info("Extracting features from sliding windows.")

    feature_rows = [extract_features(w) for w in X_array]
    X_features = pd.DataFrame(feature_rows)
    y_features = pd.Series(y_array)

    logger.info(f"Shape of features: {X_features.shape}")
    logger.info(f"Feature columns: {X_features.columns[:]}")
    
    extracted_features = X_features.copy()
    extracted_features["label"] = y_features.values
    extracted_features["timestamp"] = timestamp_array 
    
    logger.info("Feature extraction completed.")
    return extracted_features

def rename_features(extracted_features: pd.DataFrame) -> pd.DataFrame:
    logger.info("Renaming features for clarity.")
    renamed = extracted_features.rename(columns={"label": "Activity"})
    renamed = renamed.rename(columns={"subject": "Subject"})
    logger.info("Feature renaming completed.")
    return renamed
        

def filter_features_to_match_kaggle(renamed_features: pd.DataFrame, kaggle_csv_path: str) -> pd.DataFrame:
    logger.info(f"Filtering features to match Kaggle dataset from: {kaggle_csv_path}")
    kaggle_df = pd.read_csv(kaggle_csv_path)
    kaggle_columns = set(kaggle_df.columns)
    logger.debug(f"Kaggle dataset columns: {kaggle_columns}")
    my_columns = set(renamed_features.columns)
    filtered_columns = renamed_features.columns.intersection(kaggle_df.columns)
    filtered_features = renamed_features.reindex(columns=filtered_columns)
    filtered_features["timestamp"] = renamed_features["timestamp"]
    mask = filtered_features["timestamp"].isna()
    if mask.any():
        # Forward fill and add cumulative 20ms offsets to NaN positions
        filled_ts = filtered_features["timestamp"].ffill()
        
        # For each consecutive NaN, add 20ms, 40ms, 60ms, etc.
        consecutive_nans = mask.cumsum() - mask.cumsum().where(~mask).ffill().fillna(0)
        # Convert timestamp to int64 and add milliseconds
        filtered_features.loc[mask, "timestamp"] = (
            filled_ts[mask].astype('int64') + (consecutive_nans[mask] * 20).astype('int64')
        )
    logger.info(f"Filtered feature shape: {filtered_features.shape}")
    logger.info("Feature filtering completed.")
    return filtered_features

def save_processed_data_to_db(feature_df: pd.DataFrame, type_of_database: str = "sqlite"):
    logger.info(f"Saving processed data to {type_of_database} database.")
    # get database methods from database_utils
    database_methods = database_utils.database_methods_dict
    if type_of_database not in database_methods:
        logger.error(f"Unsupported database type: {type_of_database}")
        raise ValueError(f"Unsupported database type: {type_of_database}")
    logger.debug(f"Database methods available: {database_methods}")
    db_method = getattr(database_utils, database_methods[type_of_database])
    logger.info(f"Using database method: {db_method}")
    
    try:
        db_method(feature_df)
    except Exception as e:
        logger.error(f"Error saving data to database: {e}")
        raise
    
    logger.info("Processed data saved to database.")