import pandas as pd

def process_raw_sensor_data(raw_data_file_dir: str) -> pd.DataFrame:
    """
    Process raw sensor data files from the specified directory and return a DataFrame
    containing the extracted features and labels.
    """
    merged_data = merge_json_files(raw_data_file_dir)
    deduped_data = remove_duplicates(merged_data)
    merged_data50Hz = resample_data(deduped_data, target_freq=50)
    
    sliding_windows = create_sliding_windows(merged_data50Hz, window_size=128, step_size=64)
    
    extracted_features = extract_features_from_windows(sliding_windows)
    renamed_features = rename_features(extracted_features)
    intersect_with_kaggle = filter_features_to_match_kaggle(renamed_features)   
    
    feature_df = pd.DataFrame(intersect_with_kaggle)
    return feature_df

# send data to a database or file storage