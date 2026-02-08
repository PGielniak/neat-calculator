import os
from cv2 import merge
import pandas as pd
import json

def merge_csv_files(csv_dir):
    merged_df = pd.DataFrame()
    files_sorted = sorted(os.listdir(csv_dir))
    for filename in files_sorted:
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_dir, filename))
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df

labels_dir = os.getcwd()
labels_merged_df = merge_csv_files(labels_dir)

labels_merged_df.to_csv('merged_labels.csv', index=False)