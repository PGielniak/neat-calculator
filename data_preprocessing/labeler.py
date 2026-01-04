
import argparse
import os
import pandas as pd

argparser = argparse.ArgumentParser(description="Adding Labels to dataset")

argparser.add_argument(
    "--data_file_path", required=False, help="Path to the json file with results.", default='E:\\src\\neat-calculator\\data_collection\\recordings\\sensor_data\\1767099292263.json')

argparser.add_argument(
    "--labels_csv_path", required=False, help="Path to the csv file with labels.", default=None)

argparser.add_argument(
    "--manual", action='store_true', help="Flag to indicate manual labeling mode.", default=False)

args = argparser.parse_args()


def label_data_manually(data_file_path):
    labels_list = []
    print(f"Labeling data in file: {data_file_path}")
    print("Instructions:")
    print("Enter the start timestamp for each event and the corresponding label.")
    print("Insert the timestamps chronologically")
    print("The script will find the nearest timestamp in the data for labeling.")
    print("After labeling the nearest entries, the following entries will be labeled until the next start timestamp.")
    # Add your labeling logic here
    data = pd.read_json(data_file_path, convert_dates=False)
    
    copy_data = data.copy()
    copy_data.sort_values(by=['timestamp'], inplace=True)
    while True:

        print("Please provide the start timestamp for the event (in ms): ")
        print("Or press 'q' to quit labeling.")

        start_timestamp = input().strip()
        if start_timestamp.lower() == 'q':
            break
        if start_timestamp == '':
            continue
        print(f"""
                Please provide one of the labels
                0 - WALKING
                1 - STANDING
                2 - SITTING
                3 - LAYING
                4 - WALKING_UPSTAIRS
                5 - WALKING_DOWNSTAIRS
                """)
        
        label = input().strip()
        
        labels_list.append((start_timestamp, label))

        
    print("Labels provided: ", labels_list)
    
    labels_df = pd.DataFrame(labels_list, columns=['timestamp', 'label'])
    
    _label_data(labels_df, copy_data, data_file_path)
    
def label_data_from_file(data_file_path, labels_csv_path):
    print(f"Labeling data in file: {data_file_path}")
    print(f"Using labels from CSV file: {labels_csv_path}")
    
    labels_df= pd.read_csv(labels_csv_path)
    data = pd.read_json(data_file_path, convert_dates=False)
    copy_data = data.copy()
    
    copy_data.sort_values(by=['timestamp'], inplace=True)
    _label_data(labels_df, copy_data, data_file_path)
    

        
def _label_data(labels_df: pd.DataFrame, copy_data: pd.DataFrame, data_file_path: str):
    for index, row in labels_df.iterrows():
        start_idx = (copy_data['timestamp'] - int(row['timestamp'])).abs().idxmin()
        print(f"Start Index found: {start_idx}")
        copy_data.loc[start_idx:, 'label'] = row['label']
    output_file_path = data_file_path.replace('.json', '_labeled.json')
    copy_data.to_json(output_file_path, orient='records', lines=False)
    print(f"Labeled data saved to: {output_file_path}")
        
        
if args.manual:
    label_data_manually(args.data_file_path)
if not args.manual and args.labels_csv_path is not None:
    label_data_from_file(args.data_file_path, args.labels_csv_path)