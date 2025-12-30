
import argparse
import os
import pandas as pd

argparser = argparse.ArgumentParser(description="Adding Labels to dataset")

argparser.add_argument(
    "--data_file_path", required=False, help="Path to the json file with results.", default='E:\\src\\neat-calculator\\data_collection\\recordings\\sensor_data\\1767099292263.json')

args = argparser.parse_args()

def label_data(data_file_path):
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
    
    for i in range(len(labels_list)):
        print(f"Processing label {i+1} of {len(labels_list)}")
        
        print(f"Label: {labels_list[i][1]}, Start Timestamp: {labels_list[i][0]}")
        
        # Finding nearest index for start timestamp
        start_idx = (copy_data['timestamp'] - int(labels_list[i][0])).abs().idxmin()
        print(f"Start Index found: {start_idx}")
        
        copy_data.loc[start_idx:, 'label'] = int(labels_list[i][1])
        
    output_file_path = data_file_path.replace('.json', '_labeled.json')
    copy_data.to_json(output_file_path, orient='records', lines=False)
    print(f"Labeled data saved to: {output_file_path}")
    
label_data(args.data_file_path)