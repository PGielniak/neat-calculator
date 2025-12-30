
import argparse
import os
import pandas as pd

argparser = argparse.ArgumentParser(description="Adding Labels to dataset")

argparser.add_argument(
    "--data_file_path", required=True, help="Path to the json file with results.", default=None)

args = argparser.parse_args()

def label_data(data_file_path):
    labels_list = []
    print(f"Labeling data in file: {data_file_path}")
    # Add your labeling logic here
    data = pd.read_json(data_file_path)
    while True:

        print("Please provide the start timestamp for the event (in ms): ")
        start_timestamp = input().strip()
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
        print("Press 'q' to quit labeling. Press any key to continue labeling.")
        if input().strip().lower() == 'q':
            break
        
    print("Labels provided: ", labels_list)
        
    
        
label_data(args.data_file_path)