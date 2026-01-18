import argparse
import os
import json

argparser = argparse.ArgumentParser(description="Overlay timestamp on a video.")

argparser.add_argument(
    "--sensor_data_dir", required=False, help="Path to the input sensor data directory.", default="E:\\src\\neat-calculator\\data_collection\\recordings\\sensor_data\\SensorRecording")

argparser.add_argument(
    "--output_file", required=False, help="Path to save the merged output file", default="E:\\src\\neat-calculator\\data_preprocessing\\merged_sensor_data.json")

args = argparser.parse_args()

def merge_sensor_data(sensor_datadir, output_file):
    sensor_data_files = os.listdir(sensor_datadir)
    sensor_data_files.sort()
    merged_data = json.loads('[]')
    for file_name in sensor_data_files:
        file_path = os.path.join(sensor_datadir, file_name)
        print(f"Processing sensor data file: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)
            
    print(f"Saving merged data to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)   
        
        

merge_sensor_data(sensor_datadir=args.sensor_data_dir, output_file=args.output_file)