from .video_with_overlay import overlay_timestamp_on_video
import os
list_of_files = os.listdir('data_collection\\recordings\\videos\\SensorRecording')
list_of_files.sort()


for file_name in list_of_files:
    print(f"Processing file: {file_name}")
    input_video_path = f"data_collection\\recordings\\videos\\SensorRecording\\{file_name}"
    output_video_path = f"data_collection\\recordings\\videos\\SensorRecording_Timestamped\\{file_name}"
    print(len(file_name.split('_')))
    if len(file_name.split('_')) < 2:
        print(f"Not a segment file, starting timestamp from the file name and saving the end timestamp")
        start_time = int(file_name.split('.')[0])
        
        output_video, next_segment_start_time = overlay_timestamp_on_video(input_video_path, output_video_path, str(start_time))
        print(f"Next segment should start at: {next_segment_start_time}")
        
    else:
        if 'next_segment_start_time' not in locals():
            print("Error: next_segment_start_time is not defined. Skipping segment file.")
            continue
        start_time = next_segment_start_time
        print(f"Segment file, using previous end timestamp as starting timestamp: {start_time}")
        
        output_video, next_segment_start_time = overlay_timestamp_on_video(input_video_path, output_video_path, str(start_time))