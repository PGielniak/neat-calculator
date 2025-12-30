import cv2 as cv
import argparse
import os

argparser = argparse.ArgumentParser(description="Overlay timestamp on a video.")

argparser.add_argument(
    "--input_video", required=False, help="Path to the input video file.", default="E:\\src\\neat-calculator\\data_collection\\recordings\\videos\\SensorRecording\\1767099292263.mp4")

argparser.add_argument(
    "--output_video", required=False, help="Path to save the output video file with overlay", default=None)

args = argparser.parse_args()

def overlay_timestamp_on_video(input_video_path, output_video_path=None):
    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = f"{base}_with_timestamp{ext}"
        
    capture = cv.VideoCapture(input_video_path)
    file_name = os.path.basename(input_video_path)
    timestamp = file_name.split('.')[0]
    print(f"Overlaying timestamp: {timestamp} on video: {input_video_path}")
    
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    fps = capture.get(cv.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    
    # Get video properties
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    timestamp_increment = int(1000 / fps)
    
    while True:
        isTrue, frame = capture.read()

        if not isTrue:
            key = cv.waitKey(0)
            if key == ord('d'):
                break
            continue
        
        timestamp_in_frame = f"Timestamp: {timestamp}"
        cv.putText(frame, timestamp_in_frame, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv.LINE_AA)
        
        # Write frame to output video
        out.write(frame)
        
        cv.imshow('Video', frame)
        
        timestamp = str(int(timestamp) + timestamp_increment)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    
    capture.release()
    out.release()
    cv.destroyAllWindows()
    print(f"Video saved to: {output_video_path}")

overlay_timestamp_on_video(args.input_video, args.output_video)