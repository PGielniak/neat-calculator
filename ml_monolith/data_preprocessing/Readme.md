# Data Preprocessing Pipeline

This directory contains scripts for preprocessing raw sensor data collected from mobile devices for human activity recognition (HAR) analysis.

## Overview

The preprocessing pipeline transforms raw sensor recordings into labeled datasets suitable for machine learning model training. The pipeline consists of video annotation, data consolidation, and activity labeling stages.

## Pipeline Steps

### 1. Video Timestamp Overlay
**Script:** `add_timestamp_to_video.py`

Adds millisecond-precision timestamps to recorded videos to facilitate accurate activity labeling. Processes all videos in the recordings directory and outputs timestamped versions.

```bash
python -m data_preprocessing.add_timestamp_to_video
```

**Output:** Timestamped videos saved to `SensorRecording_Timestamped/` folder

### 2. Activity Label Generation
**File:** `labels.csv`

Manual creation of activity labels by reviewing timestamped videos and recording:
- Start timestamp (milliseconds)
- Activity label (WALKING, STANDING, SITTING, LAYING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS)

**Format:**
```csv
timestamp,label
1767184584055,WALKING
1767184590000,STANDING
1767184600000,SITTING
```

### 3. Sensor Data Consolidation
**Script:** `merge_sensor_data.py`

Combines segmented sensor data files from individual recording sessions into a single unified JSON file containing all accelerometer and gyroscope readings.

```bash
python -m data_preprocessing.merge_sensor_data
```

**Input:** Segmented JSON files from `data_collection/recordings/sensor_data/SensorRecording/`  
**Output:** `merged_sensor_data.json`

### 4. Activity Labeling
**Script:** `labeler.py`

Maps activity labels from `labels.csv` to corresponding sensor data records based on timestamp alignment. Labels are applied from the nearest timestamp forward until the next activity change.

```bash
python -m data_preprocessing.labeler \
  --data_file_path "merged_sensor_data.json" \
  --labels_csv_path "labels.csv"
```

**Input:** 
- `merged_sensor_data.json` (consolidated sensor readings)
- `labels.csv` (activity labels with timestamps)

**Output:** `merged_sensor_data_labeled.json`

## Scripts

### `video_with_overlay.py`
Core functionality for overlaying timestamps on video frames.

### `merge_sensor_data.py`
Consolidates segmented sensor recordings into a single dataset.

### `labeler.py`
Applies activity labels to sensor data based on timestamp matching.

**Modes:**
- Manual labeling: `--manual` flag for interactive label entry
- File-based labeling: `--labels_csv_path` for batch labeling from CSV

## Data Format

**Raw Sensor Data:**
```json
[
  {
    "accelerometerX": -0.123,
    "accelerometerY": 9.812,
    "accelerometerZ": 0.045,
    "gyroscopeX": 0.002,
    "gyroscopeY": -0.001,
    "gyroscopeZ": 0.003,
    "timestamp": 1767184584055,
    "timestampNanos": 1767184584055000000
  }
]
```

**Labeled Output:**
```json
[
  {
    "accelerometerX": -0.123,
    "accelerometerY": 9.812,
    "accelerometerZ": 0.045,
    "gyroscopeX": 0.002,
    "gyroscopeY": -0.001,
    "gyroscopeZ": 0.003,
    "timestamp": 1767184584055,
    "timestampNanos": 1767184584055000000,
    "label": "WALKING"
  }
]
```

## Activity Labels

Supported activity classifications:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `STANDING`
- `SITTING`
- `LAYING`

## Requirements

- Python 3.8+
- pandas
- OpenCV (cv2)
- numpy

