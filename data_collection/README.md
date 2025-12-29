# NEAT Data Collection App

## Overview

This Android application is designed to collect synchronized video and sensor data for machine learning research focused on identifying non-exercise activity thermogenesis (NEAT). The app records video alongside high-frequency accelerometer and gyroscope data to enable analysis and classification of everyday activities.

## Purpose

NEAT (Non-Exercise Activity Thermogenesis) represents the energy expenditure from all physical activities outside of formal exercise, sleeping, and eating. This includes activities like:
- Standing
- Sitting
- Walking
- Fidgeting
- Maintaining posture
- Other daily movements

This data collection tool enables researchers and developers to gather labeled training data for machine learning models that can automatically detect and classify these activities, ultimately helping to:
- Track daily energy expenditure more accurately
- Understand movement patterns throughout the day
- Develop personalized health and fitness insights
- Research the impact of NEAT on overall metabolic health

## Features

### Video Recording
- **HD Video Capture**: Records high-quality video using the device's camera
- **Timestamp Overlay**: Displays Unix timestamp (milliseconds) on each video frame for precise synchronization
- **Audio Recording**: Captures audio alongside video (optional, requires permission)
- **Automatic Saving**: Videos are saved to the device's storage with timestamped filenames

### Sensor Data Collection
- **Accelerometer Data**: Captures 3-axis acceleration data (X, Y, Z) at maximum sampling rate
- **Gyroscope Data**: Captures 3-axis rotational velocity data (X, Y, Z) at maximum sampling rate
- **High-Frequency Sampling**: Uses `SENSOR_DELAY_FASTEST` for maximum data resolution
- **Precise Timestamps**: Each sensor reading includes:
  - `timestamp`: System time in Unix milliseconds
  - `timestampNanos`: High-precision sensor event timestamp in nanoseconds

### Data Storage
- **Synchronized Format**: Video and sensor data share the same timestamp format for easy correlation
- **JSON Export**: Sensor data is saved as JSON files for easy parsing and analysis
- **Organized Structure**: Files are saved with matching timestamps for pairing
  - Videos: `video_YYYYMMDD_HHMMSS.3gp` (or .mp4 on newer Android versions)
  - Sensor data: `sensor_data_YYYYMMDD_HHMMSS.json`

### User Interface
- **Live Camera Preview**: Real-time view of what's being recorded
- **Sensor Data Display**: Live display of current accelerometer and gyroscope values
- **Recording Status**: Clear indication of recording state
- **Simple Controls**: Single button to start/stop recording

## Requirements

- **Android Version**: Android 5.0 (API 21) or higher
- **Hardware**:
  - Camera
  - Accelerometer sensor
  - Gyroscope sensor
- **Permissions**:
  - Camera access
  - Audio recording
  - Storage access (for saving files)

## Data Format

### Video Files
Videos are saved with an embedded timestamp overlay showing the Unix timestamp in milliseconds. This allows frame-by-frame correlation with sensor data.

### Sensor Data Files
Sensor data is saved as JSON arrays with the following structure:

```json
[
  {
    "timestamp": 1766951672616,
    "timestampNanos": 335447417916193,
    "accelerometerX": -0.23,
    "accelerometerY": 9.81,
    "accelerometerZ": 0.15,
    "gyroscopeX": 0.01,
    "gyroscopeY": -0.02,
    "gyroscopeZ": 0.00
  },
  ...
]
```

**Field Descriptions:**
- `timestamp`: Unix timestamp in milliseconds (matches video overlay)
- `timestampNanos`: High-precision sensor timestamp in nanoseconds
- `accelerometerX/Y/Z`: Acceleration in m/s² along each axis
- `gyroscopeX/Y/Z`: Angular velocity in rad/s around each axis

## Usage

1. **Launch the App**: Open the NEAT Data Collection app
2. **Grant Permissions**: Allow camera, audio, and storage permissions when prompted
3. **Position Device**: Mount or hold the device to capture the desired activity
4. **Start Recording**: Tap "Start Recording" button
5. **Perform Activity**: Execute the activity you want to record (sitting, standing, walking, etc.)
6. **Stop Recording**: Tap "Stop Recording" when finished
7. **Verify**: Check that both video and sensor data files were saved successfully

## Data Retrieval

### From Device Storage
- Videos: `Movies/SensorRecording/` or `/sdcard/video/`
- Sensor Data: `Documents/SensorRecording/`

### Using ADB (Android Debug Bridge)
```powershell
# Pull video files
adb pull /sdcard/video/ ./recordings/videos/

# Pull sensor data files
adb pull /sdcard/Documents/SensorRecording/ ./recordings/
```

## Technical Details

### Architecture
- **Language**: Kotlin
- **Camera**: CameraX library for modern camera API
- **Sensors**: Android SensorManager with fastest sampling rate
- **UI**: AndroidX with ConstraintLayout
- **Data Serialization**: Gson for JSON export

### Libraries Used
- CameraX (camera-core, camera-camera2, camera-lifecycle, camera-video, camera-view)
- AndroidX Core, AppCompat, Material Design
- ConstraintLayout
- Gson for JSON serialization

### Storage Strategy
- **Android 10+**: Uses MediaStore API for scoped storage compliance
- **Android 9 and below**: Direct file system access to public directories
- **Automatic Directory Creation**: Creates output directories if they don't exist

## Machine Learning Applications

This data can be used to train models for:
- **Activity Classification**: Identify whether someone is sitting, standing, walking, etc.
- **Movement Quality Assessment**: Analyze posture and movement patterns
- **Energy Expenditure Estimation**: Calculate NEAT contributions to daily calorie burn
- **Temporal Pattern Recognition**: Detect transitions between activities
- **Personalized Activity Detection**: Adapt models to individual movement patterns

## Development

### Building from Source
```powershell
cd data_collection/app
.\gradlew.bat assembleDebug
```

### Installing
```powershell
adb install -r app/src/build/outputs/apk/debug/app-debug.apk
```

## Future Enhancements

Potential improvements for future versions:
- Activity labeling interface for ground truth annotation
- Real-time activity prediction using on-device ML
- Bluetooth heart rate monitor integration
- GPS location tracking for outdoor activities
- Export data directly to cloud storage
- Batch recording with automatic segmentation
- Support for additional sensors (magnetometer, pressure, etc.)

## License

This project is designed for research and educational purposes related to NEAT and activity recognition.

## Credits

Developed for NEAT (Non-Exercise Activity Thermogenesis) research and machine learning applications in health and fitness tracking.
