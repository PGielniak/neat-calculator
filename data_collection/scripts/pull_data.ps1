
# Pull recorded videos and sensor data from the connected Android device
adb pull /sdcard/DCIM/SensorRecording/ ./recordings/videos/
adb pull /sdcard/Documents/SensorRecording/ ./recordings/sensor_data/

