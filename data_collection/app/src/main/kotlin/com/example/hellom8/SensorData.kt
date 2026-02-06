package com.example.hellom8

data class SensorData(
    val timestamp: Long,
    val timestampNanos: Long,
    val accelerometerX: Float,
    val accelerometerY: Float,
    val accelerometerZ: Float,
    val gyroscopeX: Float,
    val gyroscopeY: Float,
    val gyroscopeZ: Float
)

data class RecordingSession(
    val videoFileName: String,
    val dataFileName: String,
    val startTime: Long,
    val endTime: Long,
    val sensorDataCount: Int
)
