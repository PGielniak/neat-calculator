package com.example.hellom8

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import java.io.File
import java.io.FileWriter
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.Executors
import org.json.JSONObject

class MainActivity : AppCompatActivity(), SensorEventListener {

    // UI Elements
    private lateinit var recordButton: Button
    private lateinit var statusText: TextView
    private lateinit var sensorDataText: TextView
    
    // Sensors
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    
    // Execution
    private val saveExecutor = Executors.newSingleThreadExecutor()
    private val ntpExecutor = Executors.newSingleThreadExecutor()
    
    // Data & State
    private val sensorDataList = mutableListOf<SensorData>()
    private var isRecording = false
    private var recordingStartTime = 0L
    private var currentRecordingTimestamp = ""
    private var recordingSegmentNumber = 0
    
    // NTP
    private var ntpOffset = 0L
    private var isNtpSynced = false

    // Auto-save
    private val autoSaveHandler = Handler(Looper.getMainLooper())
    private val AUTO_SAVE_INTERVAL = 60000L // 1 minute
    
    private val autoSaveRunnable = object : Runnable {
        override fun run() {
            if (isRecording) {
                // Save current segment and continue
                saveSensorDataSegment()
                recordingSegmentNumber++
                autoSaveHandler.postDelayed(this, AUTO_SAVE_INTERVAL)
            }
        }
    }
    
    // Sensor values
    private var lastAccelX = 0f
    private var lastAccelY = 0f
    private var lastAccelZ = 0f
    private var lastGyroX = 0f
    private var lastGyroY = 0f
    private var lastGyroZ = 0f
    
    private var lastSensorSampleTime = 0L
    private val SAMPLE_RATE_HZ = 50 
    private val SAMPLE_INTERVAL_MS = 1000L / SAMPLE_RATE_HZ 
    
    private val REQUIRED_PERMISSIONS = if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
        arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE)
    } else {
        emptyArray()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        recordButton = findViewById(R.id.recordButton)
        statusText = findViewById(R.id.statusText)
        sensorDataText = findViewById(R.id.sensorDataText)
        
        statusText.text = "Initializing sensors..."
        
        // Initialize sensors
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        
        // Sync NTP
        syncNtpTime()
        
        if (allPermissionsGranted()) {
            registerSensors()
            statusText.text = "Ready (Syncing NTP...)"
            recordButton.isEnabled = true
        } else {
            if (REQUIRED_PERMISSIONS.isNotEmpty()) {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
            } else {
                registerSensors()
                statusText.text = "Ready (Syncing NTP...)"
                recordButton.isEnabled = true
            }
        }
        
        recordButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecording()
            }
        }
        
        recordButton.text = "Start Recording"
    }

    private fun syncNtpTime() {
        ntpExecutor.execute {
            val providers = listOf(
                TimeProvider(
                    "TimeAPI.io",
                    "https://timeapi.io/api/Time/current/zone?timeZone=UTC"
                ) { response ->
                    val json = JSONObject(response)
                    val dateTime = json.getString("dateTime")
                    // Parse ISO format using SimpleDateFormat (compatible with older Android)
                    val formatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSS", Locale.US)
                    formatter.timeZone = TimeZone.getTimeZone("UTC")
                    formatter.parse(dateTime)?.time ?: System.currentTimeMillis()
                },
                TimeProvider(
                    "WorldTimeAPI",
                    "https://worldtimeapi.org/api/timezone/UTC"
                ) { response ->
                    val json = JSONObject(response)
                    val unixtime = json.getLong("unixtime")
                    val datetime = json.getString("datetime")
                    val millis = if (datetime.contains(".")) {
                        val fractional = datetime.substringAfter(".").substringBefore("+").take(3)
                        fractional.padEnd(3, '0').toInt()
                    } else 0
                    unixtime * 1000 + millis
                }
            )

            for (provider in providers) {
                try {
                    val startTime = System.currentTimeMillis()
                    val connection = URL(provider.url).openConnection() as HttpURLConnection
                    connection.requestMethod = "GET"
                    connection.connectTimeout = 5000
                    connection.readTimeout = 5000
                    
                    val responseCode = connection.responseCode
                    if (responseCode == 200) {
                        val response = connection.inputStream.bufferedReader().use { it.readText() }
                        val endTime = System.currentTimeMillis()
                        
                        val serverTimeMs = provider.parseResponse(response)
                        val latency = (endTime - startTime) / 2
                        val localTime = System.currentTimeMillis()
                        
                        ntpOffset = (serverTimeMs + latency) - localTime
                        isNtpSynced = true
                        
                        runOnUiThread {
                            val offsetSec = ntpOffset / 1000.0
                            val formattedOffset = "%.3f".format(offsetSec)
                            Toast.makeText(this@MainActivity, "Synced via ${provider.name}. Offset: ${formattedOffset}s", Toast.LENGTH_SHORT).show()
                            if (!isRecording) statusText.text = "Ready (NTP Synced)"
                        }
                        return@execute
                    }
                } catch (e: Exception) {
                    // Continue to next provider
                }
            }
            
            // All providers failed
            runOnUiThread {
                Toast.makeText(this@MainActivity, "All NTP providers failed. Using system time.", Toast.LENGTH_SHORT).show()
                if (!isRecording) statusText.text = "Ready (No NTP)"
            }
        }
    }

    // Helper to get corrected time
    private val currentTimeMillis: Long
        get() = System.currentTimeMillis() + ntpOffset
    
    private fun registerSensors() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }
    
    private fun startRecording() {
        isRecording = true
        sensorDataList.clear()
        
        val timestamp = currentTimeMillis.toString()
        currentRecordingTimestamp = timestamp
        recordingStartTime = currentTimeMillis
        recordingSegmentNumber = 0
        
        autoSaveHandler.postDelayed(autoSaveRunnable, AUTO_SAVE_INTERVAL)
        
        recordButton.text = "Stop Recording"
        statusText.text = "Recording... (Auto-save enabled)"
        
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
    
    private fun stopRecording() {
        isRecording = false
        autoSaveHandler.removeCallbacks(autoSaveRunnable)
        
        val finalData = sensorDataList.toList()
        saveExecutor.execute {
            saveSensorData(currentRecordingTimestamp, finalData, isFinal = true)
        }
        
        recordButton.text = "Start Recording"
        statusText.text = "Stopped. Saving..."
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
    
    private fun saveSensorDataSegment() {
        if (sensorDataList.isEmpty()) return
        
        val dataToSave = sensorDataList.toList()
        sensorDataList.clear()
        
        val timestamp = "${currentRecordingTimestamp}_seg${recordingSegmentNumber}"
        
        saveExecutor.execute {
            saveSensorData(timestamp, dataToSave, isFinal = false)
        }
    }
    
    private fun saveSensorData(baseName: String, data: List<SensorData>, isFinal: Boolean) {
        try {
            val fileName = "$baseName.json"
            val gson = Gson()
            val jsonData = gson.toJson(data)
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val contentValues = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "application/json")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS + "/SensorRecording")
                }
                
                val uri = contentResolver.insert(MediaStore.Files.getContentUri("external"), contentValues)
                uri?.let {
                    contentResolver.openOutputStream(it)?.use { outputStream ->
                        outputStream.write(jsonData.toByteArray())
                    }
                }
            } else {
                val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "SensorRecording")
                if (!dir.exists()) dir.mkdirs()
                val file = File(dir, fileName)
                FileWriter(file).use { writer ->
                    writer.write(jsonData)
                }
            }
            
            runOnUiThread {
                val msg = if (isFinal) "Saved final: ${data.size} samples" else "Auto-saved seg $recordingSegmentNumber"
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
                if (isFinal) statusText.text = "Ready (NTP Synced)"
            }
        } catch (e: Exception) {
            runOnUiThread {
                Toast.makeText(this, "Error saving: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            val currentTime = currentTimeMillis
            val nanoTime = event.timestamp
            
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    lastAccelX = event.values[0]
                    lastAccelY = event.values[1]
                    lastAccelZ = event.values[2]
                }
                Sensor.TYPE_GYROSCOPE -> {
                    lastGyroX = event.values[0]
                    lastGyroY = event.values[1]
                    lastGyroZ = event.values[2]
                }
            }
            
            runOnUiThread {
                sensorDataText.text = "Accel: [%.2f, %.2f, %.2f]\nGyro: [%.2f, %.2f, %.2f]\nNTP: ${if(isNtpSynced) "Yes" else "No"} ($ntpOffset ms)".format(
                    lastAccelX, lastAccelY, lastAccelZ,
                    lastGyroX, lastGyroY, lastGyroZ
                )
            }
            
            if (isRecording) {
                if (currentTime - lastSensorSampleTime >= SAMPLE_INTERVAL_MS) {
                    val sensorData = SensorData(
                        timestamp = currentTime,
                        timestampNanos = nanoTime,
                        accelerometerX = lastAccelX,
                        accelerometerY = lastAccelY,
                        accelerometerZ = lastAccelZ,
                        gyroscopeX = lastGyroX,
                        gyroscopeY = lastGyroY,
                        gyroscopeZ = lastGyroZ
                    )
                    sensorDataList.add(sensorData)
                    lastSensorSampleTime = currentTime
                }
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    
    private fun allPermissionsGranted() = (REQUIRED_PERMISSIONS.isEmpty() || REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    })
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                registerSensors()
                recordButton.isEnabled = true
            } else {
                Toast.makeText(this, "Permissions not granted: " + permissions.joinToString(), Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        autoSaveHandler.removeCallbacks(autoSaveRunnable)
        sensorManager.unregisterListener(this)
        saveExecutor.shutdown()
        ntpExecutor.shutdown()
    }
    
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
    
    // Time provider data class
    private data class TimeProvider(
        val name: String,
        val url: String,
        val parseResponse: (String) -> Long
    )
}
