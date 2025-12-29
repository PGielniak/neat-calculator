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
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var previewView: PreviewView
    private lateinit var recordButton: Button
    private lateinit var statusText: TextView
    private lateinit var sensorDataText: TextView
    private lateinit var timestampOverlay: TimestampOverlay
    
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private lateinit var cameraExecutor: ExecutorService
    
    private val sensorDataList = mutableListOf<SensorData>()
    private var isRecording = false
    private var recordingStartTime = 0L
    
    private val timestampHandler = Handler(Looper.getMainLooper())
    private val timestampUpdateRunnable = object : Runnable {
        override fun run() {
            if (isRecording) {
                val currentTimestamp = System.currentTimeMillis()
                timestampOverlay.updateTimestamp(currentTimestamp)
                timestampHandler.postDelayed(this, 16) // ~60fps
            }
        }
    }
    
    private var lastAccelX = 0f
    private var lastAccelY = 0f
    private var lastAccelZ = 0f
    private var lastGyroX = 0f
    private var lastGyroY = 0f
    private var lastGyroZ = 0f
    
    private val REQUIRED_PERMISSIONS = mutableListOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    ).apply {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
    }.toTypedArray()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        previewView = findViewById(R.id.previewView)
        recordButton = findViewById(R.id.recordButton)
        statusText = findViewById(R.id.statusText)
        sensorDataText = findViewById(R.id.sensorDataText)
        timestampOverlay = findViewById(R.id.timestampOverlay)
        
        // Initialize sensors
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Request permissions
        if (allPermissionsGranted()) {
            startCamera()
            registerSensors()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        
        recordButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecording()
            }
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.HD))
                .build()
            videoCapture = VideoCapture.withOutput(recorder)
            
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, videoCapture
                )
            } catch (e: Exception) {
                Toast.makeText(this, "Camera binding failed: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun registerSensors() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        }
    }
    
    private fun startRecording() {
        val videoCapture = this.videoCapture ?: return
        
        recordButton.isEnabled = false
        sensorDataList.clear()
        
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val videoFileName = "video_$timestamp.mp4"
        
        val contentValues = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, videoFileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/SensorRecording")
            }
        }
        
        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()
        
        recording = videoCapture.output
            .prepareRecording(this, mediaStoreOutputOptions)
            .apply {
                if (ActivityCompat.checkSelfPermission(
                        this@MainActivity,
                        Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    withAudioEnabled()
                }
            }
            .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                when (recordEvent) {
                    is VideoRecordEvent.Start -> {
                        isRecording = true
                        recordingStartTime = System.currentTimeMillis()
                        timestampHandler.post(timestampUpdateRunnable)
                        runOnUiThread {
                            recordButton.apply {
                                text = "Stop Recording"
                                isEnabled = true
                            }
                            statusText.text = "Recording..."
                        }
                    }
                    is VideoRecordEvent.Finalize -> {
                        if (!recordEvent.hasError()) {
                            val msg = "Video saved: ${recordEvent.outputResults.outputUri}"
                            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                            saveSensorData(timestamp)
                        } else {
                            recording?.close()
                            recording = null
                            Toast.makeText(
                                this,
                                "Video recording error: ${recordEvent.error}",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                        isRecording = false
                        timestampHandler.removeCallbacks(timestampUpdateRunnable)
                        runOnUiThread {
                            recordButton.apply {
                                text = "Start Recording"
                                isEnabled = true
                            }
                            statusText.text = "Ready to record"
                        }
                    }
                }
            }
    }
    
    private fun stopRecording() {
        recording?.stop()
        recording = null
    }
    
    private fun saveSensorData(timestamp: String) {
        try {
            val fileName = "sensor_data_$timestamp.json"
            val gson = Gson()
            val jsonData = gson.toJson(sensorDataList)
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Android 10+ - use MediaStore
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
                    runOnUiThread {
                        Toast.makeText(
                            this,
                            "Sensor data saved: ${sensorDataList.size} samples",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            } else {
                // Android 9 and below - use external storage
                val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "SensorRecording")
                if (!dir.exists()) {
                    dir.mkdirs()
                }
                val file = File(dir, fileName)
                FileWriter(file).use { writer ->
                    writer.write(jsonData)
                }
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Sensor data saved: ${sensorDataList.size} samples\n${file.absolutePath}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        } catch (e: Exception) {
            runOnUiThread {
                Toast.makeText(this, "Error saving sensor data: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            val currentTime = System.currentTimeMillis()
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
            
            // Update UI
            runOnUiThread {
                sensorDataText.text = "Accel: [%.2f, %.2f, %.2f]\nGyro: [%.2f, %.2f, %.2f]".format(
                    lastAccelX, lastAccelY, lastAccelZ,
                    lastGyroX, lastGyroY, lastGyroZ
                )
            }
            
            // Record data if recording
            if (isRecording) {
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
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed for this implementation
    }
    
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
                registerSensors()
            } else {
                Toast.makeText(this, "Permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        timestampHandler.removeCallbacks(timestampUpdateRunnable)
        sensorManager.unregisterListener(this)
        cameraExecutor.shutdown()
    }
    
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
