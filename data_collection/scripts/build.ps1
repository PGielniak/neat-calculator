$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.17.10-hotspot";

cd e:\src\neat-calculator\data_collection\app; .\gradlew.bat assembleDebug

adb install -r "E:\src\neat-calculator\data_collection\app\src\build\outputs\apk\debug\app-debug.apk"