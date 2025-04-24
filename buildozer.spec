[app]
title = Kalyan Jodi Prediction
package.name = kalyanjodi
package.domain = org.kalyan.prediction
source.dir = .
source.include_exts = py,kv,png,jpg,kv,pkl,h5,csv
version = 1.0
requirements = python3,kivy,kivymd,tensorflow,scikit-learn,pandas,numpy,joblib
orientation = portrait
fullscreen = 0

[buildozer]
log_level = 2
warn_on_root = 1

[android]
android.api = 31
android.minapi = 23
android.ndk = 25b
android.ndk_path = 
android.gradle_dependencies = 
android.permissions = INTERNET
android.archs = armeabi-v7a, arm64-v8a
