[app]

# (1) App details
title = KalyanMLApp
package.name = kalyanml
package.domain = org.kalyan.ml
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,ttf,txt,md,csv,pkl,h5

# (2) Versioning
version = 1.0
requirements = python3,kivy,pandas,numpy,scikit-learn,tensorflow,joblib

# (3) Orientation & Icon (optional)
orientation = portrait
icon.filename = %(source.dir)s/icon.png
presplash.filename = %(source.dir)s/presplash.png

# (4) Permissions
android.permissions = WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# (5) Android NDK/SDK/API
android.ndk = 25b
android.api = 31
android.minapi = 21
android.ndk_api = 21
android.sdk = 31
android.ndk_path = $HOME/.buildozer/android/platform/android-ndk-r25b
android.sdk_path = $HOME/.buildozer/android/platform/android-sdk
android.archs = arm64-v8a, armeabi-v7a

# (6) Entry point
entrypoint = main.py

# (7) Asset inclusion
android.add_assets = data.csv:., model.h5:., scaler.pkl:., label_encoder.pkl:.

# (8) Logging and debugging
log_level = 2
android.logcat_filters = *:S python:D

# (9) Misc
fullscreen = 0
use_kivy_settings = 0
android.allow_backup = 1
android.additional_packages = 

# (10) Build directory
build_dir = ./.buildozer

# (11) Package format
android.packaging = default
copy_libs = 1

# (12) Don't change
android.entrypoint = org.kivy.android.PythonActivity
android.private_storage = True

# (13) Include any .so or custom modules if needed
# android.add_libs_armeabi-v7a =
# android.add_libs_arm64-v8a =

# (14) Environment variables (optional)
# p4a.local_recipes =
# p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1
