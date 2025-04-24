[app]
title = KalyanX
package.name = ksm
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0
requirements = python3,kivy, requist,numpy
orientation = portrait
fullscreen = 1
android.api = 33
android.minapi = 21
android.ndk = 23b
android.ndk_path = $HOME/.buildozer/android/platform/android-ndk-r23b
android.sdk_path = $HOME/.buildozer/android/platform/android-sdk
android.gradle_dependencies = 
android.permissions = INTERNET
android.archs = armeabi-v7a, arm64-v8a
android.build_tools_version = 34.0.0
android.sdk = 33
android.ndk_api = 21
android.aidl = true

# Optional: include this if you want the .apk file to be signed and installable
# (you can disable debug and enable release later if you get a signing key)
android.debug = 1

[buildozer]
log_level = 2
warn_on_root = 1
