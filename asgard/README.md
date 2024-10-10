##### Build Instructions
```bash
# Go to tensorflow root directory
cd tensorflow

# Create .tf_configure.bazelrc if it's not been set up
vim .tf_configure.bazelrc

# Configure .tf_configure.bazelrc. Only Android NDK versions 17-20 will work.
build --action_env ANDROID_NDK_HOME="/your/path/to/android-ndk-r20b"
build --action_env ANDROID_NDK_API_LEVEL="20"

# NOTE: Android NDK 20 could be downloaded like this
wget https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
unzip android-ndk-r20b-linux-x86_64.zip

# Build apps
# NOTE: Make sure you have Bazel installed (guide available below)
bazel build -c opt --config=android_arm64 --build_tag_filters=-no_android //asgard/...
bazel build -c opt --config=elinux_aarch64 //asgard:guest_inference_minimal

# Check output
ls -al bazel-bin/asgard
```
- Install Bazel by following this [guide](https://bazel.build/install/ubuntu#install-on-ubuntu).
