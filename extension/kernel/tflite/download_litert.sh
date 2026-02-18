#!/bin/bash
set -e

# Directory to place the library
TARGET_DIR="target/test-libs"
mkdir -p "$TARGET_DIR"

OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected OS: $OS, Arch: $ARCH"

if [ "$OS" == "Darwin" ]; then
    if [ "$ARCH" == "arm64" ]; then
        # Using community build for macOS arm64
        URL="https://github.com/kafkaesque-io/tensorflow-lite-bins/raw/master/v2.16.1/macos_arm64/libtensorflowlite_c.dylib"
        FILENAME="libtensorflowlite_c.dylib"
    elif [ "$ARCH" == "x86_64" ]; then
        URL="https://github.com/kafkaesque-io/tensorflow-lite-bins/raw/master/v2.16.1/macos_x64/libtensorflowlite_c.dylib"
        FILENAME="libtensorflowlite_c.dylib"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
elif [ "$OS" == "Linux" ]; then
    if [ "$ARCH" == "x86_64" ]; then
        URL="https://github.com/kafkaesque-io/tensorflow-lite-bins/raw/master/v2.16.1/linux_x64/libtensorflowlite_c.so"
        FILENAME="libtensorflowlite_c.so"
    elif [ "$ARCH" == "aarch64" ]; then
        URL="https://github.com/kafkaesque-io/tensorflow-lite-bins/raw/master/v2.16.1/linux_aarch64/libtensorflowlite_c.so"
        FILENAME="libtensorflowlite_c.so"
    else
         echo "Unsupported architecture: $ARCH"
         exit 1
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

echo "Downloading $FILENAME to $TARGET_DIR..."
curl -L -o "$TARGET_DIR/$FILENAME" "$URL"

echo "Download complete."
echo "Library path: $(pwd)/$TARGET_DIR/$FILENAME"
echo ""
echo "You can now run tests with: mvn test"
