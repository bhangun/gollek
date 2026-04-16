#!/bin/bash
set -e

LIB_NAME="libgguf_bridge"
LIB_VERSION="1.0.0"
TARGET_DIR="$HOME/.gollek/libs/gguf_bridge/${LIB_VERSION}"

# Use environment variable set by Maven, or default
BUILD_DIR="${GOLLEK_NATIVE_LIB_DIR:-$(cd "$(dirname "$0")/.." && pwd)/../gguf-bridge/build}"

# Resolve to absolute path
BUILD_DIR=$(cd "${BUILD_DIR}" 2>/dev/null && pwd || echo "${BUILD_DIR}")

echo "========================================="
echo "GGUF Native Library Installation"
echo "========================================="
echo "Build directory: ${BUILD_DIR}"
echo "Target directory: ${TARGET_DIR}"
echo ""

# Find the library file - prefer the versioned binary over symlinks
LIB_FILE=""
if [ -f "${BUILD_DIR}/${LIB_NAME}.1.0.0.dylib" ]; then
    LIB_FILE="${BUILD_DIR}/${LIB_NAME}.1.0.0.dylib"
    echo "Found: ${LIB_NAME}.1.0.0.dylib (macOS versioned)"
elif [ -f "${BUILD_DIR}/${LIB_NAME}.1.dylib" ]; then
    LIB_FILE="${BUILD_DIR}/${LIB_NAME}.1.dylib"
    echo "Found: ${LIB_NAME}.1.dylib (macOS symlink to version)"
elif [ -e "${BUILD_DIR}/${LIB_NAME}.dylib" ]; then
    LIB_FILE="${BUILD_DIR}/${LIB_NAME}.dylib"
    echo "Found: ${LIB_NAME}.dylib (macOS)"
elif [ -f "${BUILD_DIR}/${LIB_NAME}.so" ]; then
    LIB_FILE="${BUILD_DIR}/${LIB_NAME}.so"
    echo "Found: ${LIB_NAME}.so (Linux)"
elif [ -f "${BUILD_DIR}/${LIB_NAME}.dll" ]; then
    LIB_FILE="${BUILD_DIR}/${LIB_NAME}.dll"
    echo "Found: ${LIB_NAME}.dll (Windows)"
else
    echo "WARNING: Native library not found in ${BUILD_DIR}"
    echo "Available files:"
    ls -la "${BUILD_DIR}/" 2>/dev/null || echo "Directory does not exist"
    exit 0
fi

# Get the base name of the library file we found
LIB_BASE_NAME=$(basename "${LIB_FILE}")

# Resolve any symlinks to get the actual file
REAL_LIB_FILE=$(readlink -f "${LIB_FILE}" 2>/dev/null || echo "${LIB_FILE}")

# Create target directory
mkdir -p "${TARGET_DIR}"

# Remove any existing files/symlinks that might interfere
rm -f "${TARGET_DIR}/${LIB_BASE_NAME}" "${TARGET_DIR}/${LIB_NAME}.dylib" "${TARGET_DIR}/${LIB_NAME}.1.dylib"

# Copy library - copy the actual binary file (resolve symlinks first)
cp "${REAL_LIB_FILE}" "${TARGET_DIR}/${LIB_BASE_NAME}"
echo "Copied library to ${TARGET_DIR}/${LIB_BASE_NAME}"

# Create unversioned symlink for GGUFNative.java to find
ln -sf "${LIB_BASE_NAME}" "${TARGET_DIR}/${LIB_NAME}.dylib"
echo "Created unversioned symlink: ${LIB_NAME}.dylib -> ${LIB_BASE_NAME}"

# Set permissions
chmod 755 "${TARGET_DIR}/${LIB_BASE_NAME}"

# Clear macOS quarantine
if [ "$(uname)" = "Darwin" ] && [ -x "/usr/bin/xattr" ]; then
    xattr -dr com.apple.quarantine "${TARGET_DIR}" 2>/dev/null || true
    echo "Cleared macOS quarantine attributes"
fi

# Generate SHA-256 checksum
if [ "$(uname)" = "Darwin" ]; then
    shasum -a 256 "${TARGET_DIR}/${LIB_BASE_NAME}" | cut -d' ' -f1 > "${TARGET_DIR}/${LIB_BASE_NAME}.sha256"
else
    sha256sum "${TARGET_DIR}/${LIB_BASE_NAME}" | cut -d' ' -f1 > "${TARGET_DIR}/${LIB_BASE_NAME}.sha256"
fi
echo "Generated SHA-256 checksum"

# Create symlink without version for easier access
ln -sf "${TARGET_DIR}/${LIB_BASE_NAME}" "$HOME/.gollek/libs/${LIB_NAME}"
echo "Created symlink: $HOME/.gollek/libs/${LIB_NAME}"

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo "Location: ${TARGET_DIR}"
echo "Library: ${LIB_BASE_NAME}"
echo "Checksum: $(cat ${TARGET_DIR}/${LIB_BASE_NAME}.sha256)"
echo ""
echo "To use this library, add to your Java command:"
echo "  -Djava.library.path=${TARGET_DIR}"
echo "Or set environment variable:"
echo "  export JAVA_LIBRARY_PATH=${TARGET_DIR}"
echo "========================================="
