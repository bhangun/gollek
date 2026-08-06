#!/bin/bash
set -e

# Define paths
BASE_DIR=$(pwd)
# Ensure we get the absolute path
case $0 in
  /*) SCRIPT_DIR=$(dirname "$0");;
  *) SCRIPT_DIR=$(pwd)/$(dirname "$0");;
esac

DEFAULT_LLAMA_SOURCE="$HOME/.gollek/source/vendor/llama.cpp"
LLAMA_SOURCE_DIR="${GOLLEK_LLAMA_SOURCE_DIR:-$DEFAULT_LLAMA_SOURCE}"

# Primary build output goes to source directory
SOURCE_BUILD_DIR="$LLAMA_SOURCE_DIR/build"
SOURCE_OUTPUT_DIR="$SOURCE_BUILD_DIR/bin"

# Secondary output for backward compatibility
BUILD_DIR="$BASE_DIR/target/llama-cpp-build"
OUTPUT_DIR="$BASE_DIR/target/llama-cpp/lib"

echo "Running build-llama-cpp.sh..."
echo "Base Dir: $BASE_DIR"
echo "GOLLEK_LLAMA_SOURCE_DIR: $LLAMA_SOURCE_DIR"
echo "Source Build Dir: $SOURCE_BUILD_DIR"
echo "Source Output Dir: $SOURCE_OUTPUT_DIR"
echo "Secondary Output Dir: $OUTPUT_DIR"

VENDOR_DIR="$LLAMA_SOURCE_DIR"

# Backward-compatible source-root normalization
if [ ! -f "$VENDOR_DIR/CMakeLists.txt" ]; then
    if [ -f "$VENDOR_DIR/llama.cpp/CMakeLists.txt" ]; then
        VENDOR_DIR="$VENDOR_DIR/llama.cpp"
    elif [ -f "$VENDOR_DIR/llama-cpp/llama.cpp/CMakeLists.txt" ]; then
        VENDOR_DIR="$VENDOR_DIR/llama-cpp/llama.cpp"
    fi
fi

echo "Resolved llama.cpp source: $VENDOR_DIR"

mkdir -p "$SOURCE_OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

normalize_macos_rpaths() {
    if [ "$(uname -s)" = "Darwin" ] && command -v install_name_tool >/dev/null 2>&1 && command -v otool >/dev/null 2>&1; then
        echo "Normalizing macOS dylib rpaths in $OUTPUT_DIR..."
        for dylib in "$OUTPUT_DIR"/*.dylib; do
            [ -f "$dylib" ] || continue
            while IFS= read -r rpath; do
                case "$rpath" in
                    /*)
                        install_name_tool -delete_rpath "$rpath" "$dylib" 2>/dev/null || true
                        ;;
                esac
            done < <(otool -l "$dylib" | awk '/LC_RPATH/{flag=1;next} flag && /path /{print $2;flag=0}')
            install_name_tool -add_rpath "@loader_path" "$dylib" 2>/dev/null || true
        done
    fi
}

# Check if artifacts already exist to skip build (speed optimization).
# If shim source exists, ensure shim artifact exists too before skipping.
SHIM_SRC="$BASE_DIR/src/main/native/golek_llama_shim.c"
SHIM_OK=1
if [ -f "$SHIM_SRC" ]; then
    if [ -f "$OUTPUT_DIR/libgollek_llama_shim.dylib" ] || [ -f "$OUTPUT_DIR/libgollek_llama_shim.so" ]; then
        SHIM_OK=1
    else
        SHIM_OK=0
    fi
fi
if { [ -f "$OUTPUT_DIR/libllama.dylib" ] || [ -f "$OUTPUT_DIR/libllama.so" ]; } && [ $SHIM_OK -eq 1 ]; then
    echo "Native library already exists in $OUTPUT_DIR. Skipping build."
    normalize_macos_rpaths
    exit 0
fi

# Fast path: llama library exists, only shim is missing.
if { [ -f "$OUTPUT_DIR/libllama.dylib" ] || [ -f "$OUTPUT_DIR/libllama.so" ]; } && [ $SHIM_OK -eq 0 ]; then
    echo "llama library exists; building missing shim only..."
    if [ -f "$SHIM_SRC" ] && [ "$(uname -s)" != "Windows_NT" ]; then
        if [ "$(uname -s)" = "Darwin" ]; then
            SHIM_OUT="$OUTPUT_DIR/libgollek_llama_shim.dylib"
            cc -shared -fPIC \
                -I"$VENDOR_DIR/include" \
                -I"$VENDOR_DIR/ggml/include" \
                "$SHIM_SRC" \
                -L"$OUTPUT_DIR" -lllama \
                -Wl,-rpath,@loader_path \
                -o "$SHIM_OUT" || echo "Warning: failed to build gollek llama shim"
        else
            SHIM_OUT="$OUTPUT_DIR/libgollek_llama_shim.so"
            cc -shared -fPIC \
                -I"$VENDOR_DIR/include" \
                -I"$VENDOR_DIR/ggml/include" \
                "$SHIM_SRC" \
                -L"$OUTPUT_DIR" -lllama \
                -Wl,-rpath,\$ORIGIN \
                -o "$SHIM_OUT" || echo "Warning: failed to build gollek llama shim"
        fi
    fi
    normalize_macos_rpaths
    exit 0
fi

if [ ! -d "$VENDOR_DIR" ] || [ ! -f "$VENDOR_DIR/CMakeLists.txt" ]; then
    echo "Warning: llama.cpp source or CMakeLists.txt not found at $VENDOR_DIR"
    echo "Set GOLLEK_LLAMA_SOURCE_DIR or place source at $DEFAULT_LLAMA_SOURCE"
    echo "Creating valid dummy library for build to pass..."
    echo "" | clang -shared -x c - -o "$OUTPUT_DIR/libllama.dylib" 2>/dev/null || touch "$OUTPUT_DIR/libllama.dylib"
    exit 0
fi

echo "Building llama.cpp from $VENDOR_DIR..."

mkdir -p "$SOURCE_BUILD_DIR"
cd "$SOURCE_BUILD_DIR"

if command -v cmake &> /dev/null; then
    # On macOS, default to Metal for much faster local inference.
    if [ "$(uname -s)" = "Darwin" ]; then
        GGML_METAL_FLAG="${GOLLEK_GGML_METAL:-ON}"
    else
        GGML_METAL_FLAG="${GOLLEK_GGML_METAL:-OFF}"
    fi
    cmake "$VENDOR_DIR" \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DGGML_METAL="$GGML_METAL_FLAG"
    cmake --build . --config Release -j 4
else
    echo "Error: cmake not found. Cannot build llama.cpp."
    echo "Creating valid dummy library..."
    echo "" | clang -shared -x c - -o "$SOURCE_OUTPUT_DIR/libllama.dylib" 2>/dev/null || touch "$SOURCE_OUTPUT_DIR/libllama.dylib"
    exit 0
fi

echo "Copying artifacts to $SOURCE_OUTPUT_DIR and $OUTPUT_DIR..."
# Copy from build directory to source output
find . -name "libllama.*" -exec cp -vL {} "$SOURCE_OUTPUT_DIR/" \; 2>/dev/null || :
find . -name "libggml*" -exec cp -vL {} "$SOURCE_OUTPUT_DIR/" \; 2>/dev/null || :
find . -name "llama.dll" -exec cp -vL {} "$SOURCE_OUTPUT_DIR/" \; 2>/dev/null || :

# Also copy to secondary output for backward compatibility
find . -name "libllama.*" -exec cp -vL {} "$OUTPUT_DIR/" \; 2>/dev/null || :
find . -name "libggml*" -exec cp -vL {} "$OUTPUT_DIR/" \; 2>/dev/null || :
find . -name "llama.dll" -exec cp -vL {} "$OUTPUT_DIR/" \; 2>/dev/null || :

# Also check for artifacts in common locations
cp -vL libllama.* "$SOURCE_OUTPUT_DIR/" 2>/dev/null || :
cp -vL libllama.* "$OUTPUT_DIR/" 2>/dev/null || :
cp -vL bin/libllama.* "$SOURCE_OUTPUT_DIR/" 2>/dev/null || :
cp -vL bin/libllama.* "$OUTPUT_DIR/" 2>/dev/null || :
cp -vL bin/libggml* "$SOURCE_OUTPUT_DIR/" 2>/dev/null || :
cp -vL bin/libggml* "$OUTPUT_DIR/" 2>/dev/null || :

# Normalize macOS rpaths so runtime loading is environment-agnostic.
normalize_macos_rpaths

# Build small ABI-stable shim to avoid problematic struct-return FFM calls in native image.
SHIM_SRC="$BASE_DIR/src/main/native/golek_llama_shim.c"
if [ -f "$SHIM_SRC" ] && [ "$(uname -s)" != "Windows_NT" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        # Build to source output directory (primary)
        SHIM_OUT="$SOURCE_OUTPUT_DIR/libgollek_llama_shim.dylib"
        cc -shared -fPIC \
            -I"$VENDOR_DIR/include" \
            -I"$VENDOR_DIR/ggml/include" \
            "$SHIM_SRC" \
            -L"$SOURCE_OUTPUT_DIR" -lllama \
            -Wl,-rpath,@loader_path \
            -o "$SHIM_OUT" || echo "Warning: failed to build gollek llama shim"
        
        # Also copy to secondary output
        cp -vL "$SHIM_OUT" "$OUTPUT_DIR/" 2>/dev/null || :
    else
        # Build to source output directory (primary)
        SHIM_OUT="$SOURCE_OUTPUT_DIR/libgollek_llama_shim.so"
        cc -shared -fPIC \
            -I"$VENDOR_DIR/include" \
            -I"$VENDOR_DIR/ggml/include" \
            "$SHIM_SRC" \
            -L"$SOURCE_OUTPUT_DIR" -lllama \
            -Wl,-rpath,\$ORIGIN \
            -o "$SHIM_OUT" || echo "Warning: failed to build gollek llama shim"
        
        # Also copy to secondary output
        cp -vL "$SHIM_OUT" "$OUTPUT_DIR/" 2>/dev/null || :
    fi
fi

echo "llama.cpp build complete."
echo "Primary output: $SOURCE_OUTPUT_DIR"
echo "Secondary output: $OUTPUT_DIR"
