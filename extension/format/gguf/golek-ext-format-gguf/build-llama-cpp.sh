#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/source/llama-cpp"

echo "Building llama.cpp native library..."

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build llama.cpp native library"
    echo ""
    echo "Options:"
    echo "  -p, --path PATH    Use existing llama.cpp directory at PATH"
    echo "  -v, --version TAG  Use specific git tag/commit (default: b3561 for old builds)"
    echo "  -b, --branch NAME  Use specific git branch (default: master)"
    echo "  -h, --help         Display this help message"
    echo ""
    echo "If no path is provided, the script will clone llama.cpp from GitHub."
    exit 1
}

# Parse command line arguments
LLAMA_PATH=""
GIT_VERSION="b3561"
GIT_BRANCH="master"
FORCE_REBUILD=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            LLAMA_PATH="$2"
            shift 2
            ;;
        -v|--version)
            GIT_VERSION="$2"
            shift 2
            ;;
        -b|--branch)
            GIT_BRANCH="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_REBUILD=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done


# Function to check if rebuild is needed
check_rebuild_needed() {
    local rebuild_needed=1

    # Skip check if force rebuild is requested
    if [ $FORCE_REBUILD -eq 1 ]; then
        echo "Force rebuild requested..."
        return 0  # Return 0 to indicate rebuild IS needed
    fi

    # Check if libraries exist (support both .so and .dylib extensions)
    if [ -f "$BUILD_DIR/lib/cpu/libllama.so" ] || [ -f "$BUILD_DIR/lib/cuda/libllama.so" ] || [ -f "$BUILD_DIR/lib/cpu/libllama.dylib" ] || [ -f "$BUILD_DIR/lib/cuda/libllama.dylib" ]; then
        echo "Existing llama.cpp libraries found in: $BUILD_DIR"
        read -p "Do you want to rebuild? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rebuild_needed=1
            echo "Will rebuild llama.cpp..."
        else
            rebuild_needed=0
            echo "Using existing build..."
        fi
    else
        echo "No existing libraries found, building..."
        rebuild_needed=1
    fi

    # Convert meaning: return 0 if rebuild is needed, 1 if not needed
    if [ $rebuild_needed -eq 1 ]; then
        return 0  # Rebuild IS needed -> return 0 (success)
    else
        return 1  # Rebuild is NOT needed -> return 1 (failure)
    fi
}

# Handle llama.cpp source
if [ -n "$LLAMA_PATH" ]; then
    # Use existing path
    echo "Using existing llama.cpp directory at: $LLAMA_PATH"
    
    # Check if the path ends with llama.cpp, if not, append it
    if [[ "$LLAMA_PATH" != *"llama.cpp" ]] && [ ! -f "$LLAMA_PATH/llama.h" ] && [ ! -f "$LLAMA_PATH/CMakeLists.txt" ]; then
        # Try with llama.cpp appended
        if [ -f "$LLAMA_PATH/llama.cpp/llama.h" ] || [ -f "$LLAMA_PATH/llama.cpp/CMakeLists.txt" ]; then
            LLAMA_PATH="$LLAMA_PATH/llama.cpp"
            echo "Adjusted path to: $LLAMA_PATH"
        fi
    fi
    
    if [ ! -d "$LLAMA_PATH" ]; then
        echo "Error: Directory does not exist: $LLAMA_PATH"
        exit 1
    fi
    
    # Check for llama.h or CMakeLists.txt in the directory
    if [ ! -f "$LLAMA_PATH/llama.h" ] && [ ! -f "$LLAMA_PATH/CMakeLists.txt" ]; then
        echo "Error: Directory does not appear to be a llama.cpp repository"
        echo "Expected to find llama.h or CMakeLists.txt in: $LLAMA_PATH"
        echo "Contents of directory:"
        ls -la "$LLAMA_PATH" | head -10
        exit 1
    fi
    
    # Create symlink or copy
    if [ -d "llama.cpp" ]; then
        echo "Removing existing llama.cpp directory in build folder..."
        rm -rf "llama.cpp"
    fi
    
    echo "Creating symlink to existing llama.cpp..."
    ln -sf "$LLAMA_PATH" "llama.cpp"
    cd llama.cpp

else
    # Default behavior: clone if not exists
    if [ ! -d "llama.cpp" ]; then
        echo "Cloning llama.cpp repository..."
        git clone https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp
        
        # For backward compatibility, still support old version tag
        if [ "$GIT_VERSION" != "b3561" ] || [ "$GIT_BRANCH" != "master" ]; then
            echo "Checking out branch: $GIT_BRANCH"
            git checkout "$GIT_BRANCH"
            
            if [ "$GIT_VERSION" != "b3561" ]; then
                echo "Checking out version: $GIT_VERSION"
                git checkout "$GIT_VERSION"
            fi
        else
            # Try to checkout the old stable version, but fall back to master if it doesn't exist
            if git rev-parse --verify "$GIT_VERSION" >/dev/null 2>&1; then
                echo "Checking out stable version: $GIT_VERSION"
                git checkout "$GIT_VERSION"
            else
                echo "Version $GIT_VERSION not found, using latest master"
                git checkout master
            fi
        fi
    else
        # Check if it's a symlink
        if [ -L "llama.cpp" ]; then
            echo "Following symlink to llama.cpp..."
            cd llama.cpp
        else
            echo "Updating existing llama.cpp repository..."
            cd llama.cpp
            
            # Check if it's a git repository before pulling
            if [ -d ".git" ]; then
                git checkout "$GIT_BRANCH"
                git pull
                
                # Check out specific version if requested
                if [ "$GIT_VERSION" != "b3561" ]; then
                    echo "Checking out version: $GIT_VERSION"
                    git checkout "$GIT_VERSION"
                fi
            else
                echo "Warning: llama.cpp exists but is not a git repository"
                echo "Using existing code as-is"
            fi
        fi
    fi
fi

# Verify we're in the right directory
if [ ! -f "llama.h" ] && [ ! -f "CMakeLists.txt" ] && [ ! -f "Makefile" ]; then
    echo "Error: Could not find llama.h, CMakeLists.txt, or Makefile in llama.cpp directory"
    exit 1
fi

echo "Current directory: $(pwd)"
echo "Detecting build system..."

# Determine build system and build accordingly
if [ -f "CMakeLists.txt" ]; then
    echo "CMake build system detected"

       # Check if rebuild is needed
    if ! check_rebuild_needed; then
        echo "Skipping build, using existing libraries."
        exit 0
    fi
    
    # Create and enter build directory
    mkdir -p build
    cd build
    
    # Clean previous CMake build
    echo "Cleaning previous CMake build..."
    rm -rf ./* 2>/dev/null || true
    
    # Configure based on available features
    echo "Configuring CMake..."
    
    if command -v nvcc &> /dev/null || [ -n "$CUDA_HOME" ] || [ -d "/usr/local/cuda" ]; then
        echo "CUDA detected, building with GPU support..."
        cmake .. -DLLAMA_CUBLAS=ON -DCMAKE_BUILD_TYPE=Release
        CMAKE_TARGET="llama"
    else
        echo "Building CPU-only version..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
        CMAKE_TARGET="llama"
    fi
    
    # Build
    echo "Building with CMake..."
    cmake --build . --config Release --target $CMAKE_TARGET -j$(nproc)
    
    # Find the built library
    if [ -f "bin/libllama.so" ]; then
        LIBRARY_PATH="bin/libllama.so"
    elif [ -f "bin/libllama.dylib" ]; then
        LIBRARY_PATH="bin/libllama.dylib"
    elif [ -f "libllama.so" ]; then
        LIBRARY_PATH="libllama.so"
    elif [ -f "libllama.dylib" ]; then
        LIBRARY_PATH="libllama.dylib"
    elif [ -f "libllama.a" ]; then
        LIBRARY_PATH="libllama.a"
    else
        echo "Warning: Could not find built library in expected locations"
        echo "Trying to find it..."
        LIBRARY_PATH=$(find . -name "libllama*.so" -o -name "libllama*.dylib" -o -name "libllama*.a" | head -1)
        if [ -z "$LIBRARY_PATH" ]; then
            echo "Error: Could not find built llama library"
            exit 1
        fi
    fi
    
    echo "Found library at: $LIBRARY_PATH"

    # Store the full path to the library before changing directory
    FULL_LIBRARY_PATH="$LIBRARY_PATH"

    # Go back to llama.cpp directory
    cd ..
    
elif [ -f "Makefile" ]; then
    echo "Makefile build system detected"

        # Check if rebuild is needed
    if ! check_rebuild_needed; then
        echo "Skipping build, using existing libraries."
        exit 0
    fi
    
    # Clean previous builds
    echo "Cleaning previous builds..."
    make clean 2>/dev/null || true
    
    # Check for CUDA
    if command -v nvcc &> /dev/null; then
        echo "CUDA detected, building with GPU support..."
        make LLAMA_CUBLAS=1 -j$(nproc)
    else
        echo "Building CPU-only version..."
        make -j$(nproc)
    fi
    
    LIBRARY_PATH="libllama.so"
    # For Makefile builds, the library is in the current directory
    FULL_LIBRARY_PATH="$LIBRARY_PATH"

else
    echo "Error: Could not determine build system"
    echo "Expected CMakeLists.txt or Makefile"
    exit 1
fi

# Copy libraries based on build type
# Determine the destination library name based on the source file extension
if [[ "$FULL_LIBRARY_PATH" == *.dylib ]]; then
    DEST_LIB_NAME="libllama.dylib"
elif [[ "$FULL_LIBRARY_PATH" == *.a ]]; then
    DEST_LIB_NAME="libllama.a"
else
    # Default to .so for Linux and other systems
    DEST_LIB_NAME="libllama.so"
fi

# For CMake builds, the library path is relative to the build directory
# We need to determine this based on which build system was used
# Since we're now in the llama.cpp directory, we need to check which build was performed
# We can infer this from the original check done earlier
if [ -f "CMakeLists.txt" ]; then
    # This is not the right way to detect which build was performed
    # Let me use a different approach - store the build type earlier
    # Actually, let me just check if the library exists in the build subdirectory
    if [ -f "build/$FULL_LIBRARY_PATH" ]; then
        ACTUAL_LIBRARY_PATH="build/$FULL_LIBRARY_PATH"
    else
        ACTUAL_LIBRARY_PATH="$FULL_LIBRARY_PATH"
    fi
else
    ACTUAL_LIBRARY_PATH="$FULL_LIBRARY_PATH"
fi

if command -v nvcc &> /dev/null || [ -n "$CUDA_HOME" ] || [ -d "/usr/local/cuda" ]; then
    echo "Copying CUDA/GPU library..."
    mkdir -p "$BUILD_DIR/lib/cuda"
    cp -f "$ACTUAL_LIBRARY_PATH" "$BUILD_DIR/lib/cuda/$DEST_LIB_NAME"
fi

# Always copy CPU library (might be same as GPU version)
echo "Copying CPU library..."
mkdir -p "$BUILD_DIR/lib/cpu"
cp -f "$ACTUAL_LIBRARY_PATH" "$BUILD_DIR/lib/cpu/$DEST_LIB_NAME"

# Copy headers
echo "Copying headers..."
mkdir -p "$BUILD_DIR/include"
find . -name "*.h" -maxdepth 1 -exec cp -f {} "$BUILD_DIR/include/" \; 2>/dev/null || true

# Copy ggml headers if they exist in a ggml subdirectory
if [ -d "ggml" ]; then
    find ggml -name "*.h" -exec cp -f {} "$BUILD_DIR/include/" \; 2>/dev/null || true
fi

echo ""
echo "Build summary:"
echo "- Build directory: $BUILD_DIR"
if [ -n "$LLAMA_PATH" ]; then
    echo "- Source: Existing directory at $LLAMA_PATH"
else
    echo "- Source: Git repository (cloned/updated)"
    echo "- Git branch: $GIT_BRANCH"
    if [ "$GIT_VERSION" != "b3561" ]; then
        echo "- Git version: $GIT_VERSION"
    fi
fi
if [ -f "CMakeLists.txt" ]; then
    echo "- Build system: CMake"
elif [ -f "Makefile" ]; then
    echo "- Build system: Makefile"
fi
if command -v nvcc &> /dev/null || [ -n "$CUDA_HOME" ] || [ -d "/usr/local/cuda" ]; then
    echo "- GPU support: Enabled (CUDA)"
else
    echo "- GPU support: CPU-only"
fi
echo "- Libraries: $BUILD_DIR/lib/{cpu,cuda}/libllama.so"
echo "- Headers: $BUILD_DIR/include/"
echo "- Library found at: $LIBRARY_PATH"
echo ""
echo "llama.cpp build completed successfully!"