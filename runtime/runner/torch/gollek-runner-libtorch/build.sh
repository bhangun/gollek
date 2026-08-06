#!/bin/bash
# build.sh - Complete build script for PyTorch Quarkus Binding

set -e

echo "======================================"
echo " Gollek LibTorch Runner Build Script"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

resolve_gollek_home() {
    if [ -n "${GOLLEK_HOME:-}" ]; then
        echo "$GOLLEK_HOME"
        return
    fi
    echo "$HOME/.gollek"
}

GOLLEK_HOME="$(resolve_gollek_home)"

download_file() {
    local url="$1"
    local output="$2"

    if command -v curl &> /dev/null; then
        curl -L --fail --show-error "$url" -o "$output"
        return
    fi

    if command -v wget &> /dev/null; then
        wget -q --show-progress "$url" -O "$output"
        return
    fi

    echo -e "${RED}Error: curl or wget is required to download LibTorch.${NC}"
    exit 1
}

shared_ext() {
    case "$(uname -s)" in
        Darwin) echo ".dylib" ;;
        MINGW*|MSYS*|CYGWIN*) echo ".dll" ;;
        *) echo ".so" ;;
    esac
}

validate_libtorch_dir() {
    local dir="$1"
    local expected_ext
    expected_ext="$(shared_ext)"
    local shared_list
    shared_list="$(
        find "$dir" -maxdepth 2 -type f \( -name '*.dylib' -o -name '*.so' -o -name '*.dll' \) -print 2>/dev/null \
            | sed "s#^$dir/##" \
            | sort
    )"

    if [ -z "$shared_list" ]; then
        return 0
    fi

    if printf '%s\n' "$shared_list" | grep -q "${expected_ext}\$"; then
        return 0
    fi

    echo -e "${RED}Downloaded LibTorch archive targets a different platform.${NC}"
    echo "$shared_list"
    case "$(uname -s):$(uname -m)" in
        Darwin:arm64|Darwin:aarch64)
            echo "Use https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip on Apple Silicon."
            ;;
        Darwin:*)
            echo "Use a macOS LibTorch archive instead of a Linux .so bundle."
            ;;
    esac
    exit 1
}

default_libtorch_url() {
    case "$(uname -s):$(uname -m)" in
        Darwin:arm64|Darwin:aarch64)
            echo "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-arm64-latest.zip"
            ;;
        Darwin:*)
            echo "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip"
            ;;
        Linux:*)
            echo "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip"
            ;;
        *)
            echo ""
            ;;
    esac
}

parallel_jobs() {
    if command -v nproc &> /dev/null; then
        nproc
        return
    fi
    if [ "$(uname -s)" = "Darwin" ]; then
        sysctl -n hw.ncpu
        return
    fi
    echo 4
}

# Check requirements
check_requirements() {
    echo -e "${YELLOW}Checking requirements...${NC}"
    
    # Check JDK 25
    if ! command -v java &> /dev/null; then
        echo -e "${RED}Error: Java not found. Please install JDK 25.${NC}"
        exit 1
    fi
    
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
    if [ "$JAVA_VERSION" -lt 25 ]; then
        echo -e "${RED}Error: JDK 25 or later required. Found JDK $JAVA_VERSION${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ JDK $JAVA_VERSION found${NC}"
    
    # Check Maven
    if ! command -v mvn &> /dev/null; then
        echo -e "${RED}Error: Maven not found. Please install Maven 3.8+${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Maven found${NC}"
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}Error: CMake not found. Please install CMake 3.18+${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ CMake found${NC}"
    
    # Check C++ compiler
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        echo -e "${RED}Error: C++ compiler not found. Please install g++ or clang++${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ C++ compiler found${NC}"
}

# Download LibTorch
download_libtorch() {
    if [ -z "${LIBTORCH_PATH:-}" ]; then
        if [ -n "${GOLLEK_LIBTORCH_SOURCE_DIR:-}" ] && [ -d "${GOLLEK_LIBTORCH_SOURCE_DIR}" ]; then
            export LIBTORCH_PATH="${GOLLEK_LIBTORCH_SOURCE_DIR}"
        elif [ -d "$GOLLEK_HOME/source/vendor/libtorch" ]; then
            export LIBTORCH_PATH="$GOLLEK_HOME/source/vendor/libtorch"
        fi
    fi

    if [ -z "$LIBTORCH_PATH" ]; then
        echo -e "${YELLOW}LIBTORCH_PATH not set. Downloading LibTorch...${NC}"
        URL="${GOLLEK_LIBTORCH_ARCHIVE_URL:-}"

        if [ -z "$URL" ]; then
            if [ "$(uname -s)" = "Darwin" ]; then
                URL="$(default_libtorch_url)"
                echo "Detected macOS; using CPU LibTorch archive:"
                echo "  $URL"
            else
                echo "Select LibTorch version:"
                echo "1) CPU only (smaller, faster download)"
                echo "2) CUDA 11.8"
                echo "3) CUDA 12.1"
                read -p "Enter choice [1-3]: " CHOICE

                case $CHOICE in
                    1)
                        URL="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip"
                        ;;
                    2)
                        URL="https://download.pytorch.org/libtorch/nightly/cu118/libtorch-shared-with-deps-latest.zip"
                        ;;
                    3)
                        URL="https://download.pytorch.org/libtorch/nightly/cu121/libtorch-shared-with-deps-latest.zip"
                        ;;
                    *)
                        echo -e "${RED}Invalid choice${NC}"
                        exit 1
                        ;;
                esac
            fi
        fi

        if [ -z "$URL" ]; then
            echo -e "${RED}Error: could not determine a LibTorch download URL for this platform.${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}Downloading from $URL...${NC}"
        download_file "$URL" libtorch.zip
        
        echo -e "${YELLOW}Extracting...${NC}"
        unzip -q libtorch.zip
        rm libtorch.zip

        validate_libtorch_dir "$(pwd)/libtorch"
        
        mkdir -p "$GOLLEK_HOME/source/vendor"
        rm -rf "$GOLLEK_HOME/source/vendor/libtorch"
        mv "$(pwd)/libtorch" "$GOLLEK_HOME/source/vendor/libtorch"
        export LIBTORCH_PATH="$GOLLEK_HOME/source/vendor/libtorch"
        echo -e "${GREEN}✓ LibTorch downloaded to $LIBTORCH_PATH${NC}"
    else
        echo -e "${GREEN}✓ Using existing LibTorch at $LIBTORCH_PATH${NC}"
    fi
}

# Build native library
build_native() {
    echo -e "${YELLOW}Building native library...${NC}"
    
    if [ ! -d "$LIBTORCH_PATH" ]; then
        echo -e "${RED}Error: LibTorch not found at $LIBTORCH_PATH${NC}"
        exit 1
    fi
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with CMake
    echo -e "${YELLOW}Running CMake...${NC}"
    cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
          -DCMAKE_BUILD_TYPE=Release \
          ..
    
    # Build
    echo -e "${YELLOW}Compiling...${NC}"
    make -j"$(parallel_jobs)"
    
    cd ..
    echo -e "${GREEN}✓ Native library built successfully${NC}"
}

# Build Java project
build_java() {
    echo -e "${YELLOW}Building Java project...${NC}"
    
    mvn clean package -DskipTests
    
    echo -e "${GREEN}✓ Java project built successfully${NC}"
}

# Run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    
    export LD_LIBRARY_PATH="$LIBTORCH_PATH/lib:$PWD/build/lib:$LD_LIBRARY_PATH"
    
    java --enable-native-access=ALL-UNNAMED \
         -cp target/gollek-ext-runner-libtorch-0.1.0.jar \
         tech.kayys.gollek.inference.libtorch.examples.LibTorchForwardBenchmark
    
    echo -e "${GREEN}✓ Tests completed${NC}"
}

# Main build process
main() {
    check_requirements
    download_libtorch
    build_native
    build_java
    
    echo ""
    echo -e "${GREEN}======================================"
    echo "Build completed successfully!"
    echo "======================================${NC}"
    echo ""
    echo "To run the application:"
    echo ""
    echo "  export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$PWD/build/lib:\$LD_LIBRARY_PATH"
    echo "  mvn quarkus:dev"
    echo ""
    echo "Or run the example:"
    echo ""
    echo "  export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$PWD/build/lib:\$LD_LIBRARY_PATH"
    echo "  java --enable-native-access=ALL-UNNAMED \\"
    echo "       -cp target/gollek-ext-runner-libtorch-0.1.0.jar \\"
    echo "       tech.kayys.gollek.inference.libtorch.examples.LibTorchForwardBenchmark"
    echo ""
    
    # Optionally run tests
    read -p "Run example now? [y/N]: " RUN_EXAMPLE
    if [ "$RUN_EXAMPLE" = "y" ] || [ "$RUN_EXAMPLE" = "Y" ]; then
        run_tests
    fi
}

# Parse command line arguments
case "${1:-}" in
    clean)
        echo "Cleaning build artifacts..."
        rm -rf build target
        echo "Clean complete"
        ;;
    native)
        check_requirements
        download_libtorch
        build_native
        ;;
    java)
        check_requirements
        build_java
        ;;
    test)
        check_requirements
        run_tests
        ;;
    *)
        main
        ;;
esac
