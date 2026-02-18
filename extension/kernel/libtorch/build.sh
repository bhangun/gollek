#!/bin/bash
# build.sh - Complete build script for PyTorch Quarkus Binding

set -e

echo "======================================"
echo " PyTorch Quarkus Binding Build Script"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
    if [ -z "$LIBTORCH_PATH" ]; then
        echo -e "${YELLOW}LIBTORCH_PATH not set. Downloading LibTorch...${NC}"
        
        echo "Select LibTorch version:"
        echo "1) CPU only (smaller, faster download)"
        echo "2) CUDA 11.8"
        echo "3) CUDA 12.1"
        read -p "Enter choice [1-3]: " CHOICE
        
        case $CHOICE in
            1)
                URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip"
                ;;
            2)
                URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip"
                ;;
            3)
                URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip"
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                exit 1
                ;;
        esac
        
        echo -e "${YELLOW}Downloading from $URL...${NC}"
        wget -q --show-progress "$URL" -O libtorch.zip
        
        echo -e "${YELLOW}Extracting...${NC}"
        unzip -q libtorch.zip
        rm libtorch.zip
        
        export LIBTORCH_PATH=$(pwd)/libtorch
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
    make -j$(nproc)
    
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
    
    java --enable-preview \
         --add-modules=jdk.incubator.foreign \
         -cp target/pytorch-quarkus-binding-1.0.0-SNAPSHOT.jar \
         io.quarkus.pytorch.examples.MNISTNet
    
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
    echo "  java --enable-preview --add-modules=jdk.incubator.foreign \\"
    echo "       -cp target/pytorch-quarkus-binding-1.0.0-SNAPSHOT.jar \\"
    echo "       io.quarkus.pytorch.examples.MNISTNet"
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
