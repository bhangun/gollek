#!/bin/bash

echo "========================================"
echo "KMP Chatbot - Quick Start Script"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Java
echo -n "Checking Java installation... "
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | awk -F. '{print $1}')
    if [ "$JAVA_VERSION" -ge 17 ]; then
        echo -e "${GREEN}✓ Java $JAVA_VERSION found${NC}"
    else
        echo -e "${RED}✗ Java 17+ required (found Java $JAVA_VERSION)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Java not found${NC}"
    echo "Please install JDK 17+ from: https://www.oracle.com/java/technologies/downloads/"
    exit 1
fi

# Check API Key
echo -n "Checking API key configuration... "
if grep -q "your-api-key-here" shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt; then
    echo -e "${YELLOW}⚠ API key not configured${NC}"
    echo ""
    echo "Please configure your Anthropic API key:"
    echo "1. Get key from: https://console.anthropic.com/"
    echo "2. Edit: shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt"
    echo "3. Replace 'your-api-key-here' with your actual key"
    echo ""
    read -p "Press Enter after configuring API key, or Ctrl+C to exit..."
else
    echo -e "${GREEN}✓ API key configured${NC}"
fi

# Build project
echo ""
echo "Building shared module..."
./gradlew :shared:build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Detect platform and run
echo ""
echo "Select platform to run:"
echo "1) Desktop (recommended for quick test)"
echo "2) Android (requires Android Studio)"
echo "3) Build only (no run)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Starting Desktop app..."
        ./gradlew :desktopApp:run
        ;;
    2)
        echo "Building Android app..."
        ./gradlew :androidApp:assembleDebug
        echo ""
        echo "APK created at: androidApp/build/outputs/apk/debug/"
        echo "Install with: adb install androidApp/build/outputs/apk/debug/androidApp-debug.apk"
        ;;
    3)
        echo "Build complete!"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
echo "For more options, see SETUP_GUIDE.md"
