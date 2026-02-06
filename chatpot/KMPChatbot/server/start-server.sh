#!/bin/bash

echo "========================================"
echo "KMP Chatbot Server - Quick Start"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
    echo "Please install JDK 17+ from: https://adoptium.net/"
    exit 1
fi

# Check Maven
echo -n "Checking Maven installation... "
if command -v mvn &> /dev/null; then
    MVN_VERSION=$(mvn -version | head -n1 | awk '{print $3}')
    echo -e "${GREEN}✓ Maven $MVN_VERSION found${NC}"
else
    echo -e "${YELLOW}⚠ Maven not found (will use wrapper)${NC}"
fi

# Check API Key
echo -n "Checking API key configuration... "
if [ -z "$ANTHROPIC_API_KEY" ]; then
    if grep -q "your-api-key-here" src/main/resources/application.properties; then
        echo -e "${YELLOW}⚠ API key not configured${NC}"
        echo ""
        echo "Please set your Anthropic API key:"
        echo "1. Get key from: https://console.anthropic.com/"
        echo "2. Either:"
        echo "   - Set environment variable: export ANTHROPIC_API_KEY='sk-ant-...'"
        echo "   - Edit: src/main/resources/application.properties"
        echo ""
        read -p "Press Enter after configuring API key, or Ctrl+C to exit..."
    else
        echo -e "${GREEN}✓ API key configured${NC}"
    fi
else
    echo -e "${GREEN}✓ API key found in environment${NC}"
fi

# Select mode
echo ""
echo "Select mode:"
echo "1) Development (with live reload)"
echo "2) Production (build and run)"
echo "3) Docker (with PostgreSQL)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Starting in development mode..."
        echo "Server will start at: http://localhost:8080"
        echo "Swagger UI: http://localhost:8080/swagger-ui"
        echo "Dev UI: http://localhost:8080/q/dev"
        echo ""
        ./mvnw quarkus:dev
        ;;
    2)
        echo ""
        echo "Building for production..."
        ./mvnw clean package -DskipTests
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Build successful${NC}"
            echo ""
            echo "Starting server..."
            java -jar target/quarkus-app/quarkus-run.jar
        else
            echo -e "${RED}✗ Build failed${NC}"
            exit 1
        fi
        ;;
    3)
        echo ""
        if command -v docker &> /dev/null; then
            echo "Building and starting with Docker Compose..."
            ./mvnw clean package -DskipTests
            docker-compose up --build
        else
            echo -e "${RED}✗ Docker not found${NC}"
            echo "Please install Docker: https://docs.docker.com/get-docker/"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
