#!/bin/bash

# Quarkus Server Quick Start Script

echo "=========================================="
echo "KMP Chatbot Server - Quick Start"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
    exit 1
fi

# Check Maven
echo -n "Checking Maven installation... "
if command -v mvn &> /dev/null; then
    MVN_VERSION=$(mvn -version | head -n 1 | awk '{print $3}')
    echo -e "${GREEN}✓ Maven $MVN_VERSION found${NC}"
else
    echo -e "${YELLOW}⚠ Maven not found, using wrapper${NC}"
fi

# Check API Key
echo ""
echo "Checking API key configuration..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠ ANTHROPIC_API_KEY not set${NC}"
    echo ""
    echo "Please set your API key:"
    echo "  export ANTHROPIC_API_KEY=sk-ant-api03-..."
    echo ""
    echo "Or edit src/main/resources/application.properties"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
else
    echo -e "${GREEN}✓ API key configured${NC}"
fi

# Navigate to server directory
cd "$(dirname "$0")"

echo ""
echo "Starting Quarkus server in dev mode..."
echo ""
echo "Features available:"
echo "  • Live reload on code changes"
echo "  • Dev UI: http://localhost:8080/q/dev/"
echo "  • Swagger UI: http://localhost:8080/swagger-ui/"
echo "  • API: http://localhost:8080/api/"
echo ""

# Check if mvnw exists
if [ -f "./mvnw" ]; then
    ./mvnw quarkus:dev
elif command -v mvn &> /dev/null; then
    mvn quarkus:dev
else
    echo -e "${RED}Error: Neither mvnw nor mvn found${NC}"
    exit 1
fi
