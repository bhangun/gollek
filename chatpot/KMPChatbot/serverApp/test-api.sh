#!/bin/bash

# KMP Chatbot Server - API Test Script
# Tests all major endpoints

BASE_URL="http://localhost:8080"

echo "=========================================="
echo "KMP Chatbot Server - API Tests"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    
    echo -n "Testing $name... "
    
    if [ -z "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X $method "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X $method "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        echo "   Response: $(echo $body | jq -c . 2>/dev/null || echo $body)"
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $http_code)"
        echo "   Response: $body"
    fi
    echo ""
}

# Check if server is running
echo "Checking if server is running..."
if ! curl -s "$BASE_URL/q/health/live" > /dev/null; then
    echo -e "${RED}Error: Server is not running at $BASE_URL${NC}"
    echo "Start the server with: ./mvnw quarkus:dev"
    exit 1
fi
echo -e "${GREEN}Server is running!${NC}"
echo ""

# Health Check Tests
echo "=== Health Checks ==="
test_endpoint "Liveness Check" "GET" "/q/health/live"
test_endpoint "Readiness Check" "GET" "/q/health/ready"
test_endpoint "Chat Health" "GET" "/api/chat/health"

# Conversation Tests
echo "=== Conversation Endpoints ==="
test_endpoint "List Conversations" "GET" "/api/conversations"
test_endpoint "Create Conversation" "POST" "/api/conversations" \
    '{"title": "Test Conversation"}'

# Get first conversation ID (if exists)
CONV_ID=$(curl -s "$BASE_URL/api/conversations" | jq -r '.[0].id // empty')

if [ ! -z "$CONV_ID" ]; then
    test_endpoint "Get Conversation" "GET" "/api/conversations/$CONV_ID"
else
    echo "No conversations found, skipping conversation detail test"
    echo ""
fi

# Chat Tests
echo "=== Chat Endpoints ==="
test_endpoint "Send Message (new conversation)" "POST" "/api/chat/message" \
    '{"content": "Hello! This is a test message."}'

test_endpoint "Send Message (existing conversation)" "POST" "/api/chat/message" \
    "{\"content\": \"How are you?\", \"conversationId\": $CONV_ID}"

if [ ! -z "$CONV_ID" ]; then
    test_endpoint "Get Conversation Messages" "GET" "/api/chat/conversations/$CONV_ID/messages"
fi

# API Documentation
echo "=== API Documentation ==="
echo -n "Checking OpenAPI spec... "
if curl -s "$BASE_URL/q/openapi" > /dev/null; then
    echo -e "${GREEN}✓ Available${NC}"
    echo "   URL: $BASE_URL/q/openapi"
else
    echo -e "${RED}✗ Not available${NC}"
fi
echo ""

echo -n "Checking Swagger UI... "
if curl -s "$BASE_URL/swagger-ui/" > /dev/null; then
    echo -e "${GREEN}✓ Available${NC}"
    echo "   URL: $BASE_URL/swagger-ui/"
else
    echo -e "${RED}✗ Not available${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Server URL: $BASE_URL"
echo "Swagger UI: $BASE_URL/swagger-ui/"
echo "OpenAPI Spec: $BASE_URL/q/openapi"
echo "Dev UI: $BASE_URL/q/dev/"
echo ""
echo "All major endpoints tested!"
echo "Check Swagger UI for interactive testing."
