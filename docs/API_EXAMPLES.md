# Gollek API — Complete curl Examples

**Base URL**: `http://localhost:9131`
**Authentication**: Add header `-H 'X-API-Key: community'`

Note: Some clients (including curl without explicit Accept headers) may receive an HTML 404 page from Quarkus. If you see "Resource not found" HTML, add `-H 'Accept: application/json'` to your request. Examples below include the Accept header where appropriate.

---

## 🏥 Health Check

### Check Server Status
```bash
curl http://localhost:9131/health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

## 📦 Models Management

### List All Available Models

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models'
```

**Options:**
```bash
# List only runnable models
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models?runnableOnly=true'

# Limit results to 10
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models?limit=10'

# OpenAI-compatible format
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models?compat=openai'

# Filter by namespace
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models?namespace=huggingface'
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1234567890,
      "owned_by": "gollek"
    }
  ]
}
```

### Get Model Details

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models/gpt-3.5-turbo'
```

**Response:**
```json
{
  "modelId": "gpt-3.5-turbo",
  "format": "gguf",
  "description": "GPT-3.5 Turbo model",
  "size": 4294967296
}
```

### Get Model Capabilities

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models/gpt-3.5-turbo/capabilities'
```

**Response:**
```json
{
  "modelId": "gpt-3.5-turbo",
  "capabilities": {
    "streaming": true,
    "tools": true,
    "vision": false,
    "function_calling": true
  }
}
```

### Pull (Download) a Model

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "modelSpec": "gpt-3.5-turbo:latest",
    "revision": "main",
    "force": false
  }' \
  'http://localhost:9131/v1/models/pull'
```

**Response:**
```json
{
  "status": "pull_started",
  "jobId": "job-12345-abcde"
}
```

### Stream Model Pull Progress

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models/pull/stream/job-12345-abcde' \
  -H 'Accept: text/event-stream'
```

**Response (SSE Stream):**
```
data: {"jobId":"job-12345-abcde","status":"downloading","progress":25,"total":100}
data: {"jobId":"job-12345-abcde","status":"downloading","progress":50,"total":100}
data: {"jobId":"job-12345-abcde","status":"completed","progress":100,"total":100}
```

### Delete a Model

```bash
curl -X DELETE -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models/gpt-3.5-turbo'
```

**Response:** 204 No Content

---

## 💬 Chat Completions (OpenAI-Compatible)

### Simple Chat Completion (Non-Streaming)

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "Hello! What is 2 + 2?"
      }
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' \
  'http://localhost:9131/v1/chat/completions'
```

**Response:**
```json
{
  "id": "chatcmpl-8MlyL5dPT8qXX0q7L0F4Z5E6",
  "object": "chat.completion",
  "created": 1699107661,
  "model": "gpt-3.5-turbo",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  },
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4"
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

### Chat with System Prompt

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant that always responds in Spanish."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }' \
  'http://localhost:9131/v1/chat/completions'
```

### Streaming Chat Completion (Server-Sent Events)

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "Write a haiku about clouds"
      }
    ],
    "stream": true,
    "temperature": 0.8
  }' \
  'http://localhost:9131/v1/chat/completions'
```

**Response (SSE Stream):**
```
data: {"id":"chatcmpl-8MlyL5dPT8qXX0q7L0F4Z5E6","choices":[{"delta":{"role":"assistant","content":"Clouds"},"index":0}]}
data: {"id":"chatcmpl-8MlyL5dPT8qXX0q7L0F4Z5E6","choices":[{"delta":{"content":" drift"},"index":0}]}
data: {"id":"chatcmpl-8MlyL5dPT8qXX0q7L0F4Z5E6","choices":[{"delta":{"content":" gently"},"index":0}]}
data: [DONE]
```

### Chat with Function Calling (Tools)

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Seattle?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "City name"
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }' \
  'http://localhost:9131/v1/chat/completions'
```

### Chat with Vision (Image Analysis)

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4-vision",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg"
            }
          }
        ]
      }
    ]
  }' \
  'http://localhost:9131/v1/chat/completions'
```

### Chat with RAG Context

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "Based on the documentation, how do I use this feature?"
      }
    ],
    "metadata": {
      "rag_context": [
        {
          "source": "docs/feature.md",
          "content": "To use this feature, you need to...",
          "relevance": 0.95
        }
      ]
    }
  }' \
  'http://localhost:9131/v1/chat/completions'
```

---

## 🔌 Embeddings

### Create Embeddings

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox",
    "encoding_format": "float"
  }' \
  'http://localhost:9131/v1/embeddings'
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, 0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

### Batch Embeddings

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "text-embedding-ada-002",
    "input": [
      "First sentence",
      "Second sentence",
      "Third sentence"
    ]
  }' \
  'http://localhost:9131/v1/embeddings'
```

---

## 🤖 Inference

### Raw Inference Request

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{
    "modelId": "gpt-3.5-turbo",
    "prompt": "The answer to life is",
    "maxTokens": 50,
    "temperature": 0.7,
    "topP": 0.9,
    "topK": 40
  }' \
  'http://localhost:9131/v1/inference'
```

**Response:**
```json
{
  "modelId": "gpt-3.5-turbo",
  "text": "The answer to life is 42, according to Douglas Adams.",
  "finishReason": "stop",
  "tokenCount": 12,
  "executionTimeMs": 245
}
```

### Streaming Inference

```bash
curl -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "modelId": "gpt-3.5-turbo",
    "prompt": "Story about a robot:",
    "stream": true,
    "maxTokens": 100
  }' \
  'http://localhost:9131/v1/inference'
```

---

## 📋 Providers

### List Available Providers

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/providers'
```

**Response:**
```json
{
  "providers": [
    {
      "id": "cpu",
      "name": "CPU Provider",
      "type": "native",
      "status": "available",
      "capabilities": ["text-generation", "embeddings"]
    },
    {
      "id": "metal",
      "name": "Metal Provider",
      "type": "gpu",
      "status": "available",
      "capabilities": ["text-generation", "embeddings"]
    }
  ]
}
```

### Get Provider Details

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/providers/metal'
```

---

## 🔍 System Info

### Get System Information

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/system'
```

**Response:**
```json
{
  "version": "0.1.0",
  "platform": "darwin-aarch64",
  "javaVersion": "25.0.0",
  "memoryMb": 24576,
  "cpuCores": 10,
  "backends": ["cpu", "metal"],
  "formats": ["gguf", "onnx", "safetensor", "litert"],
  "uptime": 3600
}
```

---

## 🔐 Admin

### Get Server Configuration

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/admin/config'
```

---

## 📚 Jobs Management

### Get Job Status

```bash
curl -H 'X-API-Key: community' \
  'http://localhost:9131/v1/jobs/job-12345-abcde'
```

**Response:**
```json
{
  "jobId": "job-12345-abcde",
  "type": "pull",
  "status": "in_progress",
  "progress": 75,
  "startedAt": "2024-01-15T10:30:00Z",
  "result": null
}
```

### Cancel Job

```bash
curl -X DELETE -H 'X-API-Key: community' \
  'http://localhost:9131/v1/jobs/job-12345-abcde'
```

---

## 🛠️ Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier |
| `messages` | array | Chat message history |
| `max_tokens` | integer | Max completion tokens (default: 256) |
| `temperature` | float | Sampling temperature (0.0-2.0, default: 0.7) |
| `top_p` | float | Nucleus sampling parameter (default: 1.0) |
| `top_k` | integer | Top-K sampling (default: 40) |
| `stream` | boolean | Enable streaming SSE response |
| `stop` | array | Stop sequences |
| `tools` | array | Available function definitions |
| `tool_choice` | string | Tool selection strategy: "auto", "none", specific name |

---

## 🔐 Authentication

All requests (except `/health`) require one of:

### API Key Header
```bash
curl -H 'X-API-Key: community' http://localhost:9131/v1/models
```

### Bearer Token
```bash
curl -H 'Authorization: Bearer community' http://localhost:9131/v1/models
```

---

## ⚡ Performance Tips

1. **Use streaming** for long completions to see results faster
2. **Batch embeddings** instead of single requests
3. **Limit results** with query parameters to reduce payload
4. **Use appropriate temperature**: Lower (0.3-0.5) for consistency, higher (0.8-1.2) for creativity
5. **Specify exact model** instead of querying model list multiple times

---

## 🐛 Error Handling

All errors follow this format:

```json
{
  "error": {
    "message": "Model not found",
    "type": "not_found",
    "code": 404
  }
}
```

### Common Error Codes
- `400`: Invalid request format
- `401`: Authentication failed
- `404`: Resource not found
- `429`: Rate limit exceeded
- `500`: Internal server error

---

## 📝 Testing with jq

Parse and format responses:

```bash
curl -s -H 'Accept: application/json' -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models' | jq '.data[] | {id, object}'
```

Extract nested data:

```bash
curl -s -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hi"}]}' \
  'http://localhost:9131/v1/chat/completions' | jq '.choices[0].message.content'
```

---

## 🧪 Automated Testing

Use the provided test script to validate all endpoints:

```bash
# Run all tests
bash API_TEST_SCRIPT.sh http://localhost:9131 community false

# Run with verbose output
bash API_TEST_SCRIPT.sh http://localhost:9131 community true

# Use custom API key
bash API_TEST_SCRIPT.sh http://localhost:9131 "your-api-key" false
```

**Expected output**: 8/9 tests passing (authentication test may vary based on security configuration)

---

## 🚀 Quick Start Example

Complete script to get started:

```bash
#!/bin/bash

API_KEY="community"
BASE_URL="http://localhost:9131"
HEADERS=(-H "X-API-Key: $API_KEY" -H "Accept: application/json" -H "Content-Type: application/json")

# 1. Health check
echo "1. Checking server health..."
curl -s "${HEADERS[@]}" "$BASE_URL/health" | jq .

# 2. List models
echo "2. Listing available models..."
curl -s "${HEADERS[@]}" "$BASE_URL/v1/models" | jq '.[] | {id, object}' | head -20

# 3. Simple chat
echo "3. Sending chat message..."
curl -s -X POST "${HEADERS[@]}" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello Gollek!"}]}' \
  "$BASE_URL/v1/chat/completions" | jq '.choices[0].message'
```

---

## 📋 Environment Variables Setup

For convenience, set these environment variables:

```bash
export GOLLEK_API_KEY="community"
export GOLLEK_BASE_URL="http://localhost:9131"

# Then use in curl:
curl -H "X-API-Key: $GOLLEK_API_KEY" \
  -H "Accept: application/json" \
  "$GOLLEK_BASE_URL/v1/models"
```


