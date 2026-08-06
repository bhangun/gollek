# Gollek API Quick Reference

## Overview

Gollek exposes an **OpenAI-compatible REST API** for:
- Chat completions (streaming & non-streaming)
- Embeddings
- Model management
- Provider information
- System diagnostics

---

## 🚀 Getting Started (5 minutes)

### 1. Start the Server

```bash
cd gollek
./scripts/run-dev.sh serve --debug
```

Server starts on: `http://localhost:9131`

### 2. Test Health

```bash
curl -H 'Accept: application/json' http://localhost:9131/health
# Response: {"status":"ok"}
```

### 3. List Models

```bash
curl -H 'X-API-Key: community' \
     -H 'Accept: application/json' \
     http://localhost:9131/v1/models
```

### 4. Send Chat Message

```bash
curl -X POST \
  -H 'X-API-Key: community' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }' \
  http://localhost:9131/v1/chat/completions
```

---

## 📚 Essential Headers

Every request (except `/health`) requires **one of**:

```bash
# Option 1: API Key Header (Recommended)
-H 'X-API-Key: community'

# Option 2: Bearer Token
-H 'Authorization: Bearer community'
```

**Always include Accept header for JSON responses:**
```bash
-H 'Accept: application/json'
```

---

## 🔑 Common Endpoints

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/health` | GET | Server health | No |
| `/v1/models` | GET | List models | Yes |
| `/v1/models/:id` | GET | Model details | Yes |
| `/v1/chat/completions` | POST | Chat completion | Yes |
| `/v1/embeddings` | POST | Create embeddings | Yes |
| `/v1/providers` | GET | List providers | Yes |
| `/v1/system` | GET | System info | Yes |

---

## 💬 Chat Completions

### Non-Streaming Request

```bash
curl -X POST \
  -H 'X-API-Key: community' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }' \
  http://localhost:9131/v1/chat/completions
```

### Streaming Request

```bash
curl -X POST \
  -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }' \
  http://localhost:9131/v1/chat/completions
```

Response is **Server-Sent Events** format:
```
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" there"}}]}
data: [DONE]
```

---

## 🔌 Embeddings

```bash
curl -X POST \
  -H 'X-API-Key: community' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox"
  }' \
  http://localhost:9131/v1/embeddings
```

**Batch embeddings:**
```bash
curl -X POST \
  -H 'X-API-Key: community' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "text-embedding-ada-002",
    "input": ["Text 1", "Text 2", "Text 3"]
  }' \
  http://localhost:9131/v1/embeddings
```

---

## 📊 Response Format

### Success Response (200)
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1699107661,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop",
      "index": 0
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

### Error Response (4xx/5xx)
```json
{
  "error": {
    "message": "Model not found",
    "type": "not_found",
    "code": 404
  }
}
```

---

## ⚙️ Query Parameters

### `/v1/models` Query Parameters
- `limit` (int, default 50) - Max results to return
- `runnableOnly` (boolean) - Only return loadable models
- `compat` (string) - `openai` for OpenAI-compatible format
- `namespace` (string) - Filter by model namespace

Examples:
```bash
# List 10 models
/v1/models?limit=10

# Only runnable models
/v1/models?runnableOnly=true

# OpenAI format
/v1/models?compat=openai
```

---

## 🛠️ Common Parameter Reference

For chat completions:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | string | - | - | Model identifier |
| `messages` | array | - | - | Message history |
| `temperature` | float | 0.7 | 0.0-2.0 | Sampling temperature |
| `max_tokens` | int | 256 | 1-4096 | Max response tokens |
| `top_p` | float | 1.0 | 0.0-1.0 | Nucleus sampling |
| `top_k` | int | 40 | 1-1000 | Top-K sampling |
| `stream` | boolean | false | - | Stream responses |
| `stop` | array | - | - | Stop sequences |
| `tools` | array | - | - | Available functions |

---

## 🔍 Useful Tools

### Using jq to parse responses:

```bash
# Extract first choice content
curl -s -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models' | jq '.data[] | .id' | head -5

# Pretty print
curl -s -X POST -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hi"}]}' \
  'http://localhost:9131/v1/chat/completions' | jq '.'
```

### Measure response time:

```bash
curl -w "Time: %{time_total}s\n" \
  -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models'
```

### Save response to file:

```bash
curl -O -H 'X-API-Key: community' \
  'http://localhost:9131/v1/models' -o models.json
```

---

## 📝 Example: Python Client

```python
import requests
import json

API_KEY = "community"
BASE_URL = "http://localhost:9131"
headers = {
    "X-API-Key": API_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# List models
response = requests.get(f"{BASE_URL}/v1/models", headers=headers)
models = response.json()
print(f"Available models: {len(models)}")

# Send chat message
payload = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
}
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    headers=headers,
    json=payload
)
result = response.json()
print(result['choices'][0]['message']['content'])
```

---

## 🐛 Troubleshooting

### Issue: "Resource not found" (HTML response)

**Solution**: Add `Accept: application/json` header
```bash
# Wrong
curl http://localhost:9131/v1/models

# Correct
curl -H 'Accept: application/json' http://localhost:9131/v1/models
```

### Issue: "Invalid API key" (403)

**Solution**: Verify API key
```bash
# Check default key
curl -H 'X-API-Key: community' -H 'Accept: application/json' \
  http://localhost:9131/v1/models
```

### Issue: Connection refused

**Solution**: Verify server is running
```bash
# Check port
netstat -anv | grep 9131

# Check health
curl http://localhost:9131/health
```

### Issue: Slow responses

**Solution**: Models may not be loaded
- Check `/v1/models` - should list available models
- Load a model: `/v1/models/pull` endpoint
- Use `/v1/system` to check available resources

---

## 📖 Full API Documentation

See `API_EXAMPLES.md` for comprehensive endpoint documentation with response examples.

Run automated tests:
```bash
bash API_TEST_SCRIPT.sh http://localhost:9131 community false
```

---

## 🚀 Deployment

### Production Setup

1. Change default API key:
```bash
export GOLLEK_API_KEYS="your-secure-key-1,your-secure-key-2"
```

2. Set resources:
```bash
export JAVA_OPTS="-Xmx32g -XX:MaxDirectMemorySize=32g"
```

3. Use reverse proxy (nginx):
```nginx
server {
    listen 443 ssl;
    server_name api.example.com;
    
    location / {
        proxy_pass http://localhost:9131;
        proxy_set_header X-API-Key $http_x_api_key;
    }
}
```

---

## 📞 Support

- **Logs**: `~/.gollek/server/logs/server.log`
- **Issue reporting**: Check logs, include curl output with `-v` flag
- **Rate limiting**: Not implemented yet (coming soon)

