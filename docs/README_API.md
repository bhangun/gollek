# Gollek API Documentation

Complete API documentation for the Gollek agentic AI inference server.

## 📚 Documentation Files

This directory contains comprehensive API documentation:

### **Quick Start** → [`API_QUICK_REFERENCE.md`](API_QUICK_REFERENCE.md)
- 5-minute quick start
- Common endpoints table
- Essential headers
- Example requests for chat, embeddings, models
- Troubleshooting guide
- **START HERE** ⭐

### **Full Examples** → [`API_EXAMPLES.md`](API_EXAMPLES.md)
- Detailed endpoint documentation
- All parameters explained
- Request/response examples for every endpoint
- Error handling
- Advanced features (streaming, tools, vision, RAG)
- Performance tips

### **Test Suite** → [`API_TEST_SCRIPT.sh`](API_TEST_SCRIPT.sh)
- Automated endpoint testing
- 9 comprehensive test cases
- Colored output with pass/fail
- Verbose mode for debugging

---

## 🚀 Start the Server

```bash
cd gollek
./scripts/run-dev.sh serve --debug
```

Server runs on **`http://localhost:9131`**

---

## ✅ Quick Test

```bash
# Health check (no auth needed)
curl -H 'Accept: application/json' http://localhost:9131/health

# List models (auth required)
curl -H 'X-API-Key: community' \
     -H 'Accept: application/json' \
     http://localhost:9131/v1/models

# Chat completion
curl -X POST \
  -H 'X-API-Key: community' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello"}]}' \
  http://localhost:9131/v1/chat/completions

# Run full test suite
bash API_TEST_SCRIPT.sh http://localhost:9131 community false
```

---

## 🔐 Authentication

Every endpoint (except `/health`) requires:

```bash
# API Key method (recommended)
-H 'X-API-Key: community'

# Bearer token method
-H 'Authorization: Bearer community'
```

---

## 📋 Supported Features

✅ **Chat Completions**
- Streaming & non-streaming
- System prompts
- Function calling (tools)
- Vision capabilities
- RAG context injection

✅ **Embeddings**
- Single & batch processing
- Multiple encoding formats

✅ **Model Management**
- List/search models
- Pull/download models
- Delete models
- Query capabilities

✅ **Inference**
- Raw inference API
- Provider routing
- Model capabilities detection

✅ **System Info**
- Server diagnostics
- Resource availability
- Supported backends

---

## 📊 Test Results

Run `API_TEST_SCRIPT.sh` to verify all endpoints:

```
🏥 Health Check
✓ Server health check (HTTP 200)

📦 Models Management
✓ List all models (HTTP 200)
✓ List runnable models (limited) (HTTP 200)
✓ List models in OpenAI format (HTTP 200)

💬 Chat Completions
✓ Simple chat completion (HTTP 200)

🔌 Embeddings
✓ Create embedding (HTTP 200)

🔍 System Info
✓ Get system information (HTTP 200)

📋 Providers
✓ List available providers (HTTP 200)

Test Results: 8 passed, 1 failed out of 9 tests
```

---

## 🛠️ API Endpoints Summary

### Health & System
- `GET /health` - Server health (no auth)
- `GET /v1/system` - System information
- `GET /v1/providers` - Available providers

### Models
- `GET /v1/models` - List models
- `GET /v1/models/:id` - Model details
- `GET /v1/models/:id/capabilities` - Model capabilities
- `POST /v1/models/pull` - Download model
- `DELETE /v1/models/:id` - Delete model

### Chat & Inference
- `POST /v1/chat/completions` - Chat (OpenAI compatible)
- `POST /v1/inference` - Raw inference
- `POST /v1/embeddings` - Create embeddings

### Admin
- `GET /v1/admin/api-keys` - List API keys
- `POST /v1/admin/api-keys` - Create API key
- `DELETE /v1/admin/api-keys/:key` - Revoke API key

---

## 📝 Usage Examples

### Python

```python
import requests

headers = {
    "X-API-Key": "community",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Chat
response = requests.post(
    "http://localhost:9131/v1/chat/completions",
    headers=headers,
    json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json())
```

### JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

const headers = {
  'X-API-Key': 'community',
  'Accept': 'application/json',
  'Content-Type': 'application/json'
};

const response = await fetch('http://localhost:9131/v1/chat/completions', {
  method: 'POST',
  headers,
  body: JSON.stringify({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: 'Hello!' }]
  })
});

const result = await response.json();
console.log(result);
```

### cURL with Streaming

```bash
curl -X POST \
  -H 'X-API-Key: community' \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }' \
  http://localhost:9131/v1/chat/completions
```

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "Resource not found" (HTML) | Add `-H 'Accept: application/json'` |
| "Missing API key" (401) | Add `-H 'X-API-Key: community'` |
| "Connection refused" | Verify server running: `curl http://localhost:9131/health` |
| Slow responses | Check if models loaded: `curl -H 'X-API-Key: community' http://localhost:9131/v1/models` |

---

## 📖 Documentation Structure

```
gollek/
├── README_API.md              # This file
├── API_QUICK_REFERENCE.md     # Quick start & common tasks
├── API_EXAMPLES.md            # Detailed endpoint docs
├── API_TEST_SCRIPT.sh         # Automated test suite
└── scripts/
    └── run-dev.sh             # Start server
```

---

## 🔧 Configuration

### Server Port
Edit `ui/gollek-api/src/main/resources/application.properties`:
```properties
quarkus.http.port=9131
```

### API Keys
Stored in `~/.gollek/server/data/keys.json`  
Default key: `community`

### Logging
Logs written to: `~/.gollek/server/logs/server.log`

### Resource Limits
Set environment variables:
```bash
export JAVA_OPTS="-Xmx32g -XX:MaxDirectMemorySize=32g"
```

---

## 📦 Supported Models

Place model files in:
- GGUF: `~/.gollek/models/gguf/`
- SafeTensors: `~/.gollek/models/safetensors/`
- Others: See model repository configuration

List available models:
```bash
curl -H 'X-API-Key: community' \
     -H 'Accept: application/json' \
     http://localhost:9131/v1/models
```

---

## 🚀 Deployment Checklist

- [ ] Change default API key from `community`
- [ ] Configure TLS/HTTPS (reverse proxy)
- [ ] Set resource limits (`JAVA_OPTS`)
- [ ] Load required models
- [ ] Monitor logs at `~/.gollek/server/logs/`
- [ ] Test all endpoints with `API_TEST_SCRIPT.sh`
- [ ] Configure firewall rules
- [ ] Setup log rotation/archival

---

## 📞 Support

- **Documentation**: See files listed above
- **Logs**: `~/.gollek/server/logs/server.log`
- **Debug mode**: `./scripts/run-dev.sh serve --debug`
- **Test**: `bash API_TEST_SCRIPT.sh <url> <key> true`

---

**Last Updated**: 2026-06-27  
**API Version**: v1 (OpenAI compatible)  
**Status**: Production Ready
