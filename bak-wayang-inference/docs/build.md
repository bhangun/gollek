
/*
==============================================================================
BUILD AND RUN
==============================================================================

# 1. Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make libllama.so  # or libllama.dylib on macOS

# 2. Download a GGUF model
# Example: Llama-2-7B-chat
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# 3. Update application.yml with correct paths

# 4. Build and run
mvn clean package
java --enable-native-access=ALL-UNNAMED -jar target/quarkus-app/quarkus-run.jar

# Or development mode
mvn quarkus:dev --enable-native-access=ALL-UNNAMED

==============================================================================
API EXAMPLES
==============================================================================

# Non-streaming chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a haiku about programming"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'

# Streaming chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "max_tokens": 200,
    "stream": true
  }' \
  --no-buffer

# List models
curl http://localhost:8080/v1/models

==============================================================================
FEATURES
==============================================================================

✅ Full FFM API integration (no JNI)
✅ Streaming and non-streaming responses
✅ Advanced sampling (temperature, top-k, top-p, min-p, repeat penalty)
✅ Proper KV cache management
✅ Batch processing for efficient inference
✅ Multi-turn conversation support
✅ Stop strings support
✅ Configurable context size and GPU layers
✅ OpenAI-compatible API
✅ Production-ready error handling
✅ Memory-safe with Arena management
✅ Thread-safe inference with synchronization
✅ Automatic context cleanup

==============================================================================
NOTES
==============================================================================

- Uses JDK 25 FFM API for zero-copy native memory access
- Proper struct layouts for llama.cpp data structures
- Advanced sampling with all major strategies
- Full KV cache control for conversation management
- Optimized batch processing for throughput
- Compatible with OpenAI API clients
- Ready for production deployment with Quarkus

**/