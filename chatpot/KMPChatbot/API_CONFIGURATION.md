# API Configuration Guide

This guide explains how to configure the chatbot to work with different AI providers.

## Current Configuration (Anthropic Claude)

### Getting Your API Key

1. Visit [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to "API Keys"
4. Click "Create Key"
5. Copy your key (format: `sk-ant-api03-...`)

### Setting the API Key

**Location**: `shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt`

```kotlin
class ChatApiImpl(
    private val apiKey: String = "your-api-key-here" // ← Replace this
) : ChatApi {
    // ...
}
```

### API Endpoint Details

- **Base URL**: `https://api.anthropic.com/v1/messages`
- **API Version**: `2023-06-01`
- **Default Model**: `claude-sonnet-4-20250514`
- **Max Tokens**: `1024`

### Available Models

Update in `ChatRequest` (`shared/src/.../data/remote/dto/ChatDto.kt`):

```kotlin
val model: String = "MODEL_NAME"
```

**Options:**
- `claude-opus-4-20250514` - Most capable
- `claude-sonnet-4-20250514` - Balanced (default)
- `claude-haiku-4-20250514` - Fastest

---

## Alternative Provider: OpenAI

### 1. Update DTOs

**File**: `shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/dto/ChatDto.kt`

```kotlin
@Serializable
data class ChatRequest(
    val model: String = "gpt-4",
    val messages: List<MessageDto>,
    @SerialName("max_tokens")
    val maxTokens: Int? = null,
    val temperature: Double? = null
)

@Serializable
data class MessageDto(
    val role: String,
    val content: String
)

@Serializable
data class ChatResponse(
    val id: String,
    val choices: List<Choice>,
    val model: String
)

@Serializable
data class Choice(
    val message: MessageDto,
    @SerialName("finish_reason")
    val finishReason: String
)
```

### 2. Update API Client

**File**: `shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt`

```kotlin
class ChatApiImpl(
    private val apiKey: String = "your-openai-key"
) : ChatApi {
    
    private val client = HttpClient {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
            })
        }
    }
    
    override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
        return try {
            val response = client.post("https://api.openai.com/v1/chat/completions") {
                contentType(ContentType.Application.Json)
                header("Authorization", "Bearer $apiKey")
                setBody(request)
            }
            Result.success(response.body())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

### 3. Update Repository

**File**: `shared/src/commonMain/kotlin/com/kmpchatbot/data/repository/ChatRepository.kt`

```kotlin
suspend fun sendMessage(messages: List<Message>): Result<Message> {
    val messageDtos = messages.map { message ->
        MessageDto(
            role = when (message.role) {
                MessageRole.USER -> "user"
                MessageRole.ASSISTANT -> "assistant"
                MessageRole.SYSTEM -> "system"
            },
            content = message.content
        )
    }
    
    val request = ChatRequest(messages = messageDtos)
    
    return api.sendMessage(request).mapCatching { response ->
        val assistantMessage = response.choices.first().message
        Message(
            id = response.id,
            content = assistantMessage.content,
            role = MessageRole.ASSISTANT
        )
    }
}
```

---

## Alternative Provider: Google Gemini

### 1. Update DTOs

```kotlin
@Serializable
data class ChatRequest(
    val contents: List<Content>
)

@Serializable
data class Content(
    val parts: List<Part>,
    val role: String
)

@Serializable
data class Part(
    val text: String
)

@Serializable
data class ChatResponse(
    val candidates: List<Candidate>
)

@Serializable
data class Candidate(
    val content: Content
)
```

### 2. Update API Client

```kotlin
override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
    return try {
        val response = client.post(
            "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=$apiKey"
        ) {
            contentType(ContentType.Application.Json)
            setBody(request)
        }
        Result.success(response.body())
    } catch (e: Exception) {
        Result.failure(e)
    }
}
```

---

## Local/Self-Hosted Models

### Ollama Integration

```kotlin
class ChatApiImpl(
    private val baseUrl: String = "http://localhost:11434"
) : ChatApi {
    
    override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
        return try {
            val response = client.post("$baseUrl/api/chat") {
                contentType(ContentType.Application.Json)
                setBody(mapOf(
                    "model" to "llama2",
                    "messages" to request.messages,
                    "stream" to false
                ))
            }
            Result.success(response.body())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

---

## Environment Variables (Recommended)

### Setup

**macOS/Linux** - Add to `~/.bashrc` or `~/.zshrc`:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

**Windows** - Add via System Properties or PowerShell:
```powershell
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-...', 'User')
```

### Usage in Code

```kotlin
class ChatApiImpl(
    private val apiKey: String = System.getenv("ANTHROPIC_API_KEY")
        ?: throw IllegalStateException("API key not found in environment")
) : ChatApi
```

---

## Gradle Properties (Production)

### 1. Create `secrets.properties`

```properties
anthropic.api.key=sk-ant-api03-...
openai.api.key=sk-...
```

### 2. Update `.gitignore`

```
secrets.properties
```

### 3. Load in `build.gradle.kts`

```kotlin
val secretsFile = rootProject.file("secrets.properties")
val secrets = Properties()
if (secretsFile.exists()) {
    secretsFile.inputStream().use { secrets.load(it) }
}

android {
    defaultConfig {
        buildConfigField(
            "String",
            "API_KEY",
            "\"${secrets["anthropic.api.key"]}\""
        )
    }
}
```

### 4. Use in Code (Android)

```kotlin
private val apiKey: String = BuildConfig.API_KEY
```

---

## Testing Different Models

### Quick Model Comparison

Create a test script:

```kotlin
val models = listOf(
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-20250514"
)

models.forEach { model ->
    val request = ChatRequest(
        model = model,
        messages = listOf(
            MessageDto("user", "Explain quantum computing in one sentence")
        )
    )
    
    val response = api.sendMessage(request)
    println("$model: ${response.getOrNull()}")
}
```

---

## Rate Limiting & Costs

### Anthropic Claude
- **Rate Limit**: Varies by tier (check console)
- **Costs**: Per token (input + output)
- **Monitoring**: Check usage in console

### Best Practices

1. **Cache Responses**: Don't re-send same questions
2. **Limit max_tokens**: Reduce costs
3. **Implement Retry Logic**: Handle rate limits gracefully
4. **Monitor Usage**: Track API calls

### Example: Request Throttling

```kotlin
class RateLimitedApi(
    private val delegate: ChatApi,
    private val maxRequestsPerMinute: Int = 10
) : ChatApi {
    
    private val timestamps = mutableListOf<Long>()
    
    override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
        // Remove timestamps older than 1 minute
        val oneMinuteAgo = System.currentTimeMillis() - 60_000
        timestamps.removeAll { it < oneMinuteAgo }
        
        // Check if we've hit the limit
        if (timestamps.size >= maxRequestsPerMinute) {
            val waitTime = 60_000 - (System.currentTimeMillis() - timestamps.first())
            delay(waitTime)
        }
        
        timestamps.add(System.currentTimeMillis())
        return delegate.sendMessage(request)
    }
}
```

---

## Troubleshooting

### Common Errors

**401 Unauthorized**
- Check API key is correct
- Verify key has proper permissions

**429 Too Many Requests**
- Implement rate limiting
- Wait before retrying

**500 Server Error**
- API provider issue
- Retry with exponential backoff

**Network Timeout**
- Increase timeout in Ktor config:
  ```kotlin
  install(HttpTimeout) {
      requestTimeoutMillis = 30_000
  }
  ```

### Testing API Connection

```bash
# Test Anthropic
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'

# Test OpenAI
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Security Checklist

- [ ] API key not in source code
- [ ] `secrets.properties` in `.gitignore`
- [ ] Environment variables used in production
- [ ] HTTPS enforced
- [ ] Rate limiting implemented
- [ ] Error messages don't expose keys
- [ ] API key rotation plan in place

---

For more information, consult the documentation for your chosen AI provider:
- **Anthropic**: https://docs.anthropic.com/
- **OpenAI**: https://platform.openai.com/docs/
- **Google**: https://ai.google.dev/docs
