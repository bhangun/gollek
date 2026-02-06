# Server Integration Guide

This guide explains how to integrate the KMP mobile/desktop apps with the Quarkus backend server.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                       │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   Android    │     iOS      │   Desktop    │   Web (Future)│
│  (Compose)   │  (SwiftUI)   │  (Compose)   │               │
└──────┬───────┴──────┬───────┴──────┬───────┴───────┬───────┘
       │              │              │               │
       └──────────────┴──────────────┴───────────────┘
                      │
                      ▼
       ┌──────────────────────────────────┐
       │      HTTP/REST + WebSocket        │
       └──────────────┬───────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────┐
       │      QUARKUS SERVER               │
       │  ┌────────────────────────────┐  │
       │  │   REST API Endpoints       │  │
       │  │   - /api/chat              │  │
       │  │   - /api/conversations     │  │
       │  ├────────────────────────────┤  │
       │  │   WebSocket Endpoint       │  │
       │  │   - /ws/chat/{userId}      │  │
       │  ├────────────────────────────┤  │
       │  │   Business Logic           │  │
       │  │   - ChatService            │  │
       │  │   - AIService              │  │
       │  ├────────────────────────────┤  │
       │  │   Database (PostgreSQL)    │  │
       │  │   - Users                  │  │
       │  │   - Conversations          │  │
       │  │   - Messages               │  │
       │  └────────────────────────────┘  │
       └──────────────┬───────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────┐
       │   ANTHROPIC CLAUDE API            │
       │   (External AI Service)           │
       └──────────────────────────────────┘
```

## 🔄 Integration Modes

### Mode 1: Direct AI API (Current)

**Client → AI API directly**

- Clients call Anthropic API directly
- No server needed
- Simpler setup
- No conversation persistence

**Use when:**
- Prototyping
- Testing
- Single-user app
- No backend requirements

### Mode 2: Server-Backend (Recommended)

**Client → Server → AI API**

- Clients call Quarkus server
- Server manages AI API calls
- Full conversation persistence
- User authentication
- Better error handling
- Usage tracking

**Use when:**
- Multi-user app
- Need conversation history
- Want centralized control
- Production deployment

## 📝 Updating Client to Use Server

### Step 1: Update Shared Module API

**File**: `shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt`

```kotlin
class ChatApiImpl(
    private val baseUrl: String = "http://localhost:8080" // Server URL
) : ChatApi {
    
    private val client = HttpClient {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
            })
        }
        install(Logging) {
            level = LogLevel.INFO
        }
    }
    
    override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
        return try {
            // Call server instead of AI API directly
            val response = client.post("$baseUrl/api/chat/message") {
                contentType(ContentType.Application.Json)
                setBody(mapOf(
                    "content" to request.messages.last().content,
                    "conversationId" to null // Or track conversation ID
                ))
            }
            
            // Parse server response
            val serverResponse: ServerChatResponse = response.body()
            
            // Convert to existing ChatResponse format
            Result.success(ChatResponse(
                id = serverResponse.assistantMessage.id.toString(),
                content = listOf(ContentBlock(
                    type = "text",
                    text = serverResponse.assistantMessage.content
                )),
                role = "assistant",
                model = serverResponse.assistantMessage.modelUsed ?: "unknown"
            ))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

// Server response DTOs
@Serializable
data class ServerChatResponse(
    val userMessage: ServerMessage,
    val assistantMessage: ServerMessage,
    val conversationId: Long
)

@Serializable
data class ServerMessage(
    val id: Long,
    val content: String,
    val role: String,
    val timestamp: String,
    val modelUsed: String? = null
)
```

### Step 2: Add Server Configuration

**File**: `shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ServerConfig.kt`

```kotlin
object ServerConfig {
    // Development
    const val DEV_BASE_URL = "http://localhost:8080"
    
    // Production
    const val PROD_BASE_URL = "https://api.yourapp.com"
    
    // Android Emulator (special case)
    const val ANDROID_EMULATOR_URL = "http://10.0.2.2:8080"
    
    fun getBaseUrl(isDebug: Boolean = true): String {
        return if (isDebug) DEV_BASE_URL else PROD_BASE_URL
    }
}
```

### Step 3: Update Android to Use Server

**File**: `androidApp/src/main/kotlin/com/kmpchatbot/android/MainActivity.kt`

Add network security config for development:

**Create**: `androidApp/src/main/res/xml/network_security_config.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">localhost</domain>
        <domain includeSubdomains="true">10.0.2.2</domain>
    </domain-config>
</network-security-config>
```

**Update**: `androidApp/src/main/AndroidManifest.xml`

```xml
<application
    ...
    android:networkSecurityConfig="@xml/network_security_config">
```

### Step 4: WebSocket Integration (Optional)

For real-time chat updates:

**File**: `shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatWebSocket.kt`

```kotlin
expect class WebSocketClient {
    fun connect(url: String, onMessage: (String) -> Unit)
    fun send(message: String)
    fun disconnect()
}

// Android implementation
actual class WebSocketClient {
    private var webSocket: OkHttpWebSocket? = null
    
    actual fun connect(url: String, onMessage: (String) -> Unit) {
        val client = OkHttpClient()
        val request = Request.Builder().url(url).build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, text: String) {
                onMessage(text)
            }
        })
    }
    
    actual fun send(message: String) {
        webSocket?.send(message)
    }
    
    actual fun disconnect() {
        webSocket?.close(1000, "Client closed")
    }
}
```

## 🚀 Deployment Scenarios

### Scenario 1: Local Development

**Server**: Run locally
```bash
cd serverApp
./mvnw quarkus:dev
```

**Client**: Point to localhost
```kotlin
const val BASE_URL = "http://localhost:8080"
```

### Scenario 2: Cloud Deployment

**Server**: Deploy to cloud
```bash
# Build and push Docker image
docker build -t yourname/chatbot-server .
docker push yourname/chatbot-server

# Deploy to cloud platform
kubectl apply -f kubernetes-deployment.yaml
```

**Client**: Point to cloud URL
```kotlin
const val BASE_URL = "https://api.yourapp.com"
```

### Scenario 3: Hybrid (Development)

**Option A**: Server on cloud, clients local
- Server: Deployed to Heroku/Railway/etc.
- Clients: Local development

**Option B**: Everything local
- Server: Local Quarkus dev mode
- Clients: Local emulators/simulators

## 🔐 Authentication Flow

### 1. Login Endpoint

```kotlin
// Client calls login
suspend fun login(username: String, password: String): Result<String> {
    return try {
        val response = client.post("$baseUrl/api/auth/login") {
            setBody(mapOf(
                "username" to username,
                "password" to password
            ))
        }
        val loginResponse: LoginResponse = response.body()
        Result.success(loginResponse.token)
    } catch (e: Exception) {
        Result.failure(e)
    }
}
```

### 2. Store Token

```kotlin
// Platform-specific storage
expect class SecureStorage {
    fun saveToken(token: String)
    fun getToken(): String?
    fun clearToken()
}
```

### 3. Use Token in Requests

```kotlin
override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
    val token = secureStorage.getToken()
    
    return try {
        val response = client.post("$baseUrl/api/chat/message") {
            header("Authorization", "Bearer $token")
            setBody(request)
        }
        Result.success(response.body())
    } catch (e: Exception) {
        Result.failure(e)
    }
}
```

## 📊 API Comparison

### Direct AI API
```http
POST https://api.anthropic.com/v1/messages
Headers:
  x-api-key: sk-ant-...
  anthropic-version: 2023-06-01
Body:
{
  "model": "claude-sonnet-4",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1024
}
```

### Server API
```http
POST http://localhost:8080/api/chat/message
Headers:
  Content-Type: application/json
  Authorization: Bearer eyJ... (optional)
Body:
{
  "content": "Hello",
  "conversationId": null
}
```

**Benefits of Server API:**
- Simpler client code
- Conversation tracking automatic
- User authentication handled
- API key hidden from clients
- Usage monitoring
- Rate limiting
- Caching possibilities

## 🧪 Testing Integration

### Test Server is Running

```bash
curl http://localhost:8080/api/chat/health
```

### Test from Android Emulator

```bash
# Server URL for Android Emulator
curl http://10.0.2.2:8080/api/chat/health
```

### Test from iOS Simulator

```bash
# Server URL for iOS Simulator
curl http://localhost:8080/api/chat/health
```

### Test WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/chat/1');

ws.onopen = () => {
    ws.send(JSON.stringify({
        content: 'Hello from client!',
        conversationId: null
    }));
};

ws.onmessage = (event) => {
    console.log('Received:', JSON.parse(event.data));
};
```

## 🔧 Troubleshooting

### Client can't connect to server

**Problem**: Network errors, timeouts

**Solutions**:
1. Check server is running: `curl http://localhost:8080/q/health/live`
2. For Android emulator, use `10.0.2.2` instead of `localhost`
3. Add network security config for cleartext (dev only)
4. Check firewall settings

### CORS errors in web clients

**Problem**: Cross-origin requests blocked

**Solution**: CORS is enabled in server by default
```properties
quarkus.http.cors=true
quarkus.http.cors.origins=*
```

### Authentication failures

**Problem**: 401 Unauthorized

**Solutions**:
1. Check token is being sent in header
2. Verify token hasn't expired
3. Check JWT configuration matches

### Database connection errors

**Problem**: Server can't connect to PostgreSQL

**Solutions**:
1. Verify PostgreSQL is running
2. Check connection string in `application.properties`
3. For Docker: ensure containers are on same network

## 📋 Migration Checklist

- [ ] Server running and accessible
- [ ] Updated ChatApi to use server URL
- [ ] Network security config (Android)
- [ ] App Transport Security (iOS)
- [ ] Authentication implemented
- [ ] Token storage implemented
- [ ] Error handling updated
- [ ] Conversation persistence working
- [ ] WebSocket connected (if using)
- [ ] Tested on all platforms
- [ ] Production URLs configured
- [ ] Environment variables set

## 🎯 Best Practices

1. **Use environment variables** for URLs
2. **Implement proper error handling** for network failures
3. **Add retry logic** for failed requests
4. **Cache responses** when appropriate
5. **Handle authentication** securely
6. **Test on real devices** not just emulators
7. **Monitor API usage** and costs
8. **Implement offline mode** when possible
9. **Use WebSocket** for real-time features
10. **Keep server and clients in sync** on data models

## 🚀 Next Steps

1. **Start the server**: `cd serverApp && ./mvnw quarkus:dev`
2. **Update client code**: Modify ChatApi to use server
3. **Test integration**: Run clients and verify server communication
4. **Add features**: Implement authentication, WebSocket, etc.
5. **Deploy**: Move to production when ready

---

For more details, see:
- `serverApp/README.md` - Server documentation
- `README.md` - Main project documentation
- `SETUP_GUIDE.md` - Setup instructions
