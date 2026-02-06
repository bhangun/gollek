# Client-Server Integration Guide

This guide explains how to integrate the KMP clients (Android/iOS/Desktop) with the Quarkus backend server.

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    KMP Clients                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Android  │    │   iOS    │    │ Desktop  │          │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘          │
│       │               │               │                 │
│       └───────────────┴───────────────┘                 │
│                       │                                 │
│              Shared Kotlin Code                         │
│            (Network Layer - Ktor)                       │
└───────────────────────┬─────────────────────────────────┘
                        │
                    HTTP/WS
                        │
┌───────────────────────▼─────────────────────────────────┐
│              Quarkus Backend Server                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ REST API     │  │  WebSocket   │  │  Database    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         └─────────────────┴─────────────────┘           │
└───────────────────────┬─────────────────────────────────┘
                        │
                    HTTP API
                        │
┌───────────────────────▼─────────────────────────────────┐
│              Anthropic Claude API                        │
└──────────────────────────────────────────────────────────┘
```

## 🔄 Two Integration Modes

### Mode 1: Direct API Integration (Current)
- Clients call Anthropic API directly
- No server needed
- Good for: Quick prototyping, offline-first apps

### Mode 2: Server-Based Integration (Recommended)
- Clients call Quarkus server
- Server handles AI API calls
- Server stores conversation history
- Good for: Production apps, analytics, shared conversations

## 🛠️ Switching to Server-Based Integration

### Step 1: Start the Server

```bash
cd server
./mvnw quarkus:dev
```

Server will be available at `http://localhost:8080`

### Step 2: Update Shared Module

#### Option A: REST API Integration

**Update `ChatApi.kt`**:

```kotlin
// shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt

class ChatApiImpl(
    private val baseUrl: String = "http://localhost:8080"
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
            val serverRequest = ServerChatRequest(
                message = request.messages.last().content,
                sessionId = "client-session-${Random.nextLong()}"
            )
            
            val response = client.post("$baseUrl/api/chat/message") {
                contentType(ContentType.Application.Json)
                setBody(serverRequest)
            }
            
            val serverResponse = response.body<ServerChatResponse>()
            
            Result.success(ChatResponse(
                id = serverResponse.id.toString(),
                content = listOf(ContentBlock(
                    type = "text",
                    text = serverResponse.message
                )),
                role = serverResponse.role,
                model = "claude-sonnet-4-20250514"
            ))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

@Serializable
data class ServerChatRequest(
    val message: String,
    @SerialName("session_id")
    val sessionId: String? = null
)

@Serializable
data class ServerChatResponse(
    val id: Long,
    val message: String,
    val role: String,
    @SerialName("session_id")
    val sessionId: String,
    val timestamp: String
)
```

#### Option B: WebSocket Integration

**Create `WebSocketChatApi.kt`**:

```kotlin
// shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/WebSocketChatApi.kt

class WebSocketChatApi(
    private val baseUrl: String = "ws://localhost:8080"
) {
    private val client = HttpClient {
        install(WebSockets)
        install(ContentNegotiation) {
            json()
        }
    }
    
    suspend fun connect(
        onMessage: (ChatResponse) -> Unit,
        onError: (Throwable) -> Unit
    ) {
        try {
            client.webSocket(
                method = HttpMethod.Get,
                host = "localhost",
                port = 8080,
                path = "/ws/chat"
            ) {
                // Listen for incoming messages
                for (frame in incoming) {
                    frame as? Frame.Text ?: continue
                    val text = frame.readText()
                    
                    try {
                        val response = Json.decodeFromString<ServerChatResponse>(text)
                        
                        onMessage(ChatResponse(
                            id = response.id.toString(),
                            content = listOf(ContentBlock("text", response.message)),
                            role = response.role,
                            model = "server"
                        ))
                    } catch (e: Exception) {
                        println("Error parsing message: ${e.message}")
                    }
                }
            }
        } catch (e: Exception) {
            onError(e)
        }
    }
    
    suspend fun sendMessage(session: DefaultClientWebSocketSession, message: String) {
        val request = ServerChatRequest(
            message = message,
            sessionId = "ws-session"
        )
        
        val json = Json.encodeToString(request)
        session.send(Frame.Text(json))
    }
}
```

### Step 3: Update ViewModel

**Update `ChatViewModel.kt`** to use server:

```kotlin
class ChatViewModel(
    private val repository: ChatRepository = ChatRepository(
        api = ChatApiImpl(baseUrl = "http://localhost:8080")
    ),
    private val scope: CoroutineScope
) {
    // Rest of the code remains the same
}
```

### Step 4: Platform-Specific Configuration

#### Android

Update base URL based on environment:

```kotlin
// androidApp/src/main/kotlin/.../MainActivity.kt

private fun getServerUrl(): String {
    return if (BuildConfig.DEBUG) {
        // For emulator
        "http://10.0.2.2:8080"
    } else {
        // Production server
        "https://api.yourdomain.com"
    }
}

// Then pass to ViewModel
val viewModel: ChatViewModel = viewModel { 
    ChatViewModel(
        repository = ChatRepository(
            api = ChatApiImpl(baseUrl = getServerUrl())
        ),
        scope = scope
    ) 
}
```

#### iOS

```swift
// iosApp/iosApp/ContentView.swift

private func getServerUrl() -> String {
    #if DEBUG
    return "http://localhost:8080"
    #else
    return "https://api.yourdomain.com"
    #endif
}
```

#### Desktop

```kotlin
// desktopApp/src/main/kotlin/.../Main.kt

private fun getServerUrl(): String {
    return System.getProperty("server.url") ?: "http://localhost:8080"
}
```

## 🔐 Production Configuration

### 1. HTTPS/WSS

For production, use HTTPS and WSS:

```kotlin
class ChatApiImpl(
    private val baseUrl: String = "https://api.yourdomain.com"
) : ChatApi {
    // ...
}

class WebSocketChatApi(
    private val baseUrl: String = "wss://api.yourdomain.com"
) {
    // ...
}
```

### 2. Authentication

Add authentication headers:

```kotlin
override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
    return try {
        val response = client.post("$baseUrl/api/chat/message") {
            contentType(ContentType.Application.Json)
            header("Authorization", "Bearer $authToken")
            setBody(serverRequest)
        }
        // ...
    } catch (e: Exception) {
        Result.failure(e)
    }
}
```

### 3. Session Management

Store session ID persistently:

```kotlin
// Use platform-specific storage
expect class SessionStorage {
    fun saveSessionId(sessionId: String)
    fun getSessionId(): String?
}

class ChatRepository(
    private val api: ChatApi,
    private val sessionStorage: SessionStorage
) {
    
    suspend fun sendMessage(messages: List<Message>): Result<Message> {
        val sessionId = sessionStorage.getSessionId() 
            ?: UUID.randomUUID().toString().also {
                sessionStorage.saveSessionId(it)
            }
        
        // Use sessionId in request
        // ...
    }
}
```

## 📱 Feature Comparison

| Feature | Direct API | Server-Based |
|---------|-----------|--------------|
| Offline Support | Limited | Better (cached history) |
| Conversation History | Client-only | Persistent in DB |
| Multi-device Sync | No | Yes (via session ID) |
| Analytics | No | Yes (server-side) |
| Cost Control | Per-client | Centralized |
| Latency | Lower | Slightly higher |
| Complexity | Lower | Higher |

## 🧪 Testing Both Modes

### Test Direct API Mode

```bash
# Start only the client apps
./gradlew :desktopApp:run
```

### Test Server Mode

```bash
# Terminal 1: Start server
cd server
./mvnw quarkus:dev

# Terminal 2: Start client
./gradlew :desktopApp:run
```

## 🔄 Hybrid Approach

You can support both modes with a configuration flag:

```kotlin
class ChatViewModel(
    private val useServer: Boolean = true,
    private val scope: CoroutineScope
) {
    private val repository = if (useServer) {
        ChatRepository(api = ChatApiImpl(baseUrl = "http://localhost:8080"))
    } else {
        ChatRepository(api = DirectAnthropicApiImpl())
    }
    
    // ...
}
```

## 📊 Monitoring & Debugging

### Check Server Health

```bash
curl http://localhost:8080/q/health
```

### View Swagger API

Open: `http://localhost:8080/swagger-ui`

### Test WebSocket

```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8080/ws/chat');

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({message: 'Test'}));
};

ws.onmessage = (event) => {
  console.log('Received:', event.data);
};
```

### Server Logs

```bash
# In dev mode
# Logs appear in terminal automatically

# In production
tail -f server/logs/application.log
```

## 🚀 Deployment

### Deploy Server

```bash
# Build
cd server
./mvnw clean package

# Deploy to cloud
# - Heroku: git push heroku main
# - AWS: Use Elastic Beanstalk
# - Docker: docker-compose up
```

### Update Client URLs

Change base URLs in client apps to point to deployed server:

```kotlin
private val baseUrl = "https://your-app.herokuapp.com"
```

## 💡 Best Practices

1. **Use Server for Production**: Better control, analytics, and cost management
2. **Environment-Based URLs**: Different URLs for dev/staging/prod
3. **Error Handling**: Handle network errors gracefully
4. **Retry Logic**: Implement exponential backoff for failed requests
5. **Caching**: Cache responses on client for better offline experience
6. **Session Persistence**: Save session IDs for conversation continuity
7. **Security**: Always use HTTPS/WSS in production
8. **Monitoring**: Monitor server health and API usage

## 🔧 Troubleshooting

**Connection Refused**:
- Check server is running: `curl http://localhost:8080/q/health`
- Check firewall settings
- For Android emulator: Use `10.0.2.2` instead of `localhost`

**WebSocket Connection Failed**:
- Verify WebSocket endpoint: `ws://localhost:8080/ws/chat`
- Check CORS settings in server
- Ensure no proxy blocking WebSocket

**Session Not Persisting**:
- Verify session ID is being sent
- Check database for stored sessions
- Review server logs for errors

---

**You now have a complete client-server architecture!** 🎉
