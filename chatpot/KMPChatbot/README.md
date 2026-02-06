# KMP Chatbot - Kotlin Multiplatform Chat Application

A modern, production-ready chatbot application built with **Kotlin Multiplatform (KMP)** that runs on Android, iOS, and Desktop platforms with shared business logic.

## 🚀 Features

- **Kotlin Multiplatform**: Share business logic across all platforms
- **Real AI Integration**: Uses Anthropic's Claude API for intelligent responses
- **Backend Server**: Production-ready Quarkus server with REST API and WebSocket support
- **Modern UI**: 
  - Android: Jetpack Compose with Material 3
  - iOS: SwiftUI
  - Desktop: Compose Desktop
- **Reactive Architecture**: StateFlow for reactive state management
- **Database Persistence**: Store chat history (server-side with PostgreSQL/H2)
- **Production Ready**: Proper error handling, loading states, and user feedback
- **Network Layer**: Ktor client for cross-platform HTTP requests

## 🗂️ Project Structure

```
KMPChatbot/
├── shared/                          # Shared Kotlin code
│   └── src/
│       ├── commonMain/              # Platform-agnostic code
│       │   └── kotlin/com/kmpchatbot/
│       │       ├── data/
│       │       │   ├── remote/      # API client & DTOs
│       │       │   └── repository/  # Data repository
│       │       ├── domain/
│       │       │   └── model/       # Domain models
│       │       └── presentation/    # ViewModel
│       ├── androidMain/             # Android-specific code
│       ├── iosMain/                 # iOS-specific code
│       └── desktopMain/             # Desktop-specific code
├── androidApp/                      # Android application
│   └── src/main/
│       ├── kotlin/                  # Compose UI
│       └── AndroidManifest.xml
├── iosApp/                          # iOS application
│   └── iosApp/
│       └── ContentView.swift        # SwiftUI UI
├── desktopApp/                      # Desktop application
│   └── src/main/kotlin/             # Compose Desktop UI
└── server/                          # ⭐ Quarkus Backend Server
    ├── src/main/java/com/kmpchatbot/server/
    │   ├── domain/                  # JPA entities
    │   ├── repository/              # Data access
    │   ├── service/                 # Business logic
    │   ├── resource/                # REST endpoints
    │   ├── client/                  # AI API client
    │   └── websocket/               # WebSocket support
    ├── pom.xml                      # Maven configuration
    └── SERVER_README.md             # Server documentation
```

## 🛠️ Tech Stack

### Shared Module
- **Kotlin Multiplatform**: 1.9.21
- **Kotlinx Coroutines**: For async operations
- **Kotlinx Serialization**: JSON serialization
- **Ktor Client**: HTTP networking
  - Android: Android engine
  - iOS: Darwin engine
  - Desktop: CIO engine

### Android App
- **Jetpack Compose**: Modern declarative UI
- **Material 3**: Latest Material Design
- **Lifecycle**: ViewModel and Compose integration
- **Min SDK**: 24 (Android 7.0)
- **Target SDK**: 34

### iOS App
- **SwiftUI**: Native declarative UI
- **Shared Framework**: Kotlin/Native framework
- **Min iOS**: 14.0+

### Desktop App
- **Compose Desktop**: JetBrains Compose for Desktop
- **Packaging**: DMG (macOS), MSI (Windows), DEB (Linux)

## 📋 Prerequisites

- **JDK 17** or higher
- **Android Studio**: Hedgehog or newer (for Android)
- **Xcode 14+**: (for iOS, macOS only)
- **Gradle 8.0+**: (usually comes with Android Studio)

## 🔧 Setup & Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd KMPChatbot
```

### 2. Configure API Key

Open `shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt` and replace:

```kotlin
private val apiKey: String = "your-api-key-here"
```

With your actual Anthropic API key. Get one at: https://console.anthropic.com/

**Security Note**: For production, use environment variables or secure key storage:

```kotlin
// Better approach - read from environment or config
private val apiKey: String = System.getenv("ANTHROPIC_API_KEY") 
    ?: "your-fallback-key"
```

### 3. Build & Run

#### Android

```bash
# Via command line
./gradlew :androidApp:installDebug

# Or in Android Studio:
# 1. Open project in Android Studio
# 2. Select 'androidApp' configuration
# 3. Click Run
```

#### iOS (macOS only)

```bash
# Generate Xcode project
cd iosApp
open iosApp.xcodeproj

# Or build from command line
xcodebuild -project iosApp.xcodeproj -scheme iosApp -configuration Debug
```

#### Desktop

```bash
# Run desktop app
./gradlew :desktopApp:run

# Create distribution
./gradlew :desktopApp:packageDistributionForCurrentOS

# Output will be in: desktopApp/build/compose/binaries/main/
```

## 🏗️ Architecture

### Clean Architecture Layers

1. **Domain Layer** (`domain/model`)
   - Pure Kotlin data classes
   - No platform dependencies
   - Business entities: `Message`, `ChatState`

2. **Data Layer** (`data/`)
   - Repository pattern
   - API client abstraction
   - DTOs for serialization

3. **Presentation Layer** (`presentation/`)
   - Platform-agnostic `ChatViewModel`
   - StateFlow for reactive updates
   - Pure business logic, no UI

### State Management

```kotlin
data class ChatState(
    val messages: List<Message> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,
    val currentInput: String = ""
)
```

The ViewModel exposes a `StateFlow<ChatState>` that UI observes:
- **Android**: `collectAsStateWithLifecycle()`
- **iOS**: Custom Flow collector
- **Desktop**: `collectLatest` in `LaunchedEffect`

## 🔌 API Integration

### Anthropic Claude API

The app uses Claude's Messages API:

```kotlin
POST https://api.anthropic.com/v1/messages
Headers:
  - x-api-key: YOUR_KEY
  - anthropic-version: 2023-06-01
  - Content-Type: application/json

Body:
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 1024
}
```

### Switching AI Providers

To use a different provider (OpenAI, Google, etc.):

1. Update `ChatRequest`/`ChatResponse` DTOs in `data/remote/dto/ChatDto.kt`
2. Modify `ChatApiImpl` in `data/remote/ChatApi.kt`:

```kotlin
override suspend fun sendMessage(request: ChatRequest): Result<ChatResponse> {
    return try {
        val response = client.post("YOUR_API_ENDPOINT") {
            // Your API configuration
        }
        Result.success(response.body())
    } catch (e: Exception) {
        Result.failure(e)
    }
}
```

## 🎨 UI Components

### Android (Jetpack Compose)

```kotlin
@Composable
fun MessageBubble(content: String, isUser: Boolean) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = if (isUser) primary else secondaryContainer
    ) {
        Text(text = content, modifier = Modifier.padding(12.dp))
    }
}
```

### iOS (SwiftUI)

```swift
struct MessageBubble: View {
    let message: Message
    let isUser: Bool
    
    var body: some View {
        Text(message.content)
            .padding(12)
            .background(isUser ? Color.blue : Color.gray)
            .cornerRadius(16)
    }
}
```

### Desktop (Compose Desktop)

Same as Android - reuses Compose UI code!

## 🧪 Testing

```bash
# Run shared module tests
./gradlew :shared:test

# Run Android tests
./gradlew :androidApp:testDebugUnitTest

# Run iOS tests
cd iosApp && xcodebuild test -scheme iosApp
```

## 📦 Building for Production

### Android (APK/AAB)

```bash
# Debug APK
./gradlew :androidApp:assembleDebug

# Release AAB (for Play Store)
./gradlew :androidApp:bundleRelease

# Output: androidApp/build/outputs/
```

### iOS (IPA)

1. Open `iosApp.xcodeproj` in Xcode
2. Select **Product > Archive**
3. Distribute to App Store or ad-hoc

### Desktop (Installers)

```bash
# Create native installers for current OS
./gradlew :desktopApp:packageDistributionForCurrentOS

# Outputs:
# - macOS: DMG
# - Windows: MSI
# - Linux: DEB/RPM
```

## 🔒 Security Best Practices

1. **Never commit API keys** to version control
2. **Use ProGuard/R8** for Android release builds
3. **Enable code obfuscation** for iOS
4. **Validate user input** before sending to API
5. **Implement rate limiting** to prevent abuse
6. **Use HTTPS only** (enforced by Ktor)

## 🐛 Troubleshooting

### Build Errors

**"Cannot resolve symbol 'shared'"**
```bash
./gradlew :shared:build
```

**iOS Framework Not Found**
```bash
./gradlew :shared:embedAndSignAppleFrameworkForXcode
```

### Runtime Errors

**Network Request Fails**
- Check API key is set
- Verify internet connection
- Check API endpoint is correct

**UI Not Updating**
- Ensure ViewModel is properly scoped
- Verify StateFlow collection

## 🌐 Backend Server (Quarkus)

The project includes a production-ready backend server built with Quarkus.

### Features
- **REST API**: Full CRUD operations for conversations and messages
- **WebSocket**: Real-time bidirectional communication
- **Database**: PostgreSQL (prod) / H2 (dev) with Hibernate
- **OpenAPI/Swagger**: Auto-generated API documentation
- **Health Checks**: Built-in health and readiness endpoints
- **Docker Support**: Ready for containerization

### Quick Start

```bash
cd server

# Development mode (with live reload)
./mvnw quarkus:dev

# Server starts at: http://localhost:8080
# Swagger UI: http://localhost:8080/swagger-ui
# Dev UI: http://localhost:8080/q/dev
```

### API Endpoints

**Send Message**:
```bash
curl -X POST http://localhost:8080/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

**Get History**:
```bash
curl http://localhost:8080/api/chat/history/{sessionId}
```

**WebSocket**:
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/chat');
ws.send(JSON.stringify({message: "Hello via WebSocket"}));
```

### Documentation

See **[server/SERVER_README.md](server/SERVER_README.md)** for complete server documentation including:
- Detailed API documentation
- Database setup
- Docker deployment
- Production configuration
- Testing guide

## 📦 Production Checklist

- [ ] Replace API key with production credentials
- [ ] Enable ProGuard/R8 (Android)
- [ ] Configure signing (Android/iOS)
- [ ] Test on physical devices
- [ ] Update version codes
- [ ] Prepare store listings
- [ ] Test error scenarios
- [ ] Verify network security config

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 Resources

- [Kotlin Multiplatform Docs](https://kotlinlang.org/docs/multiplatform.html)
- [Compose Multiplatform](https://www.jetbrains.com/lp/compose-multiplatform/)
- [Ktor Documentation](https://ktor.io/docs/)
- [Anthropic API Docs](https://docs.anthropic.com/)

## 💡 Future Enhancements

- [ ] Message persistence (local database)
- [ ] Voice input support
- [ ] Image/file attachment support
- [ ] Multiple conversation threads
- [ ] Settings screen
- [ ] Dark mode support
- [ ] Streaming responses
- [ ] Markdown rendering
- [ ] Code syntax highlighting

## 👨‍💻 Author

Created with ❤️ using Kotlin Multiplatform

---

**Note**: This is a demo application. For production use, implement proper error handling, authentication, and follow security best practices.
