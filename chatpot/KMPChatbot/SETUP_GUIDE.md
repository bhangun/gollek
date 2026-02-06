# Setup Guide - KMP Chatbot

This guide will walk you through setting up and running the KMP Chatbot application on all platforms.

## Prerequisites Installation

### For All Platforms

1. **Install JDK 17+**
   - **macOS**: `brew install openjdk@17`
   - **Windows**: Download from [Oracle](https://www.oracle.com/java/technologies/downloads/)
   - **Linux**: `sudo apt install openjdk-17-jdk`

2. **Verify Java Installation**
   ```bash
   java -version
   # Should show version 17 or higher
   ```

### For Android Development

1. **Install Android Studio**
   - Download from [developer.android.com](https://developer.android.com/studio)
   - Install Android SDK (minimum SDK 24, target SDK 34)

2. **Configure SDK Location**
   ```bash
   # Create local.properties
   echo "sdk.dir=/path/to/Android/Sdk" > local.properties
   
   # Common locations:
   # macOS: /Users/username/Library/Android/sdk
   # Windows: C:\Users\username\AppData\Local\Android\Sdk
   # Linux: /home/username/Android/Sdk
   ```

### For iOS Development (macOS only)

1. **Install Xcode**
   - Download from Mac App Store
   - Install Command Line Tools:
     ```bash
     xcode-select --install
     ```

2. **Install CocoaPods** (if needed)
   ```bash
   sudo gem install cocoapods
   ```

### For Desktop Development

No additional setup needed if you have JDK installed!

---

## Getting the API Key

1. **Sign up for Anthropic Claude**
   - Visit: https://console.anthropic.com/
   - Create an account
   - Navigate to API Keys section
   - Generate a new API key

2. **Configure the API Key**

   **Option 1: Direct in Code (Quick Test)**
   ```kotlin
   // shared/src/commonMain/kotlin/com/kmpchatbot/data/remote/ChatApi.kt
   private val apiKey: String = "sk-ant-api03-..."
   ```

   **Option 2: Environment Variable (Recommended)**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export ANTHROPIC_API_KEY="sk-ant-api03-..."
   
   # Then update ChatApi.kt:
   private val apiKey: String = System.getenv("ANTHROPIC_API_KEY") 
       ?: throw IllegalStateException("API key not found")
   ```

   **Option 3: Gradle Properties (Production)**
   ```properties
   # Create secrets.properties in project root
   anthropic.api.key=sk-ant-api03-...
   
   # Add to .gitignore
   echo "secrets.properties" >> .gitignore
   ```

---

## Running the Application

### Android

**Method 1: Android Studio**
1. Open Android Studio
2. Click "Open" → Select `KMPChatbot` folder
3. Wait for Gradle sync to complete
4. Select `androidApp` from run configurations dropdown
5. Click Run (green play button)

**Method 2: Command Line**
```bash
# Build and install debug APK
./gradlew :androidApp:installDebug

# Launch app manually on device/emulator
adb shell am start -n com.kmpchatbot.android/.MainActivity
```

**Method 3: Direct APK**
```bash
# Build APK
./gradlew :androidApp:assembleDebug

# Install on connected device
adb install androidApp/build/outputs/apk/debug/androidApp-debug.apk
```

### iOS (macOS Only)

**Method 1: Xcode**
1. Build shared framework first:
   ```bash
   ./gradlew :shared:embedAndSignAppleFrameworkForXcode
   ```

2. Open Xcode project:
   ```bash
   open iosApp/iosApp.xcodeproj
   ```

3. Select iOS Simulator or Device
4. Click Run (⌘R)

**Method 2: Command Line**
```bash
# Build framework
./gradlew :shared:linkDebugFrameworkIosSimulatorArm64

# Build and run
cd iosApp
xcodebuild -scheme iosApp -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 15' build
```

### Desktop

**Method 1: Gradle**
```bash
# Run directly
./gradlew :desktopApp:run
```

**Method 2: Create Executable**
```bash
# Build distributable
./gradlew :desktopApp:packageDistributionForCurrentOS

# Output location:
# macOS: desktopApp/build/compose/binaries/main/dmg/
# Windows: desktopApp/build/compose/binaries/main/msi/
# Linux: desktopApp/build/compose/binaries/main/deb/
```

**Method 3: JAR**
```bash
# Create runnable JAR
./gradlew :desktopApp:packageUberJarForCurrentOS

# Run JAR
java -jar desktopApp/build/compose/jars/desktopApp-*.jar
```

---

## Testing the App

### 1. First Run

After launching on any platform:
1. Type "Hello" in the message input
2. Click Send button
3. Wait for AI response (may take 2-5 seconds)

### 2. Expected Behavior

✅ **Success Flow:**
- Message appears in chat immediately
- Loading indicator shows
- AI response appears after a few seconds
- Chat scrolls to bottom automatically

❌ **Common Issues & Solutions:**

**"Network error" or "API error"**
- Check internet connection
- Verify API key is correct
- Check API quota/credits

**"Cannot resolve symbol 'shared'"**
```bash
./gradlew :shared:build
```

**iOS: "Framework not found"**
```bash
./gradlew :shared:embedAndSignAppleFrameworkForXcode
```

**Desktop: App doesn't start**
```bash
# Check Java version
java -version

# Clean and rebuild
./gradlew clean :desktopApp:build
```

---

## Development Workflow

### Making Changes to Shared Code

1. Edit files in `shared/src/commonMain/`
2. Rebuild shared module:
   ```bash
   ./gradlew :shared:build
   ```
3. Re-run platform apps

### Making UI Changes

**Android:**
- Edit `androidApp/src/main/kotlin/MainActivity.kt`
- Hot reload works in Android Studio

**iOS:**
- Edit `iosApp/iosApp/ContentView.swift`
- Use Xcode's preview or re-run

**Desktop:**
- Edit `desktopApp/src/main/kotlin/Main.kt`
- Re-run Gradle task

### Adding Dependencies

1. **Shared Dependencies** → `shared/build.gradle.kts`:
   ```kotlin
   commonMain {
       dependencies {
           implementation("io.ktor:ktor-client-core:2.3.6")
       }
   }
   ```

2. **Android Dependencies** → `androidApp/build.gradle.kts`:
   ```kotlin
   dependencies {
       implementation("androidx.compose.material3:material3:1.1.2")
   }
   ```

3. **iOS Dependencies** → Use CocoaPods or Swift Package Manager

---

## Advanced Configuration

### Custom Model Selection

Edit `ChatRequest` in `shared/src/commonMain/kotlin/.../dto/ChatDto.kt`:

```kotlin
@Serializable
data class ChatRequest(
    val model: String = "claude-opus-4-20250514", // Change model
    val messages: List<MessageDto>,
    val maxTokens: Int = 2048 // Increase for longer responses
)
```

### Enable Logging

**Shared Module:**
Already enabled via Ktor Logging plugin

**Android:**
Check Logcat in Android Studio (filter: "System.out")

**iOS:**
Check Xcode console output

**Desktop:**
Terminal output shows logs automatically

### Offline Support

To add local caching:

1. Add SQLDelight to `shared/build.gradle.kts`
2. Implement message persistence
3. Show cached messages when offline

---

## Building for Production

### Android Release Build

1. **Generate Signing Key:**
   ```bash
   keytool -genkey -v -keystore release-key.jks -keyalg RSA \
     -keysize 2048 -validity 10000 -alias release
   ```

2. **Configure Signing** in `androidApp/build.gradle.kts`:
   ```kotlin
   android {
       signingConfigs {
           create("release") {
               storeFile = file("release-key.jks")
               storePassword = "your-password"
               keyAlias = "release"
               keyPassword = "your-password"
           }
       }
       buildTypes {
           release {
               signingConfig = signingConfigs.getByName("release")
               isMinifyEnabled = true
               proguardFiles(...)
           }
       }
   }
   ```

3. **Build Release AAB:**
   ```bash
   ./gradlew :androidApp:bundleRelease
   
   # Output: androidApp/build/outputs/bundle/release/
   ```

### iOS Release Build

1. Open Xcode
2. Product → Archive
3. Distribute App → App Store Connect
4. Follow Apple's submission process

### Desktop Distribution

```bash
# Create installer for current OS
./gradlew :desktopApp:packageDistributionForCurrentOS

# macOS DMG: Double-click to install
# Windows MSI: Run installer
# Linux DEB: sudo dpkg -i *.deb
```

---

## Troubleshooting

### Gradle Issues

```bash
# Clear Gradle cache
rm -rf ~/.gradle/caches/

# Clean build
./gradlew clean

# Rebuild with logs
./gradlew build --stacktrace
```

### Android Issues

```bash
# Clear Android build cache
./gradlew cleanBuildCache

# Invalidate IDE caches
# Android Studio → File → Invalidate Caches and Restart
```

### iOS Issues

```bash
# Clean Xcode build
cd iosApp
xcodebuild clean

# Delete derived data
rm -rf ~/Library/Developer/Xcode/DerivedData
```

### Network Issues

1. Check firewall settings
2. Verify proxy configuration
3. Test API endpoint manually:
   ```bash
   curl -X POST https://api.anthropic.com/v1/messages \
     -H "x-api-key: YOUR_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hi"}],"max_tokens":100}'
   ```

---

## Next Steps

✅ You're now ready to:
1. Customize the UI
2. Add features (voice, images, etc.)
3. Implement persistence
4. Deploy to stores
5. Build your own AI-powered apps!

Need help? Check the main README.md or open an issue on GitHub.

Happy coding! 🚀
