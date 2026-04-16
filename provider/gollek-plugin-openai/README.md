# OpenAI Cloud Provider Plugin

Cloud provider plugin for OpenAI models (GPT-4, GPT-3.5-turbo, etc.) in the Gollek inference engine.

## 🚀 Features

- ✅ **Multiple Models**: GPT-4, GPT-4 Turbo, GPT-3.5-turbo
- ✅ **Streaming Support**: Real-time token streaming
- ✅ **Function Calling**: Tool/function invocation support
- ✅ **Hot-Reload**: Update plugin without restart
- ✅ **Isolated**: Runs in isolated ClassLoader

## 📦 Installation

### Option 1: Build from Source

```bash
cd inference-gollek/plugins/gollek-plugin-openai
mvn clean install
```

This will automatically copy the JAR to `~/.gollek/plugins/`

### Option 2: Download Pre-built JAR

```bash
# Download latest release
wget https://github.com/your-org/gollek/releases/download/v1.0.0/gollek-plugin-openai-1.0.0.jar

# Copy to plugin directory
cp gollek-plugin-openai-1.0.0.jar ~/.gollek/plugins/
```

### Option 3: Use Fat JAR (with all dependencies)

```bash
mvn clean package -Pfat-jar
cp target/gollek-plugin-openai-1.0.0-all.jar ~/.gollek/plugins/
```

## ⚙️ Configuration

### 1. Via plugin.json

Edit `~/.gollek/plugins/plugin.json`:

```json
{
  "plugins": {
    "openai-cloud-provider": {
      "apiKey": "sk-your-api-key-here",
      "baseUrl": "https://api.openai.com/v1",
      "organization": "org-your-org-id"
    }
  }
}
```

### 2. Via System Properties

```bash
java -Dgollek.plugin.openai-cloud-provider.apiKey=sk-... \
     -Dgollek.plugin.openai-cloud-provider.baseUrl=https://api.openai.com/v1 \
     -jar gollek-runtime.jar
```

### 3. Via Environment Variables

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1
```

## 🎯 Usage

### Basic Completion

```java
InferenceRequest request = InferenceRequest.builder()
    .modelId("gpt-4")
    .message("Hello, how are you?")
    .build();

InferenceResponse response = provider.complete(request);
System.out.println(response.content());
```

### Streaming

```java
InferenceRequest request = InferenceRequest.builder()
    .modelId("gpt-3.5-turbo")
    .message("Tell me a story")
    .build();

provider.stream(request)
    .subscribe(chunk -> System.out.print(chunk.delta()));
```

### Function Calling

```java
FunctionDefinition function = FunctionDefinition.builder()
    .name("get_weather")
    .description("Get current weather")
    .parameters(JsonSchema.builder()
        .property("location", "string", "City name")
        .build())
    .build();

InferenceRequest request = InferenceRequest.builder()
    .modelId("gpt-4")
    .message("What's the weather in Tokyo?")
    .functions(List.of(function))
    .build();

InferenceResponse response = provider.complete(request);
```

## 📊 Supported Models

| Model | Context Length | Max Tokens | Description |
|-------|---------------|------------|-------------|
| gpt-4 | 8,192 | 8,192 | Most capable GPT-4 |
| gpt-4-turbo | 128,000 | 4,096 | Faster GPT-4 |
| gpt-4-32k | 32,768 | 4,096 | GPT-4 with 32K context |
| gpt-3.5-turbo | 4,096 | 4,096 | Fast & efficient |
| gpt-3.5-turbo-16k | 16,384 | 4,096 | GPT-3.5 with 16K context |

## 🔍 Health Check

Check plugin health:

```bash
curl http://localhost:8080/api/v1/plugins/openai-cloud-provider/health
```

Response:
```json
{
  "status": "healthy",
  "details": {
    "models": "gpt-4, gpt-4-turbo, gpt-3.5-turbo",
    "version": "1.0.0"
  }
}
```

## 🔄 Hot-Reload

The plugin supports hot-reload. To update:

```bash
# 1. Build new version
mvn clean package

# 2. Copy to plugin directory (auto-reloads)
cp target/gollek-plugin-openai-2.0.0.jar ~/.gollek/plugins/

# Or replace existing
cp target/gollek-plugin-openai-2.0.0.jar ~/.gollek/plugins/gollek-plugin-openai.jar
```

The plugin will automatically reload within 1 second.

## 🐛 Troubleshooting

### Plugin Not Loading

**Check logs**:
```bash
tail -f ~/.gollek/logs/gollek.log | grep openai
```

**Common issues**:
- Missing API key
- JAR not in `~/.gollek/plugins/`
- Version mismatch with SPI modules

### API Errors

**Check**:
- API key is valid
- Network connectivity
- Rate limits

### Streaming Issues

**Check**:
- Client supports Server-Sent Events (SSE)
- Network allows streaming
- Model supports streaming

## 📝 Development

### Build

```bash
mvn clean install
```

### Test

```bash
mvn test
```

### Run with Debug

```bash
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005 \
     -Dgollek.plugin.directory=~/.gollek/plugins \
     -jar gollek-runtime.jar
```

### Create New Version

1. Update version in `pom.xml`
2. Update version in `plugin.json`
3. Update CHANGELOG.md
4. Build and tag release

## 📚 API Reference

### OpenAiCloudProvider

```java
// Get provider instance
OpenAiCloudProvider provider = new OpenAiCloudProvider();

// Initialize
provider.initialize(context);

// Start
provider.start();

// Use
InferenceResponse response = provider.complete(request);
Multi<StreamingInferenceChunk> stream = provider.stream(request);

// Stop
provider.stop();

// Shutdown
provider.shutdown();
```

### Configuration Properties

| Property | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `apiKey` | string | ✅ | - | OpenAI API key |
| `baseUrl` | string | ❌ | `https://api.openai.com/v1` | Custom endpoint |
| `organization` | string | ❌ | - | Organization ID |
| `maxTokens` | number | ❌ | 4096 | Default max tokens |
| `temperature` | number | ❌ | 0.7 | Default temperature |

## 🔗 Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Gollek Plugin System Guide](../PLUGIN_SYSTEM_GUIDE.md)
- [Gollek Inference Engine](../../README.md)

## 📄 License

MIT License - See LICENSE file for details.

---

**Version**: 1.0.0  
**Status**: ✅ Ready  
**Author**: Kayys.tech
