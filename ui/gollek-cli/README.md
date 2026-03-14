# Production-Ready Gollek CLI

Production-ready CLI similar to modern LLM CLIs, integrating with various providers and model registry.

## v0.1.4 Enhancements: Dual-Format Serving

The Gollek CLI now supports **Dual-Format Serving**, allowing you to run both GGUF and SafeTensors models seamlessly.

### Key Features:
- **Format-Aware Routing**: The CLI automatically detects the model format (GGUF, SafeTensors, TorchScript) and routes to the appropriate provider.
- **Unified Model Registry**: All local models are now managed through a central `LocalModelRegistry`, improving discovery and consistency.
- **Enhanced SafeTensors Support**: Direct inference support for SafeTensors models (requires `gollek-ext-format-safetensor` extension).
- **GGUF Performance**: Optimized local inference via llama.cpp using Panama FFM API.

## Install (Release Artifacts)

Release workflow: `.github/workflows/golek-cli-release.yml`

```bash
# macOS / Linux (curl installer)
curl -fsSL https://github.com/bhangun/gollek/releases/latest/download/install.sh | bash
```

```bash
# Homebrew (after adding generated formula to your tap)
brew tap bhangun/gollek
brew install gollek
```

```powershell
# Chocolatey
choco install gollek
```

```powershell
# Windows native executable
Invoke-WebRequest -Uri "https://github.com/bhangun/gollek/releases/latest/download/gollek-windows-x64.exe" -OutFile "gollek.exe"
.\gollek.exe --version
```

```powershell
# Windows JVM fallback package (Java 21+)
# Extract gollek-jvm.zip first
.\bin\gollek.bat --version
```


```bash
  _____       _      _    
 / ____|     | |    | |   
| |  __  ___ | | ___| | __
| | |_ |/ _ \| |/ _ \ |/ /
| |__| | (_) | |  __/   < 
 \_____|\___/|_|\___|_|\_\

Model: qwen2.5-7b-instruct-GGUF
Provider: gguf
Commands: 'exit' to quit, '/reset' to clear history.
Note: Use '\' at the end of a line for multiline input.
--------------------------------------------------

>>> Hello, what can you do?

Assistant: I can help you with coding, writing, and analysis...

[Tokens: 42, Duration: 0.85s, Speed: 49.41 t/s]

```

## Build

```bash
cd inference-gollek && mvn clean package -pl ui/gollek-cli -am -DskipTests
```

## Build Native
```bash
cd inference-gollek && mvn clean package -pl ui/gollek-cli -am -Pnative -DskipTests
```

Other options to build native:

Option 1: Disable UPX compression (quickest solution) Add this property to skip the compression step:

```bash
mvn clean package -pl ui/gollek-cli -am -Pnative -DskipTests -Dquarkus.native.compression.disable=true
```
Option 2: Install UPX (for smaller executable) If you want the compressed executable, install UPX manually:

```bash
brew install upx
mvn clean package -pl ui/gollek-cli -am -Pnative -DskipTests
```

> **Note**: The GGUF provider uses Panama FFM (Foreign Function & Memory API) to interface with llama.cpp. GraalVM native image requires explicit registration of foreign function calls via `reachability-metadata.json` in the `gollek-ext-format-gguf` module.

## Goal

Create a fully functional CLI that supports:
- **Local inference** via GGUF adapter (llama.cpp)
- **Local Registry integration** for managed model downloads
- **Cloud providers** (Gemini, etc.)
- **Model management** (pull, list, show, delete)

## Architecture Overview

```mermaid
graph TD
    CLI[Gollek CLI] --> Registry[LocalModelRegistry]
    Registry --> SDK[LocalGollekSdk]
    
    SDK --> Router[FormatAwareProviderRouter]
    Router --> GGUF[GGUFProvider]
    Router --> ST[SafetensorProvider]
    Router --> DJL[DJL/TorchScript Provider]
```

## Supported Providers

| Provider | Format Support | Description |
|----------|----------------|-------------|
| `gguf` | `.gguf` | Local GGUF models via llama.cpp |
| `safetensor` | `.safetensors`, `.bin` | Direct SafeTensors/PyTorch weights |
| `djl` | `.pt`, `.pth` | TorchScript models |
| `gemini` | Cloud | Google Gemini |

## Commands

| Command | Description | Provider/Module |
|---------|-------------|-----------------|
| `gollek run` | Run inference with a model | All providers |
| `gollek pull` | Download model from registry | HuggingFace |
| `gollek list` | List local models | LocalModelRepository |
| `gollek show` | Show model details | LocalModelRegistry |
| `gollek serve` | Start API server | All providers |
| `gollek providers` | List available providers | ProviderRegistry |
| `gollek chat` | Interactive chat session | All providers |

---


Production-ready CLI with full provider support.

## Usage Examples

### Auto-Detect and Run
```bash
# GGUF model
gollek run --model qwen2.5-7b-instruct-GGUF --prompt "Hi"

# SafeTensors model
gollek run --model hf:Qwen/Qwen2.5-0.5B-Instruct --prompt "Describe the moon."
```

### Interactive Chat
```bash
gollek chat --model llama-3.2-1b-instruct
```

```bash
# MCP registry: add server config once, then reuse
gollek mcp add '{"mcpServers":{"image-downloader":{"command":"node","args":["/path/to/mcp-image-downloader/build/index.js"]}}}'
gollek mcp add --name image-downloader --command node --args-json '["/path/to/mcp-image-downloader/build/index.js"]'
gollek mcp add --from-url https://example.com/mcp-servers.json
gollek mcp add --from-registry qpd-v/mcp-image-downloader
gollek mcp add --from-registry https://mcpservers.org/servers/qpd-v/mcp-image-downloader
gollek mcp add --from-url https://example.com/mcp-servers.json --server image-downloader
gollek mcp add --list-from-registry --from-registry qpd-v/mcp-image-downloader
gollek mcp add --list-from-registry --from-url https://example.com/mcp-servers.json
gollek mcp list
gollek mcp show image-downloader
gollek mcp show image-downloader --json
gollek mcp doctor
gollek mcp test image-downloader
gollek mcp test --all
gollek mcp edit image-downloader --command node --args-json '["/new/path/index.js"]'
gollek mcp disable image-downloader
gollek mcp enable image-downloader
gollek mcp rename image-downloader image-fetcher
gollek mcp export --file /tmp/mcp-servers.json
gollek mcp import --file /tmp/mcp-servers.json --merge
gollek mcp import --file /tmp/mcp-servers.json --replace
gollek mcp remove image-downloader

# Optional enterprise mode: use centralized MCP registry API (DB-backed) instead of local ~/.gollek/mcp/servers.json
GOLLEK_ENTERPRISE_ENABLED=true \
GOLLEK_MCP_REGISTRY_MODE=remote \
GOLLEK_MCP_REGISTRY_API_BASE_URL=http://localhost:8080 \
GOLLEK_MCP_REGISTRY_API_TOKEN=<jwt-or-bearer-token> \
GOLLEK_TENANT_ID=<tenant-id> \
gollek mcp add '{"mcpServers":{"image-downloader":{"command":"node","args":["/path/to/mcp-image-downloader/build/index.js"]}}}'

GOLLEK_ENTERPRISE_ENABLED=true \
GOLLEK_MCP_REGISTRY_MODE=remote \
GOLLEK_MCP_REGISTRY_API_BASE_URL=http://localhost:8080 \
GOLLEK_MCP_REGISTRY_API_TOKEN=<jwt-or-bearer-token> \
GOLLEK_TENANT_ID=<tenant-id> \
gollek mcp remove image-downloader

# Then run using provider mcp (local model lookup is skipped for --provider mcp)
gollek run --provider mcp --model any --prompt "hello"

# Build
cd inference-gollek && mvn clean package -pl ui/gollek-cli -am -DskipTests

# List providers
java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar providers

java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar run --provider gguf --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello"

java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar run --provider gguf --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain quantum entanglement in 2 sentences."



java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar chat --provider gguf --model Qwen/Qwen2.5-0.5B-Instruct 

# Minimal output mode (recommended for native GGUF)
java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar chat --provider gguf --model Qwen/Qwen2.5-0.5B-Instruct --quiet

java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar chat --provider gguf --model Qwen/Qwen2.5-0.5B-Instruct 


java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar chat --model google-t5/t5-small

java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar chat --model HuggingFaceTB/SmolVLM-256M-Instruct

HuggingFaceTB/SmolLM2-135M
meta-llama/Llama-3.2-1B
google/gemma-2b-it


GGUF_GPU_ENABLED=true GGUF_GPU_LAYERS=8 GGUF_BATCH_SIZE=64 \
GGUF_THREADS=8 \
java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar \
chat --model meta-llama/Llama-3.2-1B-Instruct

# Run with different providers
java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar run \
  --provider openai --model gpt-4 --prompt "Hello"

java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar run \
  --provider anthropic --model claude-3-opus --prompt "Hello"

java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar run \
  --provider cerebras --model llama-3.1-8b --prompt "Hello"
```

```bash
gollek-cli % java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar providers
ID              NAME                 VERSION    STATUS    
------------------------------------------------------------
cerebras        Cerebras             1.0.0      HEALTHY   
gemini          Google Gemini        1.0.0      HEALTHY   
gguf            GGUF Provider (ll... 1.1.0      HEALTHY   
libtorch        LibTorch/TorchScript 1.1.0      HEALTHY   
mistral         Mistral AI           1.0.0      HEALTHY   
 

6 provider(s) available
```

### Run native
```bash
./ui/gollek-cli/target/gollek-cli-1.0.0-SNAPSHOT-runner chat --provider gguf --model Qwen/Qwen2.5-0.5B-Instruct  
```

### Native smoke check
```bash
./scripts/native-gguf-smoke.sh Qwen/Qwen2.5-0.5B-Instruct
```

If the native executable cannot find `llama.cpp` libraries, point it explicitly:

```bash
export GOLEK_LLAMA_LIB_DIR=/absolute/path/to/native-libs
# or:
export GOLEK_LLAMA_LIB_PATH=/absolute/path/to/libllama.dylib
```

To enable file logging explicitly:
```bash
export GOLEK_CLI_FILE_LOG=true
export GOLEK_CLI_LOG_FILE="$HOME/.gollek/logs/cli.log"
```



```bash
# First run - downloads
$ gollek run --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hi"
Checking model: Qwen/Qwen2.5-0.5B-Instruct... not found locally.
Attempting to download from Hugging Face...
Downloading: ████████████████████ 100% (468/468 MB)
✓ Model saved to: ~/.gollek/models/gguf/Qwen_Qwen2.5-0.5B-Instruct-GGUF
Model path: ~/.gollek/models/gguf/Qwen_Qwen2.5-0.5B-Instruct-GGUF
[inference output]

# Second run - uses existing
$ gollek run --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hi"
Checking model: Qwen/Qwen2.5-0.5B-Instruct... found local variant: Qwen/Qwen2.5-0.5B-Instruct-GGUF
Model path: ~/.gollek/models/gguf/Qwen_Qwen2.5-0.5B-Instruct-GGUF
[inference output]

# Using custom path
$ gollek run --model-path /my/models/custom.gguf --prompt "Hi"
Using model from: /my/models/custom.gguf
[inference output]

# Ctrl+C - exits immediately
$ gollek run --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hi"
Downloading: ████░░░░░░░░░░░░░░░░ 20% (94/468 MB)^C
✗ Download cancelled
```

## Build and Run
```bash
cd inference-gollek
mvn clean package -pl ui/gollek-cli -am -DskipTests
java -jar ui/gollek-cli/target/gollek-cli-*-runner.jar chat -m <model> --session

```

```bash
# Interactive chat with session (KV cache enabled)
gollek chat --model Qwen/Qwen2.5-0.5B-Instruct --session

# Chat with function calling model
gollek chat --model <tool-model> --session
# Then type: "What's the weather in Jakarta?"
# Output: [Tool Call] get_weather({"city": "Jakarta"})
```

## Technical Details

The CLI leverages a centralized `tech.kayys.gollek.spi.model.LocalModelRegistry` to scan and index models across multiple root directories:
- `~/.gollek/models/gguf`
- `~/.gollek/models/safetensors`
- `~/.gollek/models/torchscript`

Format detection is performed via magic bytes (ModelFormatDetector) to ensure reliable provider routing regardless of file extension.
a

## Proposed Changes

### Build Configuration

#### [MODIFY] [pom.xml](ui/gollek-cli/pom.xml)
Add dependencies for all providers and model repository:
- `gollek-sdk-java-local` - Local SDK
- `gollek-model-repo-core` - Model repository
- `gollek-ext-cloud-gemini` - Gemini provider
- `gollek-provider-huggingface` - HuggingFace for model downloads
- `gollek-ext-format-gguf` - Local GGUF inference

---

### CLI Commands

#### [MODIFY] [GollekCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/GollekCommand.java)
Update to include all subcommands.

#### [MODIFY] [RunCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/RunCommand.java)
Enhanced run command with:
- `--provider` option to select provider (litert, gguf, gemini)
- `--stream` flag for streaming output
- `--temperature`, `--max-tokens` options

#### [NEW] [PullCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/PullCommand.java)
```java
@Command(name = "pull", description = "Pull a model from registry")
```
- Support HuggingFace: `gollek pull hf:TheBloke/Llama-2-7B-GGUF`
- Progress bar display

#### [NEW] [ListCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/ListCommand.java)
```java
@Command(name = "list", aliases = "ls", description = "List local models")
```
- Display: NAME, SIZE, FORMAT, MODIFIED

#### [NEW] [ShowCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/ShowCommand.java)
Show model details (parameters, license, architecture).

#### [NEW] [ProvidersCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/ProvidersCommand.java)
List available providers with status (healthy/unhealthy).

#### [NEW] [ChatCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/ChatCommand.java)
Interactive chat mode with conversation history.

#### [NEW] [ServeCommand.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/commands/ServeCommand.java)
Start local API server (OpenAI-compatible).

---

### Supporting Classes

#### [NEW] [ProgressBar.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/util/ProgressBar.java)
Console progress bar for downloads.

#### [NEW] [OutputFormatter.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/util/OutputFormatter.java)
Format output as table, JSON, or plain text.

#### [NEW] [ProviderResolver.java](ui/gollek-cli/src/main/java/tech/kayys/gollek/cli/service/ProviderResolver.java)
Resolve model name to appropriate provider.

---

## Implementation Priority

1. **Phase 1** - Core Commands
   - `run` with provider selection
   - `list` local models
   - `show` model info
   - `providers` list

2. **Phase 2** - Model Management
   - `pull` from HuggingFace
   - Progress bar

3. **Phase 3** - Advanced Features
   - `chat` interactive mode
   - `serve` API server

---

## Verification Plan

### Build & Test
```bash
cd inference-gollek
mvn clean package -pl ui/gollek-cli -am
mvn test -pl ui/gollek-cli
```

### Manual Testing
```bash
# List providers
java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar providers

# List local models
java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar list

# Run with GGUF
java -jar ui/gollek-cli/target/quarkus-app/quarkus-run.jar run \
  --provider gguf --model /path/to/model.gguf --prompt "Hello"
  --provider litert --model /path/to/model.tflite --prompt "Hello"
```
