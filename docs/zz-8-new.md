# 🚀 Continued Implementation: Inference Kernel & Provider System

I'll continue the implementation systematically, building upon the existing foundation with a focus on modularity, robustness, and multi-tenancy support.

---

---


firefox
https://claude.ai/chat/785c4246-7dff-41c8-b5cd-816e581c9154




# Complete Implementation: Inference Provider SPI with MCP & Advanced Memory Support

Based on the architecture blueprint and your requirements, I'll provide a complete, production-ready implementation of the Inference Provider SPI module with MCP support and advanced memory capabilities.

## 🎯 Architecture Overview

```
inference-providers-spi/
├── provider-core/          # Core provider abstractions
├── provider-mcp/           # MCP protocol implementation
├── provider-memory/        # Advanced memory subsystem
├── provider-tool/          # Tool/Function calling
└── provider-integration/   # External MCP server integration
```

---

## 📦 Module 1: Provider Core

### Project Structure



### pom.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>tech.kayys.wayang</groupId>
        <artifactId>wayang-inference-server</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>

    <artifactId>inference-providers-spi</artifactId>
    <packaging>jar</packaging>
    <name>Inference Providers SPI</name>
    <description>Provider abstraction and SPI for inference engine</description>

</project>
```

### Core Provider Interfaces

#### LLMProvider.java

```java
package tech.kayys.wayang.inference.providers.core;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.inference.api.TenantContext;

import java.util.Map;

/**
 * Core SPI for LLM providers.
 * All providers (local, cloud, MCP) implement this interface.
 * 
 * Thread-safe and stateless.
 * Each provider instance may be shared across multiple tenants.
 */
public interface LLMProvider extends AutoCloseable {

    /**
     * Unique provider identifier (e.g., "ollama", "openai", "mcp-remote")
     */
    String id();

    /**
     * Human-readable display name
     */
    String displayName();

    /**
     * Provider capabilities (streaming, tools, multimodal, etc.)
     */
    ProviderCapabilities capabilities();

    /**
     * Initialize provider with configuration
     * Called once during startup or hot-reload
     * 
     * @param config Provider-specific configuration
     * @param context Initialization context
     */
    Uni<Void> initialize(
        Map<String, Object> config,
        ProviderContext context
    );

    /**
     * Execute synchronous inference
     * 
     * @param request Normalized provider request
     * @param tenantContext Multi-tenant context
     * @return Provider response
     */
    Uni<ProviderResponse> infer(
        ProviderRequest request,
        TenantContext tenantContext
    );

    /**
     * Health check for this provider
     * 
     * @return Health status with diagnostics
     */
    Uni<ProviderHealth> health();

    /**
     * Get current resource metrics
     */
    ProviderMetrics metrics();

    /**
     * Warmup the provider (optional optimization)
     * 
     * @param sampleRequests Sample requests for warming
     */
    default Uni<Void> warmup(ProviderRequest... sampleRequests) {
        return Uni.createFrom().voidItem();
    }

    /**
     * Gracefully shutdown and release resources
     */
    @Override
    void close();

    /**
     * Provider metadata for selection and routing
     */
    default ProviderDescriptor descriptor() {
        return ProviderDescriptor.builder()
            .id(id())
            .displayName(displayName())
            .capabilities(capabilities())
            .build();
    }
}
```

#### ProviderCapabilities.java

```java
package tech.kayys.wayang.inference.providers.core;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.*;

/**
 * Immutable capability descriptor for providers.
 * Used for intelligent routing and compatibility checks.
 */
public final class ProviderCapabilities {

    private final boolean streaming;
    private final boolean functionCalling;
    private final boolean toolCalling;
    private final boolean multimodal;
    private final boolean vision;
    private final boolean audio;
    private final boolean embedding;
    private final boolean fineTuning;
    private final int maxContextTokens;
    private final int maxOutputTokens;
    private final Set<String> supportedModels;
    private final Map<String, Object> metadata;

    @JsonCreator
    public ProviderCapabilities(
        @JsonProperty("streaming") boolean streaming,
        @JsonProperty("functionCalling") boolean functionCalling,
        @JsonProperty("toolCalling") boolean toolCalling,
        @JsonProperty("multimodal") boolean multimodal,
        @JsonProperty("vision") boolean vision,
        @JsonProperty("audio") boolean audio,
        @JsonProperty("embedding") boolean embedding,
        @JsonProperty("fineTuning") boolean fineTuning,
        @JsonProperty("maxContextTokens") int maxContextTokens,
        @JsonProperty("maxOutputTokens") int maxOutputTokens,
        @JsonProperty("supportedModels") Set<String> supportedModels,
        @JsonProperty("metadata") Map<String, Object> metadata
    ) {
        this.streaming = streaming;
        this.functionCalling = functionCalling;
        this.toolCalling = toolCalling;
        this.multimodal = multimodal;
        this.vision = vision;
        this.audio = audio;
        this.embedding = embedding;
        this.fineTuning = fineTuning;
        this.maxContextTokens = maxContextTokens;
        this.maxOutputTokens = maxOutputTokens;
        this.supportedModels = supportedModels != null 
            ? Collections.unmodifiableSet(new HashSet<>(supportedModels))
            : Collections.emptySet();
        this.metadata = metadata != null
            ? Collections.unmodifiableMap(new HashMap<>(metadata))
            : Collections.emptyMap();
    }

    // Getters
    public boolean isStreaming() { return streaming; }
    public boolean isFunctionCalling() { return functionCalling; }
    public boolean isToolCalling() { return toolCalling; }
    public boolean isMultimodal() { return multimodal; }
    public boolean isVision() { return vision; }
    public boolean isAudio() { return audio; }
    public boolean isEmbedding() { return embedding; }
    public boolean isFineTuning() { return fineTuning; }
    public int getMaxContextTokens() { return maxContextTokens; }
    public int getMaxOutputTokens() { return maxOutputTokens; }
    public Set<String> getSupportedModels() { return supportedModels; }
    public Map<String, Object> getMetadata() { return metadata; }

    public boolean supportsModel(String model) {
        return supportedModels.isEmpty() || supportedModels.contains(model);
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean streaming = false;
        private boolean functionCalling = false;
        private boolean toolCalling = false;
        private boolean multimodal = false;
        private boolean vision = false;
        private boolean audio = false;
        private boolean embedding = false;
        private boolean fineTuning = false;
        private int maxContextTokens = 4096;
        private int maxOutputTokens = 2048;
        private final Set<String> supportedModels = new HashSet<>();
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder functionCalling(boolean functionCalling) {
            this.functionCalling = functionCalling;
            return this;
        }

        public Builder toolCalling(boolean toolCalling) {
            this.toolCalling = toolCalling;
            return this;
        }

        public Builder multimodal(boolean multimodal) {
            this.multimodal = multimodal;
            return this;
        }

        public Builder vision(boolean vision) {
            this.vision = vision;
            return this;
        }

        public Builder audio(boolean audio) {
            this.audio = audio;
            return this;
        }

        public Builder embedding(boolean embedding) {
            this.embedding = embedding;
            return this;
        }

        public Builder fineTuning(boolean fineTuning) {
            this.fineTuning = fineTuning;
            return this;
        }

        public Builder maxContextTokens(int maxContextTokens) {
            this.maxContextTokens = maxContextTokens;
            return this;
        }

        public Builder maxOutputTokens(int maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder supportedModel(String model) {
            this.supportedModels.add(model);
            return this;
        }

        public Builder supportedModels(Set<String> models) {
            this.supportedModels.addAll(models);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public ProviderCapabilities build() {
            return new ProviderCapabilities(
                streaming, functionCalling, toolCalling, multimodal,
                vision, audio, embedding, fineTuning,
                maxContextTokens, maxOutputTokens,
                supportedModels, metadata
            );
        }
    }

    @Override
    public String toString() {
        return "ProviderCapabilities{" +
               "streaming=" + streaming +
               ", functionCalling=" + functionCalling +
               ", toolCalling=" + toolCalling +
               ", maxContextTokens=" + maxContextTokens +
               '}';
    }
}
```

#### ProviderRequest.java

```java
package tech.kayys.wayang.inference.providers.core;

```

#### ToolDefinition.java

```java
package tech.kayys.wayang.inference.providers.core;

```

#### ProviderResponse.java

```java
package tech.kayys.wayang.inference.providers.core;

```


## 📦 Completing Provider Core Module

### ProviderContext.java

```java
package tech.kayys.wayang.inference.providers.core;

```

### ProviderHealth.java

```java
package tech.kayys.wayang.inference.providers.core;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.*;

/**
 * Health status for a provider.
 */
public final class ProviderHealth {

    public enum Status {
        UP,
        DOWN,
        DEGRADED,
        UNKNOWN
    }

    private final Status status;
    private final String message;
    private final Instant timestamp;
    private final Map<String, Object> details;

    @JsonCreator
    public ProviderHealth(
        @JsonProperty("status") Status status,
        @JsonProperty("message") String message,
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("details") Map<String, Object> details
    ) {
        this.status = status;
        this.message = message;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.details = details != null
            ? Collections.unmodifiableMap(new HashMap<>(details))
            : Collections.emptyMap();
    }

    public Status getStatus() { return status; }
    public String getMessage() { return message; }
    public Instant getTimestamp() { return timestamp; }
    public Map<String, Object> getDetails() { return details; }

    public boolean isHealthy() {
        return status == Status.UP;
    }

    public static ProviderHealth up() {
        return new ProviderHealth(Status.UP, "Provider is healthy", Instant.now(), null);
    }

    public static ProviderHealth up(String message) {
        return new ProviderHealth(Status.UP, message, Instant.now(), null);
    }

    public static ProviderHealth down(String message) {
        return new ProviderHealth(Status.DOWN, message, Instant.now(), null);
    }

    public static ProviderHealth degraded(String message) {
        return new ProviderHealth(Status.DEGRADED, message, Instant.now(), null);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Status status = Status.UNKNOWN;
        private String message;
        private Instant timestamp = Instant.now();
        private final Map<String, Object> details = new HashMap<>();

        public Builder status(Status status) {
            this.status = status;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder detail(String key, Object value) {
            this.details.put(key, value);
            return this;
        }

        public ProviderHealth build() {
            return new ProviderHealth(status, message, timestamp, details);
        }
    }

    @Override
    public String toString() {
        return "ProviderHealth{" +
               "status=" + status +
               ", message='" + message + '\'' +
               '}';
    }
}
```

### ProviderMetrics.java

```java
package tech.kayys.wayang.inference.providers.core;

```

### ProviderException.java

```java
package tech.kayys.wayang.inference.providers.core;

```

### ProviderRegistry.java

```java
package tech.kayys.wayang.inference.providers.registry;

```

### ProviderDescriptor.java

```java
package tech.kayys.wayang.inference.providers.registry;

import tech.kayys.wayang.inference.providers.core.ProviderCapabilities;

```

---

## 📦 Module 3: Circuit Breaker & Resilience

### CircuitBreaker.java

```java
package tech.kayys.wayang.inference.providers.circuit;

```

### CircuitBreakerMetrics.java

```java
package tech.kayys.wayang.inference.providers.circuit;

```

### DefaultCircuitBreaker.java

```java
package tech.kayys.wayang.inference.providers.circuit;

```

### CircuitBreakerOpenException.java

```java
package tech.kayys.wayang.inference.providers.circuit;

```

---

Now I have completed the core provider module. Shall I continue with:
1. **Full MCP Protocol Implementation** (client, server integration, tool execution)
2. **Advanced Memory Subsystem** (episodic, semantic, procedural memory)
3. **Streaming Provider Implementation**

Which would you like next?


### pom.xml

```xml

```

---

## 🔌 MCP Protocol Layer

### JsonRpcMessage.java

```java
package tech.kayys.wayang.inference.providers.mcp.protocol;

```

### MCPRequest.java

```java
package tech.kayys.wayang.inference.providers.mcp.protocol;

```

### MCPResponse.java

```java
package tech.kayys.wayang.inference.providers.mcp.protocol;

```

### MCPNotification.java

```java
package tech.kayys.wayang.inference.providers.mcp.protocol;

```

### MCPError.java

```java
package tech.kayys.wayang.inference.providers.mcp.protocol;

```

---

## 🚀 MCP Client Implementation

### MCPClientConfig.java

```java
package tech.kayys.wayang.inference.providers.mcp.client;

```

### MCPTransport.java

```java
package tech.kayys.wayang.inference.providers.mcp.client;

```

### StdioTransport.java

```java
package tech.kayys.wayang.inference.providers.mcp.client;

```

### MCPClient.java

```java
package tech.kayys.wayang.inference.providers.mcp.client;

```

### MCPConnection.java

```java
package tech.kayys.wayang.inference.providers.mcp.client;

```

Would you like me to continue with:
1. **MCP Tools Implementation** (tool execution, registry, result handling)
2. **MCP Resources & Prompts** (resource providers, prompt templates)
3. **MCP Provider Implementation** (integrating MCP as LLM provider)
4. **Advanced Memory Subsystem** (episodic, semantic, procedural memory)?


firefox
https://claude.ai/chat/10c8e7b4-9437-4f29-89cc-414bc12a39bf





I'll continue with implementing the **MCP Tools Implementation** - a critical component for tool execution, registry, and result handling in your AI agent workflow platform.

---

## 📦 MCP Tools Implementation

### 1. Core Tool Abstractions

#### MCPTool.java

```java
package tech.kayys.wayang.inference.providers.mcp.tools;

```

#### MCPToolResult.java

```java
package tech.kayys.wayang.inference.providers.mcp.tools;

```

### 2. Tool Registry & Management

#### MCPToolRegistry.java

```java
package tech.kayys.wayang.inference.providers.mcp.tools;

```

### 3. Tool Executor

#### MCPToolExecutor.java

```java
package tech.kayys.wayang.inference.providers.mcp.tools;

```

### 4. Tool Call Context

#### MCPToolCallContext.java

```java
package tech.kayys.wayang.inference.providers.mcp.tools;

import tech.kayys.wayang.inference.api.TenantContext;

```

---

Would you like me to continue with:
1. **MCP Resources & Prompts** (resource providers, prompt templates, resource management)
2. **MCP Provider Integration** (integrating MCP as an LLM provider in the inference engine)
3. **Advanced Memory Subsystem** (episodic, semantic, procedural memory with MCP integration)
4. **Integration Tests** (comprehensive test suite for MCP tools)?


I'll continue with **MCP Resources & Prompts** implementation - completing the full MCP integration for your AI agent workflow platform.

---

## 📦 MCP Resources & Prompts Implementation

### 1. Resource Abstractions

#### MCPResource.java

```java
package tech.kayys.wayang.inference.providers.mcp.resources;

```

#### MCPResourceContent.java

```java
package tech.kayys.wayang.inference.providers.mcp.resources;

```

### 2. Resource Registry & Provider

#### MCPResourceRegistry.java

```java
package tech.kayys.wayang.inference.providers.mcp.resources;

```

#### MCPResourceProvider.java

```java
package tech.kayys.wayang.inference.providers.mcp.resources;

```

#### MCPResourceCache.java

```java
package tech.kayys.wayang.inference.providers.mcp.resources;

```

### 3. Prompt Abstractions

#### MCPPrompt.java

```java
package tech.kayys.wayang.inference.providers.mcp.prompts;

```

#### MCPPromptMessage.java

```java
package tech.kayys.wayang.inference.providers.mcp.prompts;

```

#### MCPPromptResult.java

```java
package tech.kayys.wayang.inference.providers.mcp.prompts;

```

### 4. Prompt Provider

#### MCPPromptProvider.java

```java
package tech.kayys.wayang.inference.providers.mcp.prompts;

```

### 5. Exception Classes

```java
package tech.kayys.wayang.inference.providers.mcp.resources;




```

```java
package tech.kayys.wayang.inference.providers.mcp.prompts;



```

---

Would you like me to continue with:
1. **MCP Provider Integration** (integrating MCP as an LLM provider with full tool/resource/prompt support)
2. **Advanced Memory Subsystem** (episodic, semantic, procedural memory with MCP integration)
3. **Integration Tests & Examples** (comprehensive test suite showing full MCP workflow)
4. **REST API Layer** (exposing MCP capabilities via REST endpoints)?


I'll continue with **MCP Provider Integration** - integrating MCP as a full-featured LLM provider in the inference engine with tool/resource/prompt support.

---

## 📦 MCP Provider Integration

### 1. MCP Provider Implementation

#### MCPProvider.java

```java
package tech.kayys.wayang.inference.providers.mcp;

```

### 2. MCP Inference Context

#### MCPInferenceContext.java

```java
package tech.kayys.wayang.inference.providers.mcp;

import tech.kayys.wayang.inference.api.Message;

```

#### MCPInferenceResult.java

```java
package tech.kayys.wayang.inference.providers.mcp;

```

### 3. MCP Provider Configuration

#### MCPProviderConfiguration.java

```java
package tech.kayys.wayang.inference.providers.mcp;

```

### 4. MCP Provider Initializer

#### MCPProviderInitializer.java

```java
package tech.kayys.wayang.inference.providers.mcp;

```

### 5. Application Configuration Example

#### application.yml

```yaml

```

---

Would you like me to continue with:
1. **Advanced Memory Subsystem** (episodic, semantic, procedural memory with MCP integration)
2. **REST API Layer** (exposing MCP capabilities via REST endpoints with Swagger/OpenAPI)
3. **Integration Tests & Examples** (comprehensive test suite and usage examples)
4. **Plugin System Architecture** (making MCP a pluggable component in the inference kernel)?



I'll continue with **Advanced Memory Subsystem** - implementing episodic, semantic, and procedural memory with MCP integration for the AI agent workflow platform.

---

## 🧠 Advanced Memory Subsystem

### 1. Core Memory Abstractions

#### MemoryType.java

```java
package tech.kayys.wayang.inference.memory;

```

#### MemoryEntry.java

```java
package tech.kayys.wayang.inference.memory;

```

### 2. Memory Query & Retrieval

#### MemoryQuery.java

```java
package tech.kayys.wayang.inference.memory;

```

#### MemoryRankingStrategy.java

```java
package tech.kayys.wayang.inference.memory;

```

### 3. Memory Service Interface

#### MemoryService.java

```java
package tech.kayys.wayang.inference.memory;

```

---

Would you like me to continue with:
1. **Memory Service Implementation** (PostgreSQL + pgvector storage)
2. **Memory Embedder Service** (text embedding for semantic search)
3. **Memory Consolidation Service** (automatic memory merging and summarization)
4. **Integration with MCP** (using MCP resources as memory sources)?

firefox
https://claude.ai/chat/88b1a973-e4c9-40d1-9207-2d55943ebe9f




I'll create a comprehensive **Plugin System Architecture** that makes MCP and other components pluggable, followed by **REST API Layer** and **Advanced Memory Subsystem**.

---

## 🔌 Part 4: Plugin System Architecture

### 1. Core Plugin Abstractions

#### Plugin.java

```java
package tech.kayys.wayang.inference.plugin;

import io.smallrye.mutiny.Uni;

/**
 * Base interface for all inference plugins.
 * Plugins extend the inference engine with additional capabilities.
 */
public interface Plugin {

    /**
     * Unique plugin identifier (e.g., "mcp-tools", "rag-retriever")
     */
    String id();

    /**
     * Plugin display name
     */
    String name();

    /**
     * Plugin version (semantic versioning)
     */
    String version();

    /**
     * Plugin execution order (lower = earlier)
     * Used to determine plugin invocation sequence within a phase.
     */
    default int order() {
        return 100;
    }

    /**
     * Initialize plugin with engine context
     * Called once during engine startup
     */
    default Uni<Void> initialize(PluginContext context) {
        return Uni.createFrom().voidItem();
    }

    /**
     * Shutdown plugin gracefully
     * Called during engine shutdown
     */
    default Uni<Void> shutdown() {
        return Uni.createFrom().voidItem();
    }

    /**
     * Check if plugin is healthy
     */
    default PluginHealth health() {
        return PluginHealth.healthy();
    }

    /**
     * Get plugin metadata
     */
    default PluginMetadata metadata() {
        return PluginMetadata.builder()
            .id(id())
            .name(name())
            .version(version())
            .build();
    }
}
```

#### PluginContext.java

```java
package tech.kayys.wayang.inference.plugin;

import tech.kayys.wayang.inference.engine.EngineContext;

import java.util.Map;
import java.util.Optional;

/**
 * Context provided to plugins during initialization.
 * Gives plugins access to engine resources and configuration.
 */
public interface PluginContext {

    /**
     * Get engine context (global state)
     */
    EngineContext engineContext();

    /**
     * Get plugin configuration
     */
    Map<String, Object> config();

    /**
     * Get typed configuration value
     */
    <T> Optional<T> getConfig(String key, Class<T> type);

    /**
     * Get configuration value with default
     */
    <T> T getConfigOrDefault(String key, T defaultValue);

    /**
     * Check if plugin is enabled
     */
    boolean isEnabled();

    /**
     * Get plugin registry (for inter-plugin communication)
     */
    PluginRegistry registry();

    /**
     * Put shared data accessible to other plugins
     */
    void putSharedData(String key, Object value);

    /**
     * Get shared data from other plugins
     */
    <T> Optional<T> getSharedData(String key, Class<T> type);
}
```

#### PluginMetadata.java

```java
package tech.kayys.wayang.inference.plugin;

import java.util.*;

/**
 * Plugin metadata for discovery and documentation.
 */
public final class PluginMetadata {

    private final String id;
    private final String name;
    private final String version;
    private final String description;
    private final String author;
    private final Set<String> tags;
    private final Map<String, String> properties;
    private final List<PluginDependency> dependencies;

    private PluginMetadata(Builder builder) {
        this.id = builder.id;
        this.name = builder.name;
        this.version = builder.version;
        this.description = builder.description;
        this.author = builder.author;
        this.tags = Set.copyOf(builder.tags);
        this.properties = Map.copyOf(builder.properties);
        this.dependencies = List.copyOf(builder.dependencies);
    }

    // Getters
    public String getId() { return id; }
    public String getName() { return name; }
    public String getVersion() { return version; }
    public String getDescription() { return description; }
    public String getAuthor() { return author; }
    public Set<String> getTags() { return tags; }
    public Map<String, String> getProperties() { return properties; }
    public List<PluginDependency> getDependencies() { return dependencies; }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String id;
        private String name;
        private String version;
        private String description;
        private String author;
        private final Set<String> tags = new HashSet<>();
        private final Map<String, String> properties = new HashMap<>();
        private final List<PluginDependency> dependencies = new ArrayList<>();

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder version(String version) {
            this.version = version;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder author(String author) {
            this.author = author;
            return this;
        }

        public Builder tag(String tag) {
            this.tags.add(tag);
            return this;
        }

        public Builder tags(Set<String> tags) {
            this.tags.addAll(tags);
            return this;
        }

        public Builder property(String key, String value) {
            this.properties.put(key, value);
            return this;
        }

        public Builder dependency(PluginDependency dependency) {
            this.dependencies.add(dependency);
            return this;
        }

        public PluginMetadata build() {
            Objects.requireNonNull(id, "id is required");
            Objects.requireNonNull(name, "name is required");
            Objects.requireNonNull(version, "version is required");
            return new PluginMetadata(this);
        }
    }

    /**
     * Plugin dependency declaration
     */
    public record PluginDependency(
        String pluginId,
        String minVersion,
        String maxVersion,
        boolean optional
    ) {
        public PluginDependency {
            Objects.requireNonNull(pluginId, "pluginId");
        }

        public static PluginDependency required(String pluginId, String minVersion) {
            return new PluginDependency(pluginId, minVersion, null, false);
        }

        public static PluginDependency optional(String pluginId) {
            return new PluginDependency(pluginId, null, null, true);
        }
    }
}
```

#### PluginHealth.java

```java
package tech.kayys.wayang.inference.plugin;

import java.time.Instant;
import java.util.*;

/**
 * Health status of a plugin.
 */
public final class PluginHealth {

    public enum Status {
        HEALTHY,
        DEGRADED,
        UNHEALTHY,
        UNKNOWN
    }

    private final Status status;
    private final String message;
    private final Instant timestamp;
    private final Map<String, Object> details;

    private PluginHealth(
        Status status,
        String message,
        Map<String, Object> details
    ) {
        this.status = Objects.requireNonNull(status);
        this.message = message;
        this.timestamp = Instant.now();
        this.details = details != null
            ? Collections.unmodifiableMap(new HashMap<>(details))
            : Collections.emptyMap();
    }

    public Status getStatus() { return status; }
    public String getMessage() { return message; }
    public Instant getTimestamp() { return timestamp; }
    public Map<String, Object> getDetails() { return details; }

    public boolean isHealthy() {
        return status == Status.HEALTHY;
    }

    public static PluginHealth healthy() {
        return new PluginHealth(Status.HEALTHY, "Plugin is healthy", null);
    }

    public static PluginHealth healthy(String message) {
        return new PluginHealth(Status.HEALTHY, message, null);
    }

    public static PluginHealth degraded(String message) {
        return new PluginHealth(Status.DEGRADED, message, null);
    }

    public static PluginHealth degraded(String message, Map<String, Object> details) {
        return new PluginHealth(Status.DEGRADED, message, details);
    }

    public static PluginHealth unhealthy(String message) {
        return new PluginHealth(Status.UNHEALTHY, message, null);
    }

    public static PluginHealth unhealthy(String message, Map<String, Object> details) {
        return new PluginHealth(Status.UNHEALTHY, message, details);
    }

    public static PluginHealth unknown() {
        return new PluginHealth(Status.UNKNOWN, "Health status unknown", null);
    }

    @Override
    public String toString() {
        return "PluginHealth{" +
               "status=" + status +
               ", message='" + message + '\'' +
               ", timestamp=" + timestamp +
               '}';
    }
}
```

### 2. Phase-Bound Plugin System

#### InferencePhasePlugin.java

```java
package tech.kayys.wayang.inference.plugin;

import io.smallrye.mutiny.Uni;
import tech.kayys.wayang.inference.execution.ExecutionContext;
import tech.kayys.wayang.inference.pipeline.InferencePhase;

/**
 * Plugin that executes during a specific inference phase.
 * Phase plugins can modify execution context, validate inputs,
 * enrich outputs, or perform side effects.
 */
public interface InferencePhasePlugin extends Plugin {

    /**
     * The phase this plugin is bound to
     */
    InferencePhase phase();

    /**
     * Execute plugin logic for the given context
     * 
     * @param context Execution context (mutable)
     * @return Uni completing when plugin execution finishes
     */
    Uni<Void> execute(ExecutionContext context);

    /**
     * Check if plugin should execute for this context
     * Allows conditional plugin execution based on context state
     */
    default boolean shouldExecute(ExecutionContext context) {
        return true;
    }

    /**
     * Handle plugin execution failure
     * Return true to continue pipeline, false to halt
     */
    default boolean onFailure(ExecutionContext context, Throwable error) {
        context.setError(error);
        return false; // Halt by default
    }
}
```

#### ConfigurablePlugin.java

```java
package tech.kayys.wayang.inference.plugin;

import io.smallrye.mutiny.Uni;

import java.util.Map;

/**
 * Plugin that supports dynamic configuration updates.
 */
public interface ConfigurablePlugin extends Plugin {

    /**
     * Handle configuration update
     * Called when plugin configuration changes at runtime
     */
    Uni<Void> onConfigUpdate(Map<String, Object> newConfig);

    /**
     * Validate configuration before applying
     */
    default boolean validateConfig(Map<String, Object> config) {
        return true;
    }

    /**
     * Get current configuration
     */
    Map<String, Object> currentConfig();
}
```

### 3. Plugin Registry

#### PluginRegistry.java

```java
package tech.kayys.wayang.inference.plugin;

import tech.kayys.wayang.inference.pipeline.InferencePhase;

import java.util.List;
import java.util.Optional;

/**
 * Central registry for all inference plugins.
 * Manages plugin lifecycle, discovery, and execution order.
 */
public interface PluginRegistry {

    /**
     * Register a plugin
     */
    void register(Plugin plugin);

    /**
     * Unregister a plugin
     */
    void unregister(String pluginId);

    /**
     * Get plugin by ID
     */
    Optional<GolekPlugin> getPlugin(String pluginId);

    /**
     * Get all registered plugins
     */
    List<GolekPlugin> getAllPlugins();

    /**
     * Get plugins by type
     */
    <T extends Plugin> List<T> getPluginsByType(Class<T> type);

    /**
     * Get plugins for specific phase (ordered by execution order)
     */
    List<InferencePhasePlugin> getPluginsForPhase(InferencePhase phase);

    /**
     * Check if plugin is registered
     */
    boolean isRegistered(String pluginId);

    /**
     * Get plugin count
     */
    int count();

    /**
     * Clear all plugins (use with caution)
     */
    void clear();
}
```

#### DefaultPluginRegistry.java

```java
package tech.kayys.wayang.inference.plugin;

```

### 4. Plugin Loader & Discovery

#### PluginLoader.java

```java
package tech.kayys.wayang.inference.plugin;

```

### 5. Example: MCP as a Plugin

#### MCPToolPlugin.java

```java
package tech.kayys.wayang.inference.providers.mcp.plugins;

```

Would you like me to continue with:
1. **REST API Layer** (complete REST endpoints with OpenAPI/Swagger)
2. **Advanced Memory Subsystem** (episodic, semantic, procedural memory)
3. **More Plugin Examples** (validation, safety, routing plugins)



You're right to question this! The plugin system I designed is **NOT just for MCP** - it's a **general-purpose plugin architecture** for the entire inference engine. MCP is just **one type of plugin** that can be integrated.

Let me clarify the architecture:

## 🎯 Plugin System Scope

### **General Plugin System** (What I Built)
- Works for ANY extension to the inference engine
- Examples:
  - MCP integration (tools, resources, prompts)
  - Safety/guardrail plugins
  - RAG plugins
  - Validation plugins
  - Routing plugins
  - Observability plugins
  - Custom business logic plugins

### **MCP-Specific Integration**
MCP itself is a **protocol**, not a plugin system. We need to:
1. Use the plugin system to **integrate MCP capabilities**
2. MCP provides: tools, resources, prompts
3. Our plugins **wrap MCP protocol** to make it work within the inference engine

---

## 🔧 Corrected Architecture

Let me show you the **two-layer approach**:

### Layer 1: General Plugin System (Already Built ✅)
- `Plugin` interface
- `InferencePhasePlugin` interface
- `PluginRegistry`
- `PluginLoader`

### Layer 2: MCP Integration via Plugins (What We Need)

I'll now create **MCP-specific plugins** that use the general plugin system:

---

## 📦 MCP Integration Plugins

### 1. MCP Connection Plugin (Manages MCP Server Connections)

#### MCPConnectionPlugin.java

```java
package tech.kayys.wayang.inference.providers.mcp.plugins;

```

### 2. MCP Tool Plugin (Phase-Bound)

#### MCPToolPlugin.java

```java
package tech.kayys.wayang.inference.providers.mcp.plugins;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.execution.ExecutionContext;
import tech.kayys.wayang.inference.pipeline.InferencePhase;
import tech.kayys.wayang.inference.plugin.*;
import tech.kayys.wayang.inference.providers.mcp.tools.MCPToolExecutor;

import java.util.HashMap;
import java.util.Map;

/**
 * Plugin that executes MCP tools during inference.
 * Phase-bound to PRE_PROCESSING.
 * Depends on MCPConnectionPlugin being initialized.
 */
@ApplicationScoped
public class MCPToolPlugin implements InferencePhasePlugin, ConfigurablePlugin {

    private static final Logger LOG = Logger.getLogger(MCPToolPlugin.class);
    private static final String PLUGIN_ID = "mcp-tools";

    @Inject
    MCPToolExecutor toolExecutor;

    private Map<String, Object> config = new HashMap<>();
    private boolean enabled = true;

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String name() {
        return "MCP Tool Executor";
    }

    @Override
    public String version() {
        return "1.0.0";
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        return 50;
    }

    @Override
    public Uni<Void> initialize(PluginContext context) {
        this.config = new HashMap<>(context.config());
        this.enabled = context.getConfigOrDefault("enabled", true);
        
        LOG.infof("Initialized %s (enabled: %s)", name(), enabled);
        return Uni.createFrom().voidItem();
    }

    @Override
    public boolean shouldExecute(ExecutionContext context) {
        if (!enabled) {
            return false;
        }

        // Check if request has tool calls
        return context.getVariable("tool_calls", Map.class).isPresent();
    }

    @Override
    public Uni<Void> execute(ExecutionContext context) {
        @SuppressWarnings("unchecked")
        Map<String, Map<String, Object>> toolCalls = 
            (Map<String, Map<String, Object>>) context.getVariable("tool_calls", Map.class)
                .orElse(Map.of());

        if (toolCalls.isEmpty()) {
            return Uni.createFrom().voidItem();
        }

        LOG.debugf("Executing %d MCP tools", toolCalls.size());

        return toolExecutor.executeTools(toolCalls)
            .onItem().invoke(results -> {
                context.putVariable("mcp_tool_results", results);
                context.putMetadata("mcp_tools_count", results.size());
            })
            .replaceWithVoid();
    }

    @Override
    public Uni<Void> onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        return Uni.createFrom().voidItem();
    }

    @Override
    public Map<String, Object> currentConfig() {
        return new HashMap<>(config);
    }

    @Override
    public PluginMetadata metadata() {
        return PluginMetadata.builder()
            .id(id())
            .name(name())
            .version(version())
            .description("Executes MCP tools during inference")
            .author("Kayys Tech")
            .tag("mcp")
            .tag("tools")
            .dependency(PluginMetadata.PluginDependency.required("mcp-connections", "1.0.0"))
            .build();
    }
}
```

### 3. MCP Resource Plugin (Phase-Bound)

#### MCPResourcePlugin.java

```java
package tech.kayys.wayang.inference.providers.mcp.plugins;

```

---

## 🎯 Summary

**The plugin system is GENERAL-PURPOSE**, not MCP-specific:

✅ **General Plugin System** → Works for any inference extension  
✅ **MCP Integration** → Uses the plugin system to integrate MCP protocol  
✅ **Other Plugins** → Can add validation, safety, RAG, etc., using the same system

**MCP-specific components:**
- `MCPConnectionPlugin` - Manages MCP server connections
- `MCPToolPlugin` - Executes MCP tools
- `MCPResourcePlugin` - Fetches MCP resources
- `MCPPromptPlugin` - Executes MCP prompts (can be added)

Should I continue with:
1. **REST API Layer** (exposing the inference engine via REST with OpenAPI)
2. **More Plugin Examples** (Safety, Validation, RAG plugins)
3. **Advanced Memory Subsystem**


---

## 📦 Part 3: REST API Layer

### 1. Main Inference Resource


firefox
https://claude.ai/chat/d094f9dc-f3ba-489a-8265-82ffa9f2b4e3







# 🎯 Continuing Implementation

Ill implement the requested components in order: Model Router Service, Provider Configuration Management, Circuit Breaker, and REST API Layer.

---

## 📦 1. Model Router Service

### ModelRouterService.java

```java
package tech.kayys.wayang.inference.core.service;

```

### Supporting Classes

```java
package tech.kayys.wayang.inference.core.service;

import tech.kayys.wayang.inference.api.InferenceRequest;
import tech.kayys.wayang.inference.api.TenantContext;

```

```java
package tech.kayys.wayang.inference.core.service;

import tech.kayys.wayang.inference.provider.LLMProvider;

```

```java
package tech.kayys.wayang.inference.core.service;

```

```java
package tech.kayys.wayang.inference.core.service;


```

```java
package tech.kayys.wayang.inference.core.service;


```

### SelectionPolicy.java

```java
package tech.kayys.wayang.inference.core.service;

```

### HardwareDetector.java

```java
package tech.kayys.wayang.inference.core.service;

```

### RuntimeMetricsCache.java

```java
package tech.kayys.wayang.inference.core.service;

```

---

## 📦 4. Provider Configuration Management

### ProviderConfig.java

```java
package tech.kayys.wayang.inference.provider.core;

```

### ProviderConfigLoader.java

```java
package tech.kayys.wayang.inference.core.config;

```

### Example Provider Configuration Files

**config/providers/openai.yaml**
```yaml
id: openai
name: OpenAI Provider
description: OpenAI GPT models
version: 1.0.0
enabled: true
priority: 80
timeout: PT30S

properties:
  api.base.url: https://api.openai.com/v1
  api.version: v1
  max.retries: 3
  retry.delay.ms: 1000
  rate.limit.requests: 100
  rate.limit.window: PT1M
  models:
    - gpt-4
    - gpt-3.5-turbo
    - gpt-4-turbo
```

**config/providers/pytorch.yaml**
```yaml
id: pytorch
name: PyTorch Local Provider
description: Local PyTorch model execution
version: 1.0.0
enabled: true
priority: 90
timeout: PT60S

properties:
  models.path: ./models/pytorch
  device: cuda
  cuda.device.id: 0
  batch.size: 1
  num.threads: 4
  enable.quantization: false
  cache.models: true
  max.cached.models: 3
```

**config/providers/onnx.yaml**
```yaml
id: onnx
name: ONNX Runtime Provider
description: ONNX model execution
version: 1.0.0
enabled: true
priority: 85
timeout: PT45S

properties:
  models.path: ./models/onnx
  execution.provider: CUDA
  intra.op.threads: 4
  inter.op.threads: 2
  enable.profiling: false
  optimization.level: 3
```
