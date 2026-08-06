# Gollek Java SDK

The Gollek Java SDK provides a comprehensive client for interacting with the Gollek inference engine. It offers both synchronous and asynchronous methods for inference operations, along with support for streaming, batch processing, and async job execution.

## Installation

Add the following dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-sdk-java</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Quick Start

### Creating a Client

```java
import tech.kayys.gollek.client.GollekClient;

GollekClient client = GollekClient.builder()
    .baseUrl("https://api.gollek.example.com")
    .apiKey("your-api-key")
    .build();
```

### Simple Inference

```java
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.client.builder.InferenceRequest;

var request = InferenceRequest.builder()
    .model("llama3:latest")
    .userMessage("Hello, how are you?")
    .temperature(0.7)
    .maxTokens(100)
    .build();

var response = client.createCompletion(request);
System.out.println("Response: " + response.getContent());
```

### Streaming Inference

```java
import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

Multi<StreamingInferenceChunk> stream = client.streamCompletion(request);

stream.subscribe().with(
    chunk -> System.out.println("Received: " + chunk.getDelta()),
    failure -> System.err.println("Error: " + failure.getMessage())
);
```

### Agent/OpenAI-Compatible Streaming

Use the typed agent stream helpers when integrating Gollek as a serving engine
behind an agent framework. The helper calls the OpenAI-compatible endpoints,
sets `stream: true`, and turns SSE payloads into `AgentStreamEvent` objects.

```java
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.client.agent.AgentRequestOptions;
import tech.kayys.gollek.client.agent.AgentStreamAccumulator;
import tech.kayys.gollek.client.agent.AgentStreamEvent;

AgentRequestOptions trace = AgentRequestOptions.builder()
    .requestId("req-123")
    .traceId("trace-456")
    .sessionId("conversation-789")
    .userId("user-abc")
    .build();
AgentStreamAccumulator accumulator = new AgentStreamAccumulator();

client.streamOpenAiChat(Map.of(
    "model", "demo-model",
    "messages", List.of(Map.of("role", "user", "content", "Stream answer")),
    "stream_options", Map.of("include_usage", true)
), trace).subscribe().with(event -> {
    AgentStreamAccumulator.Snapshot state = accumulator.accept(event);
    if (event.hasDelta()) {
        System.out.print(event.delta());
    }
    if (event.hasToolCalls()) {
        event.toolCalls().forEach(call -> {
            try {
                Map<String, Object> args = call.argumentsMap();
                System.out.println("tool: " + call.name() + " args: " + args);
            } catch (Exception e) {
                System.err.println("invalid tool arguments: " + call.arguments());
            }
        });
    }
    if (event.isCompleted() && event.usage() != null) {
        System.out.println("final answer: " + state.outputText());
        System.out.println("total tokens: " + state.usage().totalTokens());
    }
});
```

### Agent Integration Discovery

Use `client.agent()` before wiring an agent runtime to discover the serving
contract, validate request/tool schemas, and fetch MCP tool definitions. These
helpers are read-only: Gollek exposes schemas and validation, while the caller's
agent framework owns planning, authorization, and tool execution.

The same classes are also published as the lightweight `gollek-sdk-agent`
artifact for external orchestrators that do not need the full remote SDK graph.
Call `servingReadiness(...)` when you want the SDK to perform the discovery and
dry-run validation calls as one serving-only preflight report.
Use `AgentServingPreflightRequest` when a route should skip optional stages such
as MCP discovery or tool validation.
Use `servingPreflightReadiness(...)` when the server exposes
`POST /v1/agent/preflight` and you want Gollek to compose those same checks in a
single request.

```java
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.client.agent.AgentCapabilitiesView;
import tech.kayys.gollek.client.agent.AgentEmbeddingView;
import tech.kayys.gollek.client.agent.AgentMcpDiscoveryView;
import tech.kayys.gollek.client.agent.AgentModelCapabilitiesView;
import tech.kayys.gollek.client.agent.AgentReadinessIssueCatalogView;
import tech.kayys.gollek.client.agent.AgentResponseView;
import tech.kayys.gollek.client.agent.AgentServingContract;
import tech.kayys.gollek.client.agent.AgentServingPreflightRequest;
import tech.kayys.gollek.client.agent.AgentServingReadinessReport;
import tech.kayys.gollek.client.agent.AgentStreamAccumulator;
import tech.kayys.gollek.client.agent.AgentToolValidationView;
import tech.kayys.gollek.client.agent.AgentValidationView;

AgentCapabilitiesView capabilities = client.agent().capabilitiesView();
if (!capabilities.hasRequiredAgentServingCapabilities()) {
    throw new IllegalStateException("Gollek capabilities mismatch: "
        + capabilities.agentServingCapabilityIssues());
}

AgentServingContract contract = client.agent().contractView();
if (!contract.hasRequiredServingBoundary()) {
    throw new IllegalStateException("Gollek serving boundary mismatch: " + contract.servingBoundaryIssues());
}

AgentReadinessIssueCatalogView issueCatalog = client.agent().readinessIssueCatalogView();
issueCatalog.find("TOOL_DEFINITIONS_INVALID").ifPresent(issue ->
    System.out.println(issue.code() + ": " + issue.remediation()));

AgentModelCapabilitiesView model = client.agent().modelCapabilitiesView("demo-model");
if (!model.hasRequiredAgentServingRoute()) {
    throw new IllegalStateException("Model cannot serve this agent route: "
        + model.agentServingRouteIssues());
}

AgentMcpDiscoveryView toolDefinitions = client.agent().mcpToolsView(true, true);
if (!toolDefinitions.discoveryOnly()) {
    throw new IllegalStateException("MCP discovery endpoint crossed the serving boundary");
}

AgentToolValidationView toolCheck = client.agent().validateToolsView(
    Map.of("tools", toolDefinitions.openAiToolDefinitions()));
if (toolCheck.hasWarnings()) {
    System.out.println("tool schema warnings: " + toolCheck.warnings());
}

AgentEmbeddingView queryEmbedding = client.agent().createEmbeddingView(Map.of(
    "model", "demo-embed",
    "input", List.of("Question or search query for caller-owned retrieval")
), trace);
List<Double> queryVector = queryEmbedding.firstVector();
// Use queryVector with your vector store, then pass selected chunks as rag_context.

Map<String, Object> chatRequest = Map.of(
    "model", "demo-model",
    "messages", List.of(Map.of("role", "user", "content", "Answer with the discovered tools")),
    "rag_context", List.of(Map.of(
        "title", "Retrieved chunk",
        "text", "Relevant retrieved document text.")),
    "tools", toolDefinitions.openAiToolDefinitions()
);

AgentValidationView dryRun = client.agent().validateRequestView("chat", chatRequest, trace);
if (!dryRun.validationOnly() || dryRun.modelInvoked()) {
    throw new IllegalStateException("Validation endpoint crossed the serving boundary");
}

AgentServingReadinessReport readiness = AgentServingReadinessReport.builder()
    .capabilities(capabilities)
    .contract(contract)
    .modelRoute(model)
    .mcpDiscovery(toolDefinitions, true)
    .toolValidation(toolCheck, true)
    .requestValidation(dryRun, true)
    .build();
readiness.requireReady("Gollek serving preflight failed");
Map<String, Object> readinessReport = readiness.toReport();
System.out.println("readiness: " + readinessReport.get("status")
    + " checks=" + readinessReport.get("checks"));

AgentResponseView chatView = client.agent().createChatCompletionView(chatRequest, trace);
chatView.toolCalls().forEach(call -> {
    try {
        Map<String, Object> args = call.argumentsMap();
        System.out.println("tool: " + call.name() + " args: " + args);
    } catch (Exception e) {
        System.err.println("invalid tool arguments: " + call.arguments());
    }
});

AgentValidationView validation = client.agent().validateRequestView("chat", Map.of(
    "model", "demo-model",
    "messages", List.of(Map.of("role", "user", "content", "Validate this request")),
    "tools", toolDefinitions.openAiToolDefinitions()
), trace);
System.out.println("validated surface: " + validation.surface() + " model: " + validation.model());

AgentStreamAccumulator.Snapshot stream = client.agent().streamChatCompletion(Map.of(
    "model", "demo-model",
    "messages", List.of(Map.of("role", "user", "content", "Stream this request")),
    "tools", toolDefinitions.openAiToolDefinitions()
), trace, event -> {
    if (event.hasDelta()) {
        System.out.print(event.delta());
    }
});
```

For the server-side preflight endpoint:

```java
AgentServingReadinessReport readiness = client.agent().servingPreflightReadiness(
    AgentServingPreflightRequest.builder()
        .modelId("demo-model")
        .surface("chat")
        .request(chatRequest)
        .build());
readiness.requireReady("Gollek serving preflight failed");
Map<String, Object> ciReport = readiness.toReport();
```

`toReport()` emits `object: "gollek.agent_readiness_report"`, status, issue
counts, check counts, per-check messages, `issue_hints`, and the
validation-only boundary. Use `toMetadata()` when a caller only needs compact
status fields and remediation metadata for request logs or dashboards.
The SDK also maps server-side `check_results` from `/v1/agent/preflight`, so
Java callers receive the same per-area readiness status that CLI and REST
clients see. Use `readiness.checkResults()` or `readiness.check("preflight")`
for typed `Check` objects with `status`, `ready`, `requested`, `skipped()`,
`blockingMessages()`, and `warningMessages()`; use `readiness.checks()` when
you need the raw report-compatible map.

### Async Job Submission

```java
// Submit a long-running job
String jobId = client.submitAsyncJob(request);
System.out.println("Job submitted with ID: " + jobId);

// Check job status
var status = client.getJobStatus(jobId);
System.out.println("Job status: " + status.getStatus());

// Wait for job to complete
var finalStatus = client.waitForJob(jobId,
    Duration.ofMinutes(10),
    Duration.ofSeconds(5));
```

### Batch Inference

```java
import tech.kayys.gollek.client.model.BatchInferenceRequest;

var batchRequest = BatchInferenceRequest.builder()
    .requests(Arrays.asList(request1, request2, request3))
    .maxConcurrent(5)
    .build();

var responses = client.batchInference(batchRequest);
responses.forEach(response ->
    System.out.println("Response: " + response.getContent()));
```

## Configuration Options

The client supports various configuration options:

- `baseUrl`: The base URL of the Gollek API
- `apiKey`: Your API key for authentication
- `connectTimeout`: Connection timeout duration
- `sslContext`: Custom SSL context for HTTPS connections

## API Key Note

Multi-tenancy is resolved on the backend using your API key. For community/standalone deployments, use the API key `"community"`.

## Error Handling

The SDK provides specific exception types for different error conditions:

- `AuthenticationException`: Authentication failures
- `RateLimitException`: Rate limiting errors
- `ModelException`: Model-specific errors
- `GollekClientException`: General client errors

Example error handling:

```java
try {
    var response = client.createCompletion(request);
} catch (AuthenticationException e) {
    System.err.println("Authentication failed: " + e.getMessage());
} catch (RateLimitException e) {
    System.err.println("Rate limited, retry after: " + e.getRetryAfterSeconds() + " seconds");
} catch (ModelException e) {
    System.err.println("Model error: " + e.getMessage() + ", model: " + e.getModelId());
} catch (GollekClientException e) {
    System.err.println("Client error: " + e.getMessage());
}
```

## Advanced Usage

### Tool Usage

```java
import tech.kayys.gollek.spi.tool.ToolDefinition;

var tool = ToolDefinition.builder()
    .name("get_weather")
    .description("Get weather information for a city")
    .parameter("city", Map.of("type", "string", "description", "City name"))
    .build();

var request = InferenceRequest.builder()
    .model("llama3:latest")
    .userMessage("What's the weather in Tokyo?")
    .tool(tool)
    .build();

var response = client.createCompletion(request);
```

### Custom Parameters

```java
var request = InferenceRequest.builder()
    .model("llama3:latest")
    .userMessage("Summarize this document")
    .parameter("temperature", 0.5)
    .parameter("top_p", 0.9)
    .parameter("repetition_penalty", 1.1)
    .maxTokens(500)
    .build();
```

### Model Operations

#### List Available Models

```java
import tech.kayys.gollek.sdk.core.model.ModelInfo;

// List all models
List<ModelInfo> models = client.listModels();
models.forEach(model ->
    System.out.println("Model: " + model.getId() + ", Size: " + model.getSize()));

// List models with pagination
List<ModelInfo> pagedModels = client.listModels(0, 10);
```

#### Get Model Information

```java
import java.util.Optional;
import tech.kayys.gollek.sdk.core.model.ModelInfo;

Optional<ModelInfo> modelInfo = client.getModelInfo("llama3:latest");
if (modelInfo.isPresent()) {
    System.out.println("Model: " + modelInfo.get().getId());
    System.out.println("Size: " + modelInfo.get().getSize());
    System.out.println("Description: " + modelInfo.get().getDescription());
} else {
    System.out.println("Model not found");
}
```

#### Pull a Model

```java
import tech.kayys.gollek.sdk.core.model.PullProgress;

// Pull a model with progress tracking
client.pullModel("llama3:8b", progress -> {
    System.out.printf("Progress: %.2f%% - Status: %s%n",
        progress.getPercentage(),
        progress.getStatus());
    System.out.println("Message: " + progress.getMessage());
});
```

#### Delete a Model

```java
// Delete a model
try {
    client.deleteModel("llama3:old-version");
    System.out.println("Model deleted successfully");
} catch (GollekClientException e) {
    System.err.println("Failed to delete model: " + e.getMessage());
}
```

## Best Practices

1. **Reuse Client Instances**: Create a single client instance and reuse it across your application
2. **Handle Errors Appropriately**: Implement proper error handling for different exception types
3. **Use Appropriate Timeouts**: Configure timeouts based on your use case
4. **Monitor Rate Limits**: Implement retry logic with exponential backoff for rate-limited requests
5. **Secure API Keys**: Store API keys securely and never hardcode them

## Support

For support, please check the [official documentation](https://docs.gollek.example.com) or contact our support team.
