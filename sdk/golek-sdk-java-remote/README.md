# Golek Java SDK

The Golek Java SDK provides a comprehensive client for interacting with the Golek inference engine. It offers both synchronous and asynchronous methods for inference operations, along with support for streaming, batch processing, and async job execution.

## Installation

Add the following dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>tech.kayys.golek</groupId>
    <artifactId>golek-sdk-java</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Quick Start

### Creating a Client

```java
import tech.kayys.golek.client.GolekClient;

GolekClient client = GolekClient.builder()
    .baseUrl("https://api.golek.example.com")
    .apiKey("your-api-key")
    .defaultTenantId("default")
    .build();
```

### Simple Inference

```java
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.client.builder.InferenceRequestBuilder;

var request = InferenceRequestBuilder.builder()
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
import tech.kayys.golek.spi.stream.StreamChunk;

Multi<StreamChunk> stream = client.streamCompletion(request);

stream.subscribe().with(
    chunk -> System.out.println("Received: " + chunk.getContent()),
    failure -> System.err.println("Error: " + failure.getMessage())
);
```

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
import tech.kayys.golek.client.model.BatchInferenceRequest;

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

- `baseUrl`: The base URL of the Golek API
- `apiKey`: Your API key for authentication
- `defaultTenantId`: The default tenant ID to use (only required when multi-tenancy is enabled)
- `connectTimeout`: Connection timeout duration
- `sslContext`: Custom SSL context for HTTPS connections

## Multi-Tenancy Note

Golek is single-tenant by default. Multi-tenancy is enabled only when `tenant-golek-ext` is added or `wayang.multitenancy.enabled=true` is set. In single-tenant mode, `defaultTenantId` can remain `"default"` and the API does not require `X-Tenant-ID`.

## Error Handling

The SDK provides specific exception types for different error conditions:

- `AuthenticationException`: Authentication failures
- `RateLimitException`: Rate limiting errors
- `ModelException`: Model-specific errors
- `GolekClientException`: General client errors

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
} catch (GolekClientException e) {
    System.err.println("Client error: " + e.getMessage());
}
```

## Advanced Usage

### Tool Usage

```java
import tech.kayys.golek.spi.tool.ToolDefinition;

var tool = ToolDefinition.builder()
    .name("get_weather")
    .description("Get weather information for a city")
    .parameter("city", Map.of("type", "string", "description", "City name"))
    .build();

var request = InferenceRequestBuilder.builder()
    .model("llama3:latest")
    .userMessage("What's the weather in Tokyo?")
    .tool(tool)
    .build();

var response = client.createCompletion(request);
```

### Custom Parameters

```java
var request = InferenceRequestBuilder.builder()
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
import tech.kayys.golek.sdk.core.model.ModelInfo;

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
import tech.kayys.golek.sdk.core.model.ModelInfo;

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
import tech.kayys.golek.sdk.core.model.PullProgress;

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
} catch (GolekClientException e) {
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

For support, please check the [official documentation](https://docs.golek.example.com) or contact our support team.
