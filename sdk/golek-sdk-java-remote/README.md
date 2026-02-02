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
    .defaultTenantId("your-tenant-id")
    .build();
```

### Simple Inference

```java
import tech.kayys.golek.api.Message;
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
import tech.kayys.golek.api.stream.StreamChunk;

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
- `defaultTenantId`: The default tenant ID to use
- `connectTimeout`: Connection timeout duration
- `sslContext`: Custom SSL context for HTTPS connections

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
import tech.kayys.golek.api.tool.ToolDefinition;

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

## Best Practices

1. **Reuse Client Instances**: Create a single client instance and reuse it across your application
2. **Handle Errors Appropriately**: Implement proper error handling for different exception types
3. **Use Appropriate Timeouts**: Configure timeouts based on your use case
4. **Monitor Rate Limits**: Implement retry logic with exponential backoff for rate-limited requests
5. **Secure API Keys**: Store API keys securely and never hardcode them

## Support

For support, please check the [official documentation](https://docs.golek.example.com) or contact our support team.