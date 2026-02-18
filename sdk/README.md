# Gollek SDK - Modular Architecture

This repository contains the modular SDK for the Gollek inference engine, designed with a clean separation between interface and implementation.

## Architecture

The SDK is divided into three main modules:

### 1. gollek-sdk-core
Contains the core interfaces and shared models that define the contract for the Gollek SDK.

- `GollekSdk` - The main interface defining all inference operations
- Shared models like `AsyncJobStatus`, `BatchInferenceRequest`
- Core exception hierarchy
- Factory for creating SDK instances

### 2. gollek-sdk-java-local
A local implementation of the SDK that runs within the same JVM as the inference engine.

- Direct access to internal services without HTTP overhead
- Optimized for performance when running embedded
- Uses CDI for dependency injection

### 3. gollek-sdk-java-remote
A remote implementation of the SDK that communicates with the inference engine via HTTP API.

- Communicates with the engine over HTTP/HTTPS
- Handles authentication, retries, and error mapping
- Suitable for external applications

## Usage

### For Local (Embedded) Usage:
```java
import tech.kayys.gollek.sdk.factory.GollekSdkFactory;
import tech.kayys.gollek.sdk.core.GollekSdk;

GollekSdk sdk = GollekSdkFactory.createLocalSdk();
```

### For Remote (HTTP) Usage:
```java
import tech.kayys.gollek.sdk.factory.GollekSdkFactory;
import tech.kayys.gollek.sdk.core.GollekSdk;

GollekSdk sdk = GollekSdkFactory.createRemoteSdk(
    "https://api.gollek.example.com",
    "your-api-key"
);
```

### Using the SDK:
```java
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.sdk.core.GollekSdk;

// Create an inference request
var request = InferenceRequest.builder()
    .model("llama3:latest")
    .userMessage("Hello, how are you?")
    .temperature(0.7)
    .maxTokens(100)
    .build();

// Execute the request
InferenceResponse response = sdk.createCompletion(request);
System.out.println("Response: " + response.getContent());
```

### API Key Note

Multi-tenancy is resolved on the backend using your API key. For community/standalone deployments, use the API key `"community"`.

## Benefits of This Architecture

1. **Flexibility**: Choose the appropriate implementation based on your deployment scenario
2. **Performance**: Local implementation avoids HTTP overhead when running embedded
3. **Maintainability**: Clear separation of concerns between interface and implementation
4. **Testability**: Easy to mock the core interface for testing
5. **Consistency**: Same API regardless of implementation used
