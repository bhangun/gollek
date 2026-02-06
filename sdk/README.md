# Golek SDK - Modular Architecture

This repository contains the modular SDK for the Golek inference engine, designed with a clean separation between interface and implementation.

## Architecture

The SDK is divided into three main modules:

### 1. golek-sdk-core
Contains the core interfaces and shared models that define the contract for the Golek SDK.

- `GolekSdk` - The main interface defining all inference operations
- Shared models like `AsyncJobStatus`, `BatchInferenceRequest`
- Core exception hierarchy
- Factory for creating SDK instances

### 2. golek-sdk-java-local
A local implementation of the SDK that runs within the same JVM as the inference engine.

- Direct access to internal services without HTTP overhead
- Optimized for performance when running embedded
- Uses CDI for dependency injection

### 3. golek-sdk-java-remote
A remote implementation of the SDK that communicates with the inference engine via HTTP API.

- Communicates with the engine over HTTP/HTTPS
- Handles authentication, retries, and error mapping
- Suitable for external applications

## Usage

### For Local (Embedded) Usage:
```java
import tech.kayys.golek.sdk.factory.GolekSdkFactory;
import tech.kayys.golek.sdk.core.GolekSdk;

GolekSdk sdk = GolekSdkFactory.createLocalSdk();
```

### For Remote (HTTP) Usage:
```java
import tech.kayys.golek.sdk.factory.GolekSdkFactory;
import tech.kayys.golek.sdk.core.GolekSdk;

GolekSdk sdk = GolekSdkFactory.createRemoteSdk(
    "https://api.golek.example.com",
    "your-api-key",
    "default"
);
```

### Using the SDK:
```java
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.sdk.core.GolekSdk;

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

### Multi-Tenancy Note

Golek is single-tenant by default. `defaultTenantId` is only required when multi-tenancy is enabled (via `tenant-golek-ext` or `wayang.multitenancy.enabled=true`). In single-tenant mode, you can leave it as `"default"`.

## Benefits of This Architecture

1. **Flexibility**: Choose the appropriate implementation based on your deployment scenario
2. **Performance**: Local implementation avoids HTTP overhead when running embedded
3. **Maintainability**: Clear separation of concerns between interface and implementation
4. **Testability**: Easy to mock the core interface for testing
5. **Consistency**: Same API regardless of implementation used
