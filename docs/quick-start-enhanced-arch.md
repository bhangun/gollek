# Quick Start Guide: Enhanced Architecture

## Overview

This guide shows how to use the enhanced architecture for the Golek inference platform. The new architecture provides improved modularity, plugin support, configuration management, observability, and reliability.

## Getting Started

### 1. System Initialization

To initialize the system with the enhanced architecture:

```java
import tech.kayys.golek.core.module.SystemModule;

// Create and initialize the system module
SystemModule systemModule = new SystemModule();
systemModule.initialize();
systemModule.start();

// Access core components
PluginManager pluginManager = systemModule.getPluginManager();
EnhancedConfigurationManager configManager = systemModule.getEnhancedConfigurationManager();
ReliabilityManager reliabilityManager = systemModule.getReliabilityManager();
```

### 2. Creating a Plugin

To create a custom plugin:

```java
import tech.kayys.golek.core.plugin.AbstractPlugin;
import tech.kayys.golek.core.plugin.PluginContext;

public class MyCustomPlugin extends AbstractPlugin {
    
    public MyCustomPlugin() {
        super("mycompany.my-custom-plugin", "My Custom Plugin", "1.0.0");
    }
    
    @Override
    protected void onInitialize(PluginContext context) {
        // Initialize your plugin here
        System.out.println("MyCustomPlugin initialized");
    }
    
    @Override
    protected void onStart() {
        // Start your plugin functionality
        System.out.println("MyCustomPlugin started");
    }
    
    @Override
    protected void onStop() {
        // Clean up resources
        System.out.println("MyCustomPlugin stopped");
    }
    
    @Override
    protected void onDestroy() {
        // Final cleanup
        System.out.println("MyCustomPlugin destroyed");
    }
}

// Register the plugin
PluginManager pluginManager = systemModule.getPluginManager();
pluginManager.registerPlugin(new MyCustomPlugin());
```

### 3. Using Configuration Management

The enhanced configuration system supports multiple sources:

```java
// Get configuration with fallback chain
String apiEndpoint = configManager.getProperty(
    "myplugin.api.endpoint",           // key
    "mycompany.my-custom-plugin",      // plugin ID (optional)
    String.class,                      // type
    "https://default.example.com"      // default value
);

// Set runtime configuration
configManager.setRuntimeProperty("myplugin.feature.enabled", true);

// Access configuration in your plugin
String value = context.getConfiguration().getProperty("some.key", String.class);
```

### 4. Implementing Reliable Operations

Use the reliability manager for resilient operations:

```java
ReliabilityManager reliabilityManager = systemModule.getReliabilityManager();

// Execute with circuit breaker protection
Uni<String> result = reliabilityManager.executeWithCircuitBreaker(
    "my-operation",
    () -> performRiskyOperation()
);

// Execute with fallback
Uni<String> resultWithFallback = reliabilityManager.executeWithFallback(
    "my-operation",
    () -> performPrimaryOperation(),
    () -> performFallbackOperation()
);

// Execute with retry logic
Uni<String> resultWithRetry = reliabilityManager.executeWithRetry(
    "my-operation",
    () -> performOperation(),
    3,                                    // max retries
    Duration.ofSeconds(1)                 // initial delay
);

// Execute with all resilience patterns
Uni<String> resilientResult = reliabilityManager.executeResiliently(
    "my-operation",
    () -> performPrimaryOperation(),
    () -> performFallbackOperation(),
    3,                                    // max retries
    Duration.ofSeconds(1)                 // retry delay
);
```

### 5. Using Observability

Track operations with metrics and tracing:

```java
ObservabilityManager obsManager = systemModule.getObservabilityManager();

// Record metrics
Attributes attributes = Attributes.of(
    AttributeKey.stringKey("operation"), "my-operation",
    AttributeKey.stringKey("result"), "success"
);
obsManager.recordMetric("myplugin.operation.duration", 123.45, attributes);

// Trace operations
try {
    String result = obsManager.traced(
        "my-operation",
        SpanKind.INTERNAL,
        Attributes.of(AttributeKey.stringKey("operation"), "my-operation"),
        () -> performTracedOperation()
    );
} catch (Exception e) {
    // Exception will be automatically recorded in the trace
    throw e;
}
```

### 6. Service Registration

Register and access services through the service registry:

```java
// Assuming you have access to the service registry
ServiceRegistry serviceRegistry = new ServiceRegistry();

// Register a service
MyService myService = new MyServiceImpl();
serviceRegistry.registerService(MyService.class, myService);

// Register a named service
serviceRegistry.registerNamedService("my-named-service", myService);

// Get a service
MyService retrievedService = serviceRegistry.getService(MyService.class);

// Get a named service
Object namedService = serviceRegistry.getNamedService("my-named-service");
```

## Best Practices

### Plugin Development
1. Always implement proper lifecycle methods
2. Use the configuration system instead of hardcoded values
3. Handle failures gracefully with fallback strategies
4. Use the observability system to track important metrics
5. Follow the single responsibility principle

### Configuration Management
1. Use meaningful configuration keys with namespace prefixes
2. Provide sensible default values
3. Validate configuration values during initialization
4. Document configuration options clearly

### Reliability
1. Apply circuit breakers to external dependencies
2. Implement appropriate fallback strategies
3. Use timeouts to prevent hanging operations
4. Monitor circuit breaker metrics for operational insights

## Example: Complete Plugin Implementation

```java
import tech.kayys.golek.core.plugin.AbstractPlugin;
import tech.kayys.golek.core.plugin.ExtensionPoint;
import tech.kayys.golek.core.plugin.PluginContext;
import io.smallrye.mutiny.Uni;

public class ExampleProcessingPlugin extends AbstractPlugin {
    
    private ProcessingService processingService;
    
    public ExampleProcessingPlugin() {
        super("mycompany.example-processing", "Example Processing Plugin", "1.0.0");
    }
    
    @Override
    protected void onInitialize(PluginContext context) {
        // Initialize with configuration
        String endpoint = context.getConfiguration()
            .getProperty("endpoint", String.class, "https://api.example.com");
        
        int maxRetries = context.getConfiguration()
            .getProperty("max-retries", Integer.class, 3);
        
        // Create service with configuration
        this.processingService = new ProcessingService(endpoint, maxRetries);
    }
    
    @Override
    protected void onStart() {
        // Start the service
        processingService.start();
    }
    
    @Override
    protected void onStop() {
        // Stop the service gracefully
        if (processingService != null) {
            processingService.stop();
        }
    }
    
    @Override
    protected void onDestroy() {
        // Clean up resources
        this.processingService = null;
    }
    
    // Method to perform processing with resilience
    public Uni<String> processRequest(String input) {
        return executeResiliently(
            "process-request",
            () -> processingService.process(input),
            () -> processingService.fallbackProcess(input),
            3,
            java.time.Duration.ofSeconds(1)
        );
    }
    
    // Example extension point
    @Override
    public ExtensionPoint[] getExtensionPoints() {
        return new ExtensionPoint[] {
            new ProcessingExtensionPoint(this)
        };
    }
}
```

## Conclusion

The enhanced architecture provides a solid foundation for building robust, scalable, and maintainable inference platform extensions. By following the patterns and best practices outlined in this guide, you can leverage the full power of the modular, plugin-based architecture while ensuring reliability and observability.