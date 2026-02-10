# Wayang Tool Core

Core module for Wayang's Tool Architecture. Defines the standard interfaces (SPI) for defining, registering, and executing tools.

## Architecture

The Tool Architecture consists of three main components:

1.  **Tool SPI (`tech.kayys.wayang.tool.spi.Tool`)**: Interface that all tools must implement.
2.  **Tool Registry (`tech.kayys.wayang.tool.spi.ToolRegistry`)**: Central repository for discovering available tools.
3.  **Tool Executor (`tech.kayys.wayang.tool.spi.ToolExecutor`)**: Strategy for executing tools.

## Usage

### Defining a Tool

Implement the `Tool` interface:

```java
public class MyTool implements Tool {
    @Override
    public String id() { return "my-tool"; }
    
    @Override
    public Uni<Map<String, Object>> execute(Map<String, Object> args, Map<String, Object> context) {
        // Logic
        return Uni.createFrom().item(Map.of("result", "success"));
    }
    // ... implement other methods
}
```

### Registering a Tool

Inject `ToolRegistry` and register your tool (or mark your Tool bean as `@ApplicationScoped` if using automatic discovery features in the future):

```java
@Inject
ToolRegistry registry;

void init() {
    registry.register(new MyTool());
}
```

### Executing a Tool

Inject `ToolExecutor` to execute tools by ID:

```java
@Inject
ToolExecutor executor;

void run() {
    executor.execute("my-tool", args, context)
        .subscribe().with(result -> System.out.println(result));
}
```

## Integration

This module is used by `wayang-agent-core-executor` to provide tool capabilities to agents. Implementations of specific tool protocols (like MCP, UTCP) should depend on this module and provide their own `Tool` implementations.
