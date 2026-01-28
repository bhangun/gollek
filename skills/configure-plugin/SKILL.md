---
name: configure-plugin
description: Build and deploy custom inference-golek plugins (model-router, quota, content-safety) for extending platform capabilities
metadata:
  short-description: Extend platform with custom plugins
  category: plugins
  difficulty: intermediate
---

# Configure Plugin Skill

Create, build, and deploy custom inference-golek plugins to extend platform capabilities without modifying core code.

## When to Use

- You need custom model routing logic
- You want to add quota/rate limiting policies
- You need content safety checks
- You want to extend inference behavior

## Supported Plugin Types

1. **Model Router** - Custom model selection logic
2. **Quota Manager** - Tenant/user quota enforcement
3. **Content Safety** - Input/output validation
4. **Custom Executor** - New inference backend

## Prerequisites

- Maven 3.8+
- JDK 17+
- inference-golek plugin API on classpath

## Steps

### 1. Create Plugin Project

```bash
mvn archetype:generate \
  -DgroupId=com.example \
  -DartifactId=my-inference-plugin \
  -DpackageName=com.example.inference.plugin \
  -Dversion=1.0.0
```

### 2. Add Plugin Dependency

```xml
<dependency>
  <groupId>tech.kayys.wayang.inference</groupId>
  <artifactId>golek-plugin-api</artifactId>
  <version>1.0.0</version>
  <scope>provided</scope>
</dependency>
```

### 3. Implement Plugin Interface

```java
package com.example.inference.plugin;

import tech.kayys.wayang.inference.plugin.spi.InferencePlugin;
import tech.kayys.wayang.inference.api.ProviderRequest;
import io.smallrye.mutiny.Uni;

public class MyModelRouterPlugin implements InferencePlugin {
  
  @Override
  public String getId() {
    return "my-model-router";
  }
  
  @Override
  public Uni<ProviderRequest> preProcess(ProviderRequest request) {
    // Custom routing logic
    if (request.isBudgetConstrained()) {
      request.setModelId("efficient-model");  // Route to cheaper model
    }
    return Uni.createFrom().item(request);
  }
  
  @Override
  public void initialize(Map<String, Object> config) {
    // Initialize plugin
  }
}
```

### 4. Create Plugin Metadata

```java
// Add to META-INF/services/tech.kayys.wayang.inference.plugin.spi.InferencePlugin
com.example.inference.plugin.MyModelRouterPlugin
```

### 5. Build Plugin JAR

```bash
mvn clean package -DskipTests
```

### 6. Deploy Plugin

```bash
# Copy to plugin directory
cp target/my-inference-plugin-1.0.0.jar \
   /opt/inference-golek/plugins/

# Or register dynamically
curl -X POST http://localhost:8080/api/plugins/register \
  -H "Content-Type: application/json" \
  -d '{
    "pluginId": "my-model-router",
    "jarPath": "/opt/plugins/my-inference-plugin-1.0.0.jar",
    "enabled": true
  }'
```

## Example: Quota Plugin

```java
public class QuotaPlugin implements InferencePlugin {
  
  private Map<String, QuotaManager> quotas = new ConcurrentHashMap<>();
  
  @Override
  public String getId() {
    return "quota-manager";
  }
  
  @Override
  public Uni<ProviderRequest> preProcess(ProviderRequest request) {
    String tenantId = request.getTenantContext().getTenantId();
    QuotaManager manager = quotas.computeIfAbsent(
      tenantId,
      t -> new QuotaManager()
    );
    
    if (manager.isExceeded()) {
      return Uni.createFrom().failure(
        new QuotaExceededException(tenantId)
      );
    }
    
    manager.consume(request.getEstimatedTokens());
    return Uni.createFrom().item(request);
  }
}
```

## Example: Content Safety Plugin

```java
public class ContentSafetyPlugin implements InferencePlugin {
  
  private ContentFilter filter = new OpenAIContentFilter();
  
  @Override
  public String getId() {
    return "content-safety";
  }
  
  @Override
  public Uni<ProviderRequest> preProcess(ProviderRequest request) {
    return filter.checkInput(request.getPrompt())
      .onItem().transform(safe -> {
        if (!safe) {
          throw new UnsafeContentException("Input violates policy");
        }
        return request;
      });
  }
}
```

## Plugin Lifecycle

```
LOADED → INITIALIZED → ACTIVE ↔ (PROCESS_REQUESTS) → UNLOADED
```

### Lifecycle Hooks

```java
@Override
public void initialize(Map<String, Object> config) {
  // Called when plugin is loaded
}

@Override
public void shutdown() {
  // Called when plugin is unloaded
}

@Override
public PluginInfo getInfo() {
  return PluginInfo.builder()
    .id("my-plugin")
    .version("1.0.0")
    .author("Your Team")
    .build();
}
```

## Plugin Extension Points

1. **preProcess** - Modify request before inference
2. **postProcess** - Transform response after inference
3. **onError** - Handle failures
4. **metrics** - Report custom metrics

## Plugin Configuration

```yaml
# application.properties
inference.plugins.enabled=true
inference.plugins.directory=/opt/plugins

# Per-plugin config
inference.plugin.my-model-router.enabled=true
inference.plugin.my-model-router.budget=100.00
```

## Testing Plugin

```java
@Test
void testQuotaEnforcement() {
  QuotaPlugin plugin = new QuotaPlugin();
  plugin.initialize(Map.of("quota_tokens", "1000"));
  
  ProviderRequest request = ProviderRequest.builder()
    .tenantId("tenant-1")
    .estimatedTokens(500)
    .build();
  
  Uni<ProviderRequest> result = plugin.preProcess(request);
  
  result.onItem().invoke(req -> {
    assert plugin.getRemainingQuota("tenant-1") == 500;
  }).subscribe();
}
```

## Best Practices

1. **Isolation** - Don't modify global state
2. **Async-First** - Use Uni/Mutiny for non-blocking
3. **Error Handling** - Fail gracefully, provide context
4. **Logging** - Include plugin ID in logs
5. **Performance** - Minimize latency added by plugin
6. **Testing** - Test with real provider instances

## Troubleshooting

### Plugin Not Loading
- Check META-INF/services entry
- Verify JAR is on classpath
- Check logs for initialization errors

### ClassNotFoundException
- Add missing dependencies to pom.xml
- Use provided scope for API deps

### Performance Issues
- Profile plugin code
- Implement caching if needed
- Consider async operations

## See Also

- [Run Inference](./run-inference.md)
- [Handle Multi-Tenancy](./handle-multi-tenancy.md)
- [Plugin Development Guide](../references/plugin-development.md)
