---
name: monitor-inference
description: Monitor inference performance with OpenTelemetry tracing, Prometheus metrics, structured logging, and Kafka event streaming
metadata:
  short-description: Observe inference execution
  category: observability
  difficulty: beginner
---

# Monitor Inference Skill

Monitor and observe inference execution with distributed tracing, metrics, logging, and event streaming.

## When to Use

- You need to track inference latency
- You want to monitor error rates
- You need distributed tracing across services
- You want to stream inference events
- You need to debug inference issues

## Monitoring Stack

```
┌─────────────────────────────────────────┐
│   OpenTelemetry (Tracing & Metrics)     │
│                                         │
│  ├─ Jaeger/Tempo (trace storage)        │
│  ├─ Prometheus (metrics storage)        │
│  ├─ Loki (logs aggregation)             │
│  └─ Grafana (visualization)             │
│                                         │
│   Kafka (Event Streaming)               │
│   Structured Logging (JSON)             │
└─────────────────────────────────────────┘
```

## Steps

### 1. Enable OpenTelemetry Tracing

```yaml
# application.properties
quarkus.otel.enabled=true
quarkus.otel.exporter.otlp.endpoint=http://jaeger:4317
quarkus.otel.traces.exporter=otlp
quarkus.otel.logs.exporter=otlp
```

### 2. Enable Metrics Export

```yaml
quarkus.micrometer.enabled=true
quarkus.micrometer.export.prometheus.enabled=true

# Custom metrics
inference.metrics.enabled=true
inference.metrics.latency.buckets=10,50,100,500,1000
```

### 3. Trace Inference Requests

```java
@Traced  // Automatic tracing
public Uni<ProviderResponse> inferenceWithTracing(
    ProviderRequest request) {
  
  // Parent span created automatically
  Span span = Tracer.getCurrentSpan();
  span.setAttribute("tenant_id", request.getTenantId());
  span.setAttribute("model", request.getModelId());
  span.setAttribute("user_id", request.getUserId());
  
  return provider.infer(request)
    .onItem().invoke(response -> {
      span.addEvent("inference_completed", Attributes.of(
        AttributeKey.longKey("latency_ms"), 
          response.getLatencyMs()
      ));
    });
}
```

### 4. Track Custom Metrics

```java
// Counters
Counter inferenceCounter = meterRegistry.counter(
  "inference.requests",
  "model", "bert-base",
  "tenant", "tenant-1"
);
inferenceCounter.increment();

// Timers
Timer.Sample sample = Timer.start(meterRegistry);
try {
  result = provider.infer(request);
} finally {
  sample.stop(Timer.builder("inference.latency")
    .tag("model", request.getModelId())
    .tag("status", "success")
    .publishPercentiles(0.5, 0.95, 0.99)
    .register(meterRegistry)
  );
}

// Gauges
meterRegistry.gauge("inference.active_requests",
  activeRequests, 
  AtomicInteger::get
);
```

### 5. Structured Logging

```java
import org.jboss.logging.Logger;

Logger log = Logger.getLogger(MyService.class);

// Structured logging with context
log.infof("Inference request: model=%s, tenant=%s, user=%s",
  request.getModelId(),
  request.getTenantId(),
  request.getUserId()
);

// With additional context
log.infof("Inference completed: latency_ms=%d, tokens=%d, cost=%f",
  response.getLatencyMs(),
  response.getTokensUsed(),
  response.getCostUsd()
);

// Error logging
try {
  result = provider.infer(request);
} catch (Exception e) {
  log.errorf(e, "Inference failed: model=%s, error=%s",
    request.getModelId(),
    e.getMessage()
  );
}
```

### 6. Event Streaming to Kafka

```yaml
# application.properties
kafka.bootstrap.servers=localhost:9092
mp.messaging.outgoing.inference-events.connector=smallrye-kafka
mp.messaging.outgoing.inference-events.topic=inference-events
```

```java
@ApplicationScoped
public class InferenceEventPublisher {
  
  @Inject
  @Channel("inference-events")
  Emitter<InferenceEvent> emitter;
  
  public void publishEvent(ProviderRequest request,
                          ProviderResponse response) {
    InferenceEvent event = InferenceEvent.builder()
      .timestamp(Instant.now())
      .tenantId(request.getTenantId())
      .userId(request.getUserId())
      .modelId(request.getModelId())
      .latencyMs(response.getLatencyMs())
      .tokensUsed(response.getTokensUsed())
      .costUsd(response.getCostUsd())
      .status("success")
      .build();
    
    emitter.send(event);
  }
}
```

### 7. Query Metrics

```bash
# Get current latency p99
curl http://localhost:8080/q/metrics | \
  grep 'inference_latency_seconds_max'

# Get error rate
curl http://localhost:8080/q/metrics | \
  grep 'inference_errors_total'
```

## Key Metrics to Monitor

### Latency Metrics
```
inference.latency_ms
  - Percentiles: p50, p95, p99
  - Tags: model, provider, tenant
  - Alert threshold: p99 > 1000ms
```

### Throughput Metrics
```
inference.requests_total
  - Counter: total requests
  - Tags: status (success, error), model, tenant
  - Alert threshold: sudden drop
```

### Error Metrics
```
inference.errors_total
  - Counter: failed requests
  - Tags: error_type, model, tenant
  - Alert threshold: error rate > 5%
```

### Resource Metrics
```
inference.memory_usage_bytes
inference.gpu_memory_mb
inference.cpu_usage_percent
  - Tags: executor, tenant
  - Alert threshold: > 80% usage
```

### Cost Metrics
```
inference.cost_usd_total
  - Counter: accumulated cost
  - Tags: tenant, model, user
  - Track per tenant for billing
```

## Example: Full Monitoring Setup

```java
@ApplicationScoped
@Path("/api/infer")
public class MonitoredInferenceService {
  
  @Inject MeterRegistry meterRegistry;
  @Inject Logger log;
  @Inject InferenceEventPublisher publisher;
  @Inject Provider provider;
  
  @POST
  @Traced
  public Uni<ProviderResponse> infer(ProviderRequest request) {
    
    Timer.Sample sample = Timer.start(meterRegistry);
    Span span = Tracer.getCurrentSpan();
    
    // Add attributes
    span.setAttributes(Attributes.of(
      AttributeKey.stringKey("tenant_id"), 
        request.getTenantId(),
      AttributeKey.stringKey("model"), 
        request.getModelId(),
      AttributeKey.stringKey("user_id"), 
        request.getUserId()
    ));
    
    // Log request
    log.infof("Inference request: model=%s, tenant=%s",
      request.getModelId(),
      request.getTenantId()
    );
    
    // Execute
    return provider.infer(request)
      .onItem().invoke(response -> {
        // Record metrics
        sample.stop(meterRegistry.timer("inference.latency",
          "model", request.getModelId(),
          "status", "success"
        ));
        
        meterRegistry.counter("inference.tokens",
          "model", request.getModelId()
        ).increment(response.getTokensUsed());
        
        meterRegistry.counter("inference.cost_usd",
          "tenant", request.getTenantId()
        ).increment(response.getCostUsd());
        
        // Publish event
        publisher.publishEvent(request, response);
        
        // Log response
        log.infof("Inference completed: latency=%dms, tokens=%d",
          response.getLatencyMs(),
          response.getTokensUsed()
        );
      })
      .onFailure().invoke(ex -> {
        // Record error
        sample.stop(meterRegistry.timer("inference.latency",
          "model", request.getModelId(),
          "status", "error"
        ));
        
        meterRegistry.counter("inference.errors",
          "error_type", ex.getClass().getSimpleName()
        ).increment();
        
        // Log error
        log.errorf(ex, "Inference failed: model=%s",
          request.getModelId()
        );
      });
  }
}
```

## Grafana Dashboards

Create dashboards to visualize:

1. **Inference Latency** - p50, p95, p99 over time
2. **Error Rate** - Errors per minute by model
3. **Throughput** - Requests per second
4. **Cost** - Cumulative cost per tenant
5. **Resource Usage** - CPU, memory, GPU per executor
6. **Model Performance** - Accuracy, latency by model

## Alerting Rules

```yaml
groups:
  - name: inference
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.99, inference_latency_ms) > 1000
        for: 5m
      
      - alert: HighErrorRate
        expr: rate(inference_errors[5m]) > 0.05
        for: 2m
      
      - alert: HighMemoryUsage
        expr: inference_memory_bytes / 1e9 > 10
        for: 5m
```

## Best Practices

1. **Always Enable Tracing** - Understand request flow
2. **Monitor Key Metrics** - Latency, errors, cost
3. **Use Structured Logs** - Machine-readable format
4. **Stream Events** - Real-time processing
5. **Set Up Alerts** - Proactive monitoring
6. **Create Dashboards** - Visualize trends
7. **Archive Logs** - Long-term retention

## Troubleshooting

### Missing Traces
- Check Jaeger endpoint configuration
- Verify OTLP exporter is enabled
- Check network connectivity

### Metrics Not Appearing
- Verify Prometheus scrape interval
- Check metric names and tags
- Ensure metrics endpoint is accessible

### High Memory Usage
- Check for memory leaks
- Monitor model pool sizes
- Review long-running requests

## See Also

- [Run Inference](./run-inference.md)
- [Handle Multi-Tenancy](./handle-multi-tenancy.md)
- [Observability Setup](../references/observability.md)
