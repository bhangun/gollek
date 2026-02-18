---
name: handle-multi-tenancy
description: Implement tenant isolation with resource quotas, separate model pools, and secure credential management
metadata:
  short-description: Manage multiple isolated tenants
  category: multi-tenancy
  difficulty: intermediate
---

# Handle Multi-Tenancy Skill

Configure and manage multiple isolated tenants in inference-gollek with resource quotas, separate model pools, and audit trails.

## When to Use

- You're building a multi-tenant inference service
- You need per-tenant resource quotas
- You want isolated model pools per tenant
- You need tenant-specific audit logs

## Multi-Tenancy Architecture

```
┌─────────────────────────────────────┐
│     inference-gollek Platform        │
├─────────────────────────────────────┤
│  Tenant 1    │  Tenant 2  │ Tenant 3│
│ (isolated)   │ (isolated) │(isolated)
│ - Models     │ - Models   │- Models │
│ - Quotas     │ - Quotas   │- Quotas │
│ - Audit      │ - Audit    │- Audit  │
└─────────────────────────────────────┘
```

## Steps

### 1. Configure Tenants

```yaml
# application.properties
inference.multi-tenancy.enabled=true
inference.multi-tenancy.isolation-level=full

# Tenant configurations
inference.tenant.tenant-1.enabled=true
inference.tenant.tenant-1.model-quota=50
inference.tenant.tenant-1.daily-quota=10000

inference.tenant.tenant-2.enabled=true
inference.tenant.tenant-2.model-quota=100
inference.tenant.tenant-2.daily-quota=50000
```

### 2. Create Tenant Context

```java
RequestContext context = RequestContext.builder()
  .requestId("tenant-1")
  .userId("user-456")
  .metadata(Map.of(
    "region", "us-east-1",
    "subscription_tier", "premium"
  ))
  .build();
```

### 3. Include in Every Request

```java
ProviderRequest request = ProviderRequest.builder()
  .requestContext(context)
  .requestId("tenant-1")  // Also for convenience
  .modelId("bert-base")
  .prompt("Analyze this: ...")
  .build();

Uni<ProviderResponse> result = provider.infer(request);
```

### 4. Access Tenant-Scoped Resources

```java
// Models are isolated per tenant
ModelRepository repo = modelRepositoryFactory
  .forTenant("tenant-1");

// Quotas are tracked per tenant
QuotaManager quota = quotaManager
  .forTenant("tenant-1");

long remaining = quota.getRemainingTokens();
```

## Tenant Resource Isolation

### Model Pool Isolation

```java
// Each tenant has separate warm model pool
WarmModelPool pool1 = modelPoolManager.getPool("tenant-1");
WarmModelPool pool2 = modelPoolManager.getPool("tenant-2");

// Models loaded for tenant-1 don't affect tenant-2
pool1.preload("bert-base");     // Tenant-1 only
pool2.preload("gpt2");          // Tenant-2 only
```

### Quota Management

```java
// Per-tenant quotas
QuotaPolicy policy = QuotaPolicy.builder()
  .dailyTokenLimit(100000)
  .monthlyTokenLimit(3000000)
  .concurrentRequests(10)
  .tokensPerMinute(1000)
  .build();

quotaManager.setPolicy("tenant-1", policy);
```

### Credential Isolation

```java
// Each tenant has separate credentials
SecretManager secrets = secretManager.forTenant("tenant-1");

String apiKey = secrets.getSecret("openai-key");
String hubToken = secrets.getSecret("huggingface-token");

// Credentials cannot be accessed across tenants
// Attempting to access tenant-2 secret as tenant-1 fails
```

## Audit Trail Per Tenant

```java
// Every operation is logged per tenant
AuditLog log = auditLogger.getLog("tenant-1");

log.getAllEvents()
  .filter(event -> event.getTimestamp() > cutoffTime)
  .onItem().invoke(event -> {
    System.out.println(event.getAction());      // "INFERENCE_EXECUTED"
    System.out.println(event.getUser());        // "user-456"
    System.out.println(event.getModel());       // "bert-base"
    System.out.println(event.getTokensUsed());  // 145
    System.out.println(event.getCost());        // $0.0015
  });
```

## Tenant Provisioning

```java
// Add new tenant at runtime
TenantManager manager = tenantManager.getInstance();

Uni<Void> provision = manager.createTenant(
  "tenant-3",
  TenantConfig.builder()
    .name("Customer Corp")
    .subscription_tier("enterprise")
    .model_quota(500)
    .daily_quota(100000)
    .contact_email("admin@customer.com")
    .build()
);

provision.onItem().invoke(() -> {
  System.out.println("Tenant created");
});
```

## Access Control Per Tenant

```java
// Tenant admins can only access their tenant data
if (!request.getRequestContext().canAccess("tenant-2")) {
  throw new UnauthorizedException(
    "Cannot access another tenant's data"
  );
}

// Audit trail is also tenant-isolated
AuditLog log = auditLogger.getLog(
  request.getRequestContext().getRequestId()
);
```

## Cost Tracking Per Tenant

```java
// Track costs per tenant
CostTracker tracker = costTracker.forTenant("tenant-1");

Uni<Double> cost = provider.infer(request)
  .onItem().invoke(response -> {
    double requestCost = response.getTokensUsed() * pricePerToken;
    tracker.addCost(requestCost);
  })
  .map(response -> response.getOutput());

// Get tenant billing summary
tracker.getBillingPeriod(Period.ofMonths(1))
  .onItem().invoke(billing -> {
    System.out.println("Total Cost: $" + billing.getTotalCost());
    System.out.println("Tokens Used: " + billing.getTotalTokens());
    System.out.println("API Calls: " + billing.getApiCalls());
  });
```

## Tenant Metrics Isolation

```java
// Metrics are isolated per tenant
MetricsRegistry metrics = metricsRegistry.forTenant("tenant-1");

metrics.getMetric("inference.latency_p99")   // Tenant-1 only
metrics.getMetric("inference.error_rate")    // Tenant-1 only
metrics.getMetric("model.cache_hits")        // Tenant-1 only
```

## Example: Multi-Tenant Inference Pipeline

```java
public Uni<InferenceResponse> multiTenantInference(
    String requestId,
    String userId,
    String modelId,
    String prompt) {
  
  // Create tenant context
  RequestContext context = RequestContext.builder()
    .requestId(requestId)
    .userId(userId)
    .build();
  
  // Create request with tenant context
  ProviderRequest request = ProviderRequest.builder()
    .requestContext(context)
    .modelId(modelId)
    .prompt(prompt)
    .build();
  
  // Check quota
  return quotaManager.forTenant(requestId)
    .checkAvailable(request.getEstimatedTokens())
    .chain(() -> provider.infer(request))
    .onItem().invoke(response -> {
      // Log to tenant's audit trail
      auditLogger.forTenant(requestId).log(
        AuditEvent.builder()
          .action("INFERENCE_EXECUTED")
          .userId(userId)
          .model(modelId)
          .tokensUsed(response.getTokensUsed())
          .build()
      );
      
      // Track costs
      costTracker.forTenant(requestId)
        .addCost(response.getTokensUsed() * PRICE_PER_TOKEN);
    });
}
```

## Tenant Validation

```java
// Always validate tenant context in requests
TenantValidator validator = new TenantValidator();

validator.validate(request.getRequestContext())
  .onFailure().invoke(ex -> {
    log.error("Invalid tenant: " + ex.getMessage());
  });
```

## Best Practices

1. **Always Include Tenant ID** - Every request must have tenant context
2. **Validate Tenant Access** - Verify users can access requested tenants
3. **Isolate Credentials** - Never share credentials across tenants
4. **Monitor Quotas** - Alert when approaching limits
5. **Audit Everything** - Log all tenant operations
6. **Isolate Models** - Keep model pools separate
7. **Cost Transparency** - Track and report per-tenant costs

## Troubleshooting

### Quota Exceeded
```
Error: Daily quota exceeded for tenant-1
Fix: Check quota limits or request increase
```

### Cross-Tenant Data Leak
```
Mitigation: Always validate tenant context
Never access tenant-2 resources as tenant-1
```

### Model Cache Conflicts
```
Solution: Models cached per tenant
No cross-tenant cache sharing
```

## See Also

- [Run Inference](./run-inference.md)
- [Configure Plugin](./configure-plugin.md)
- [Monitor Inference](./monitor-inference.md)
- [Multi-Tenancy Architecture](../references/multi-tenancy.md)
