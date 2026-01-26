Alright — let’s **go on concretely**, not conceptually.

Below I’ll **continue by refactoring your engine into a truly plugin-driven execution model**, with **real code**, aligned with what you already have (Quarkus, SPI, warm pools, selection, tenants).

I’ll do this in **clear stages** so it’s easy to integrate.

---

# STEP 1 — Introduce the Plugin Kernel (Minimal, Non-Invasive)

We start by adding a **plugin kernel** that does **not break existing code**.

## 1.1 Core Plugin SPI (engine-level)

📦 `inference-core/plugin`

```java
public interface InferencePlugin {

    PluginDescriptor descriptor();

    default void initialize(PluginContext context) {}

    default void shutdown() {}
}
```

```java
public record PluginDescriptor(
    String id,
    String name,
    String version,
    PluginType type,
    Set<PluginCapability> capabilities,
    SemanticVersion minEngineVersion,
    SemanticVersion maxEngineVersion
) {}
```

```java
public enum PluginType {
    POLICY,
    EXECUTION_HOOK,
    VALIDATION,
    OBSERVABILITY,
    CONTROL_PLANE
}
```

```java
public enum PluginCapability {
    REQUEST_MUTATION,
    RATE_LIMIT,
    COST_CONTROL,
    TENANT_GUARD,
    FALLBACK_CONTROL,
    AUDIT,
    SHADOW_EXECUTION
}
```

---

## 1.2 Plugin Context (what plugins are allowed to see)

🔐 **This is critical for safety**

```java
public interface PluginContext {

    EngineInfo engineInfo();

    PluginConfiguration configuration();

    MetricsPublisher metrics();

    Clock clock();

    ExecutorService executor();
}
```

❗ Plugins **do NOT** get:

* ModelRunnerFactory
* Repository write access
* Security credentials
* CDI container

This prevents plugin abuse.

---

# STEP 2 — Execution Lifecycle Plugins (THE Core Upgrade)

Now we make inference **observable and controllable** without rewriting logic.

## 2.1 Execution Phases

```java
public enum InferencePhase {
    REQUEST_RECEIVED,
    MODEL_RESOLVED,
    RUNNER_SELECTED,
    PRE_INFERENCE,
    POST_INFERENCE,
    FAILURE,
    RESPONSE_READY
}
```

---

## 2.2 Hook Context (Safe & Immutable)

```java
public record InferenceHookContext(
    TenantContext tenantContext,
    String modelId,
    InferenceRequest request,
    RequestContext requestContext,
    ModelManifest manifest,
    RunnerCandidate selectedRunner,
    Map<String, Object> attributes
) {
    public InferenceHookContext withAttribute(String key, Object value) {
        Map<String, Object> copy = new HashMap<>(attributes);
        copy.put(key, value);
        return new InferenceHookContext(
            tenantContext, modelId, request, requestContext,
            manifest, selectedRunner, copy
        );
    }
}
```

---

## 2.3 Execution Hook Plugin

```java
public interface InferenceExecutionPlugin extends InferencePlugin {

    void onPhase(
        InferencePhase phase,
        InferenceHookContext context
    );
}
```

---

# STEP 3 — Plugin Registry (Runtime Brain)

📦 `inference-core/plugin`

```java
@ApplicationScoped
public class PluginRegistry {

    @Inject
    Instance<InferencePlugin> plugins;

    private final Map<PluginType, List<InferencePlugin>> byType =
        new EnumMap<>(PluginType.class);

    @PostConstruct
    void init() {
        for (InferencePlugin plugin : plugins) {
            PluginDescriptor d = plugin.descriptor();
            byType.computeIfAbsent(d.type(), k -> new ArrayList<>())
                  .add(plugin);
        }
    }

    @SuppressWarnings("unchecked")
    public <T extends InferencePlugin> List<T> getByType(
        PluginType type, Class<T> clazz
    ) {
        return byType.getOrDefault(type, List.of())
                     .stream()
                     .map(clazz::cast)
                     .toList();
    }

    public List<InferenceExecutionPlugin> executionPlugins() {
        return getByType(
            PluginType.EXECUTION_HOOK,
            InferenceExecutionPlugin.class
        );
    }
}
```

---

# STEP 4 — Refactor InferenceOrchestrator (IMPORTANT)

We **do not rewrite logic**, we **wrap it with hooks**.

## 4.1 Inject PluginRegistry

```java
@Inject
PluginRegistry pluginRegistry;
```

---

## 4.2 Emit Phases (Minimal Changes)

### Before execution

```java
InferenceHookContext hookCtx =
    new InferenceHookContext(
        tenantContext,
        modelId,
        request,
        ctx,
        manifest,
        null,
        Map.of()
    );

emit(InferencePhase.REQUEST_RECEIVED, hookCtx);
```

---

### After runner selection

```java
RunnerCandidate candidate = candidates.get(0);

hookCtx = hookCtx.withAttribute("candidateScore", candidate.score());

emit(InferencePhase.RUNNER_SELECTED, 
    new InferenceHookContext(
        tenantContext,
        modelId,
        request,
        ctx,
        manifest,
        candidate,
        hookCtx.attributes()
    )
);
```

---

### Before inference

```java
emit(InferencePhase.PRE_INFERENCE, hookCtx);
```

---

### After success

```java
emit(InferencePhase.POST_INFERENCE, hookCtx);
emit(InferencePhase.RESPONSE_READY, hookCtx);
```

---

### On failure

```java
emit(InferencePhase.FAILURE, hookCtx);
```

---

### Emit helper

```java
private void emit(
    InferencePhase phase,
    InferenceHookContext ctx
) {
    for (InferenceExecutionPlugin plugin :
            pluginRegistry.executionPlugins()) {
        try {
            plugin.onPhase(phase, ctx);
        } catch (Exception e) {
            Log.warnf(
                e,
                "Plugin %s failed at phase %s",
                plugin.descriptor().id(),
                phase
            );
        }
    }
}
```

⚠️ **Plugins can fail without breaking inference**

---

# STEP 5 — Policy Plugins (Move Logic OUT of Orchestrator)

## 5.1 Policy SPI

```java
public interface InferencePolicyPlugin extends InferencePlugin {

    PolicyDecision evaluate(InferencePolicyContext context);
}
```

```java
public record InferencePolicyContext(
    TenantContext tenantContext,
    String modelId,
    InferenceRequest request,
    ModelManifest manifest
) {}
```

---

## 5.2 Policy Enforcement Point

Inside `execute()`:

```java
for (InferencePolicyPlugin policy :
        pluginRegistry.getByType(
            PluginType.POLICY,
            InferencePolicyPlugin.class
        )) {

    PolicyDecision decision = policy.evaluate(
        new InferencePolicyContext(
            tenantContext, modelId, request, manifest
        )
    );

    if (decision == PolicyDecision.DENY) {
        throw new PolicyDeniedException(
            policy.descriptor().id()
        );
    }
}
```

---

# STEP 6 — Example Plugins (So This Is Real)

## 6.1 Tenant Quota Plugin

```java
@ApplicationScoped
public class TenantQuotaPlugin
    implements InferencePolicyPlugin {

    @Override
    public PluginDescriptor descriptor() {
        return new PluginDescriptor(
            "tenant-quota",
            "Tenant Quota Guard",
            "1.0.0",
            PluginType.POLICY,
            Set.of(PluginCapability.RATE_LIMIT),
            SemanticVersion.of("2.0.0"),
            SemanticVersion.of("3.0.0")
        );
    }

    @Override
    public PolicyDecision evaluate(
        InferencePolicyContext ctx
    ) {
        if (quotaExceeded(ctx.tenantContext())) {
            return PolicyDecision.DENY;
        }
        return PolicyDecision.ALLOW;
    }
}
```

---

## 6.2 Shadow Execution Plugin

```java
@ApplicationScoped
public class ShadowInferencePlugin
    implements InferenceExecutionPlugin {

    @Override
    public PluginDescriptor descriptor() {
        return new PluginDescriptor(
            "shadow-exec",
            "Shadow Execution",
            "1.0.0",
            PluginType.EXECUTION_HOOK,
            Set.of(PluginCapability.SHADOW_EXECUTION),
            SemanticVersion.of("2.0.0"),
            SemanticVersion.of("3.0.0")
        );
    }

    @Override
    public void onPhase(
        InferencePhase phase,
        InferenceHookContext ctx
    ) {
        if (phase == InferencePhase.POST_INFERENCE) {
            // async shadow call
            CompletableFuture.runAsync(() -> {
                // execute alternative runner for comparison
            });
        }
    }
}
```

---

# STEP 7 — Why This Makes You Future-Proof

You now have:

✅ **Zero adapter changes**
✅ **Behavior externalized**
✅ **Safe extension surface**
✅ **Tenant-aware policies**
✅ **Dynamic rollout ready**
✅ **Agentic / golek compatible**

This architecture can absorb:

* Agent-based routing
* Negotiation plugins
* Cost arbitrage
* Compliance enforcement
* Autonomous optimization
* Control-plane orchestration

---

## NEXT OPTIONS (Pick One)

1️⃣ **Dynamic plugin loading (external JARs)**
2️⃣ **golek / ACP integration using this plugin system**
3️⃣ **Control-plane + config compiler design**
4️⃣ **Executor-level plugins (inside ModelRunner)**
5️⃣ **Formal Plugin SDK for third parties**

Tell me which one you want next, and I’ll go just as deep.
