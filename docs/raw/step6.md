Perfect.
**Step 6 is where we lock extensibility without breaking kernel purity.**
Still **NO workflow, NO orchestration, NO hallucination drift**.

---

# 🔹 STEP 6 — PluginRegistry & Plugin Loading Model

This step defines **how plugins are discovered, validated, ordered, and exposed** to the kernel.

---

## 1️⃣ What PluginRegistry IS (and IS NOT)

### ✅ IS

* Central **catalog of loaded plugins**
* Groups plugins by **capability & phase**
* Enforces **deterministic ordering**
* Enforces **type safety**

### ❌ IS NOT

* Execution engine
* Configuration system
* Dependency injector
* Workflow coordinator

---

## 2️⃣ Plugin Taxonomy (final & consistent)

We **explicitly limit plugin types** to avoid chaos.

```java
public interface Plugin {
    String name();
    PluginType type();
}
```

### PluginType (canonical)

```java
public enum PluginType {

    MODEL_VALIDATION,
    INFERENCE_PHASE,
    PROVIDER_ADAPTER,

    CONFIGURATION,
    OBSERVABILITY,
    SECURITY
}
```

🔒 **No generic “do anything” plugin**

---

## 3️⃣ Specialized Plugin Interfaces

### InferencePhasePlugin (already defined in Step 5)

```java
public interface InferencePhasePlugin extends Plugin {
    InferencePhase phase();
    int order();
    boolean supports(InferenceContext context);
    InferenceContext execute(InferenceContext context);
}
```

---

### ModelValidationPlugin

```java
public interface ModelValidationPlugin extends Plugin {

    @Override
    default PluginType type() {
        return PluginType.MODEL_VALIDATION;
    }

    void validate(ModelDescriptor model)
        throws ModelValidationException;
}
```

Used **before inference pipeline begins**.

---

### ProviderAdapterPlugin (LLM Cloud / Local)

```java
public interface ProviderAdapterPlugin extends Plugin {

    @Override
    default PluginType type() {
        return PluginType.PROVIDER_ADAPTER;
    }

    String providerId();

    InferenceResponse infer(
        ModelDescriptor model,
        Prompt prompt
    );
}
```

🔒 This is the **ONLY boundary to LLM vendors**

---

## 4️⃣ PluginRegistry Interface (authoritative)

```java
public interface PluginRegistry {

    List<ModelValidationPlugin> modelValidators();

    Map<InferencePhase, List<InferencePhasePlugin>>
        inferencePhasePlugins();

    ProviderAdapterPlugin provider(String providerId);

    <T extends Plugin> List<T> plugins(Class<T> type);
}
```

---

## 5️⃣ DefaultPluginRegistry (kernel-owned)

```java
public final class DefaultPluginRegistry
        implements PluginRegistry {

    private final List<ModelValidationPlugin> modelValidators;
    private final Map<InferencePhase, List<InferencePhasePlugin>> phasePlugins;
    private final Map<String, ProviderAdapterPlugin> providers;

    public DefaultPluginRegistry(
        Collection<Plugin> plugins
    ) {
        this.modelValidators = collectModelValidators(plugins);
        this.phasePlugins = collectPhasePlugins(plugins);
        this.providers = collectProviders(plugins);
    }

    @Override
    public List<ModelValidationPlugin> modelValidators() {
        return modelValidators;
    }

    @Override
    public Map<InferencePhase, List<InferencePhasePlugin>>
    inferencePhasePlugins() {
        return phasePlugins;
    }

    @Override
    public ProviderAdapterPlugin provider(String providerId) {
        return providers.get(providerId);
    }
}
```

---

## 6️⃣ Deterministic Ordering Rule (VERY IMPORTANT)

```java
private Map<InferencePhase, List<InferencePhasePlugin>>
collectPhasePlugins(Collection<Plugin> plugins) {

    return plugins.stream()
        .filter(p -> p instanceof InferencePhasePlugin)
        .map(p -> (InferencePhasePlugin) p)
        .collect(Collectors.groupingBy(
            InferencePhasePlugin::phase,
            Collectors.collectingAndThen(
                Collectors.toList(),
                list -> list.stream()
                    .sorted(Comparator.comparingInt(
                        InferencePhasePlugin::order))
                    .toList()
            )
        ));
}
```

🔒 **Same input → same output → same execution**

---

## 7️⃣ Plugin Loading Models (supported)

### 🔹 Model A — Static (compile-time)

```java
List<Plugin> plugins = List.of(
    new OpenAIGolekPlugin(),
    new DefaultPromptBuilderPlugin(),
    new SafetyFilterPlugin()
);

PluginRegistry registry =
    new DefaultPluginRegistry(plugins);
```

✔ Fast
✔ Embedded
❌ No runtime flexibility

---

### 🔹 Model B — Java SPI (recommended default)

```java
ServiceLoader<Plugin> loader =
    ServiceLoader.load(Plugin.class);

PluginRegistry registry =
    new DefaultPluginRegistry(
        loader.stream()
            .map(ServiceLoader.Provider::get)
            .toList()
    );
```

✔ Modular
✔ Enterprise-friendly
✔ No reflection hacks

---

### 🔹 Model C — Dynamic / Hot-loaded (advanced)

* Isolated ClassLoader
* Signature verification
* Version compatibility checks

(We will design this **later**, not now.)

---

## 8️⃣ Why PluginRegistry is SAFE

| Risk                | Mitigation            |
| ------------------- | --------------------- |
| Plugin chaos        | Strict interfaces     |
| Non-determinism     | Ordered phases        |
| Vendor lock-in      | ProviderAdapterPlugin |
| Runtime instability | Pre-validation        |

---

## 9️⃣ Kernel Usage (clean & minimal)

```java
PluginRegistry registry = ...;

PhasePipeline pipeline =
    new PhasePipeline(registry);

pipeline.execute(context);
```

Kernel **never inspects plugin internals**.

---

# ✅ Step 6 COMPLETE

You now have:

* **Formal plugin taxonomy**
* **Deterministic loading**
* **Cloud / local provider isolation**
* **Enterprise-safe extension model**

---

## ⏭️ Step 7 (next)

**LLM Provider Abstraction**

* Cloud vs Local
* Streaming vs Sync
* Token accounting
* Timeouts & retries

Say **“step 7”** and I will continue with the same precision.
