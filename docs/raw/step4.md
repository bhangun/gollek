
Good.
**STEP 4 is the heart of the kernel.**
I’ll be extremely strict and systematic here.

> 🎯 **Step 4 goal**
> Introduce a **single, consistent, non-hallucinated plugin system**
> that binds plugins to **InferencePhase** and nothing else.

No workflow.
No provider logic.
No magic hooks.

---

# STEP 4 — Plugin System (FINAL, CONSISTENT)

---

## 0️⃣ Plugin Model (LOCKED)

There are **exactly 4 plugin interfaces** — no more.

```
Plugin
 └── ConfigurablePlugin
       └── InferencePhasePlugin
             └── ModelValidationPlugin
```

Everything else is composition.

---

## 1️⃣ `Plugin` (Root Interface)

📍 `plugin/Plugin.java`

```java
package com.kayys.golek.inference.kernel.plugin;

public interface Plugin {

    /**
     * Globally unique plugin id.
     */
    String id();

    /**
     * Ordering within the same phase.
     * Lower value = earlier execution.
     */
    int order();

    /**
     * Engine lifecycle hook.
     */
    default void initialize(PluginContext context) {}

    /**
     * Engine lifecycle hook.
     */
    default void shutdown() {}
}
```

🔒 No phase
🔒 No request context
🔒 Engine lifecycle only

---

## 2️⃣ `ConfigurablePlugin`

📍 `plugin/ConfigurablePlugin.java`

```java
package com.kayys.golek.inference.kernel.plugin;

import java.util.Map;

public interface ConfigurablePlugin extends Plugin {

    /**
     * Configure once at engine startup.
     */
    default void configure(Map<String, Object> config) {}
}
```

📌 No runtime mutation
📌 Immutable config per plugin instance

---

## 3️⃣ `InferencePhasePlugin` (Execution Extension)

📍 `plugin/InferencePhasePlugin.java`

```java
package com.kayys.golek.inference.kernel.plugin;

import com.kayys.golek.inference.kernel.context.InferenceContext;
import com.kayys.golek.inference.kernel.engine.EngineContext;
import com.kayys.golek.inference.kernel.phase.InferencePhase;

public interface InferencePhasePlugin
        extends ConfigurablePlugin {

    /**
     * Phase this plugin belongs to.
     */
    InferencePhase phase();

    /**
     * Execute during its phase.
     */
    void execute(
        InferenceContext context,
        EngineContext engine
    );
}
```

📌 Deterministic
📌 Phase-bound
📌 No async
📌 No provider calls

---

## 4️⃣ `ModelValidationPlugin` (Typed Specialization)

📍 `plugin/ModelValidationPlugin.java`

```java
package com.kayys.golek.inference.kernel.plugin;

import com.kayys.golek.inference.kernel.phase.InferencePhase;

/**
 * Marker interface for validation plugins.
 */
public interface ModelValidationPlugin
        extends InferencePhasePlugin {

    @Override
    default InferencePhase phase() {
        return InferencePhase.VALIDATION;
    }
}
```

🧠 This exists **only for clarity & grouping**
Not mandatory, but future-proof.

---

## 5️⃣ `PluginContext` (Engine-Scoped)

📍 `plugin/PluginContext.java`

```java
package com.kayys.golek.inference.kernel.plugin;

import com.kayys.golek.inference.kernel.engine.EngineContext;

public interface PluginContext {

    EngineContext engine();
}
```

No request access.
No provider access.

---

## 6️⃣ `PluginRegistry` (Single Source of Truth)

📍 `plugin/PluginRegistry.java`

```java
package com.kayys.golek.inference.kernel.plugin;

import com.kayys.golek.inference.kernel.phase.InferencePhase;

import java.util.List;

public interface PluginRegistry {

    List<InferencePhasePlugin> pluginsFor(InferencePhase phase);
}
```

No dynamic lookups.
No mutation.

---

## 7️⃣ `DefaultPluginRegistry` (STRICT IMPLEMENTATION)

📍 `plugin/DefaultPluginRegistry.java`

```java
package com.kayys.golek.inference.kernel.plugin;

import com.kayys.golek.inference.kernel.phase.InferencePhase;

import java.util.*;
import java.util.stream.Collectors;

public final class DefaultPluginRegistry
        implements PluginRegistry {

    private final Map<InferencePhase, List<InferencePhasePlugin>> plugins;

    public DefaultPluginRegistry(
        Collection<InferencePhasePlugin> discoveredPlugins,
        PluginContext context
    ) {
        Objects.requireNonNull(discoveredPlugins);
        Objects.requireNonNull(context);

        Map<InferencePhase, List<InferencePhasePlugin>> map =
            new EnumMap<>(InferencePhase.class);

        for (InferencePhase phase : InferencePhase.values()) {
            map.put(phase, new ArrayList<>());
        }

        for (InferencePhasePlugin plugin : discoveredPlugins) {
            plugin.initialize(context);
            map.get(plugin.phase()).add(plugin);
        }

        // Sort by order
        map.replaceAll((phase, list) ->
            list.stream()
                .sorted(Comparator.comparingInt(Plugin::order))
                .collect(Collectors.toUnmodifiableList())
        );

        this.plugins = Map.copyOf(map);
    }

    @Override
    public List<InferencePhasePlugin> pluginsFor(
        InferencePhase phase
    ) {
        return plugins.getOrDefault(phase, List.of());
    }
}
```

🔒 Immutable
🔒 Ordered
🔒 Deterministic

---

## 8️⃣ Wire Plugins Into Pipeline (FINAL STEP)

📍 `pipeline/DefaultInferencePipeline.java` (UPDATED)

```java
package com.kayys.golek.inference.kernel.pipeline;

import com.kayys.golek.inference.kernel.context.InferenceContext;
import com.kayys.golek.inference.kernel.engine.EngineContext;
import com.kayys.golek.inference.kernel.observer.InferenceObserver;
import com.kayys.golek.inference.kernel.phase.InferencePhase;
import com.kayys.golek.inference.kernel.plugin.InferencePhasePlugin;

import java.util.Objects;

public final class DefaultInferencePipeline
        implements InferencePipeline {

    private final EngineContext engine;
    private final InferenceObserver observer;

    public DefaultInferencePipeline(
        EngineContext engine,
        InferenceObserver observer
    ) {
        this.engine = Objects.requireNonNull(engine);
        this.observer = Objects.requireNonNull(observer);
    }

    @Override
    public void execute(InferenceContext context) {

        for (InferencePhase phase : InferencePhase.ordered()) {

            if (context.isFailed()) {
                return;
            }

            observer.onPhase(phase, context);

            for (InferencePhasePlugin plugin
                    : engine.pluginRegistry().pluginsFor(phase)) {

                try {
                    plugin.execute(context, engine);
                } catch (Throwable t) {
                    context.fail(t);
                    return;
                }
            }
        }
    }
}
```

---

## 9️⃣ Execution Flow (Now REAL)

```
infer()
 ├── context created
 ├── VALIDATION
 │    ├── plugin A
 │    └── plugin B
 ├── PRE_PROCESSING
 ├── PROVIDER_DISPATCH
 └── POST_PROCESSING
```

Linear. Predictable. Safe.

---

## 10️⃣ What Is STILL NOT IMPLEMENTED

❌ LLMProvider
❌ ProviderRegistry
❌ Streaming
❌ Actual inference

That is **STEP 5**.

---

## ✅ Step 4 Status

✔ Single plugin system
✔ No inconsistencies
✔ No mixed abstractions
✔ Phase-safe
✔ Enterprise-grade

---

Say **“step 5”**
and we implement **LLMProvider + cloud providers (OpenAI, Anthropic, etc.)**
**without touching plugins or pipeline again.**
