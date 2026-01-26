
Perfect.
We now do **STEP 3 only**.

> 🎯 **Step 3 goal**
> Implement **InferencePipeline** and **phase execution**
> ❌ still no plugins logic
> ❌ no providers
> ❌ no business behavior

We wire **structure only**.

---

# STEP 3 — InferencePipeline & Phase Execution

---

## 1️⃣ `InferencePhase` (Execution Order)

📍 `phase/InferencePhase.java`

```java
package com.kayys.golek.inference.kernel.phase;

/**
 * Deterministic inference lifecycle.
 * Order MUST NOT change at runtime.
 */
public enum InferencePhase {

    VALIDATION,
    PRE_PROCESSING,
    PROVIDER_DISPATCH,
    POST_PROCESSING;

    public static InferencePhase[] ordered() {
        return values();
    }
}
```

🔒 Frozen order
🔒 No extension here
🔒 Kernel-owned

---

## 2️⃣ `InferencePipeline` Interface

📍 `pipeline/InferencePipeline.java`

```java
package com.kayys.golek.inference.kernel.pipeline;

import com.kayys.golek.inference.kernel.context.InferenceContext;

public interface InferencePipeline {

    void execute(InferenceContext context);
}
```

Minimal. No generics. No async.

---

## 3️⃣ `DefaultInferencePipeline` (Phase Loop)

📍 `pipeline/DefaultInferencePipeline.java`

```java
package com.kayys.golek.inference.kernel.pipeline;

import com.kayys.golek.inference.kernel.context.InferenceContext;
import com.kayys.golek.inference.kernel.phase.InferencePhase;
import com.kayys.golek.inference.kernel.observer.InferenceObserver;

import java.util.Objects;

public final class DefaultInferencePipeline
        implements InferencePipeline {

    private final InferenceObserver observer;

    public DefaultInferencePipeline(InferenceObserver observer) {
        this.observer = Objects.requireNonNull(observer);
    }

    @Override
    public void execute(InferenceContext context) {

        for (InferencePhase phase : InferencePhase.ordered()) {

            if (context.isFailed()) {
                return;
            }

            observer.onPhase(phase, context);

            // ⚠️ Plugins will execute here (STEP 4)
        }
    }
}
```

📌 Key points:

* Linear execution
* Early exit on failure
* Observer notified per phase
* **No plugins yet**

---

## 4️⃣ Observer Extension (Phase Awareness)

📍 `observer/InferenceObserver.java` (UPDATED)

```java
package com.kayys.golek.inference.kernel.observer;

import com.kayys.golek.inference.kernel.context.InferenceContext;
import com.kayys.golek.inference.kernel.phase.InferencePhase;

public interface InferenceObserver {

    default void onStart(InferenceContext context) {}

    default void onPhase(
        InferencePhase phase,
        InferenceContext context
    ) {}

    default void onSuccess(InferenceContext context) {}

    default void onFailure(
        Throwable error,
        InferenceContext context
    ) {}
}
```

No implementation yet.

---

## 5️⃣ Engine ↔ Pipeline Wiring

Update `EngineContext` usage:

📍 `engine/EngineContext.java` (unchanged, just reaffirmed)

```java
InferencePipeline pipeline();
```

📍 `engine/DefaultInferenceEngine.java`

Already calls:

```java
engineContext.pipeline().execute(context);
```

So pipeline is now real.

---

## 6️⃣ Execution Flow (Current State)

```
infer()
  ├── create context
  ├── observer.onStart()
  ├── pipeline.execute()
        ├── VALIDATION
        ├── PRE_PROCESSING
        ├── PROVIDER_DISPATCH
        └── POST_PROCESSING
  ├── observer.onSuccess()
  └── return response
```

⚠️ Phases are **empty by design** at this step.

---

## 7️⃣ Why This Matters

You now have:

✔ deterministic phase model
✔ predictable execution
✔ correct failure short-circuit
✔ observer visibility

Without introducing **any behavior** prematurely.

---

## 8️⃣ What Comes Next (STEP 4)

Now we are ready for **the most critical part**:

### STEP 4

➡️ Plugin system:

* `Plugin`
* `ConfigurablePlugin`
* `InferencePhasePlugin`
* `PluginRegistry`
* Ordered execution per phase

This is where extensibility happens.

---

Say **“step 4”** when ready.
