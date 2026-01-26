
Good.
We now **implement STEP 2 only** — nothing else.

> 🎯 **Step 2 goal**
> Implement the **entry point + request context**
> No pipeline logic yet. No plugins yet. No providers yet.

---

# STEP 2 — InferenceEngine & InferenceContext (FOUNDATION CODE)

---

## 1️⃣ `InferenceEngine` (Kernel Entry Point)

📍 `engine/InferenceEngine.java`

```java
package com.kayys.golek.inference.kernel.engine;

import com.kayys.golek.inference.kernel.model.InferenceRequest;
import com.kayys.golek.inference.kernel.model.InferenceResponse;

public interface InferenceEngine {

    InferenceResponse infer(InferenceRequest request);
}
```

**Rules**

* Single method
* Blocking (sync) only
* Streaming handled later

---

## 2️⃣ `EngineContext` (Kernel-Level Shared Services)

📍 `engine/EngineContext.java`

```java
package com.kayys.golek.inference.kernel.engine;

import com.kayys.golek.inference.kernel.pipeline.InferencePipeline;
import com.kayys.golek.inference.kernel.observer.InferenceObserver;
import com.kayys.golek.inference.kernel.plugin.PluginRegistry;

public interface EngineContext {

    InferencePipeline pipeline();

    PluginRegistry pluginRegistry();

    InferenceObserver observer();
}
```

🧠 **Important**

* This is **engine-scoped**
* Not request-scoped
* Created once at boot

---

## 3️⃣ `InferenceContext` (Request-Scoped State)

📍 `context/InferenceContext.java`

```java
package com.kayys.golek.inference.kernel.context;

import com.kayys.golek.inference.kernel.model.InferenceRequest;
import com.kayys.golek.inference.kernel.model.InferenceResponse;

import java.util.Map;

public interface InferenceContext {

    String requestId();

    InferenceRequest request();

    InferenceResponse response();

    void setResponse(InferenceResponse response);

    Map<String, Object> attributes();

    void fail(Throwable error);

    boolean isFailed();

    Throwable error();
}
```

### Why this design?

* Explicit lifecycle
* No hidden magic
* Kernel controls failure semantics

---

## 4️⃣ `DefaultInferenceContext` (Concrete Implementation)

📍 `context/DefaultInferenceContext.java`

```java
package com.kayys.golek.inference.kernel.context;

import com.kayys.golek.inference.kernel.model.InferenceRequest;
import com.kayys.golek.inference.kernel.model.InferenceResponse;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public final class DefaultInferenceContext
        implements InferenceContext {

    private final String requestId;
    private final InferenceRequest request;
    private final Map<String, Object> attributes = new ConcurrentHashMap<>();

    private volatile InferenceResponse response;
    private volatile Throwable error;

    public DefaultInferenceContext(InferenceRequest request) {
        this.requestId = UUID.randomUUID().toString();
        this.request = request;
    }

    @Override
    public String requestId() {
        return requestId;
    }

    @Override
    public InferenceRequest request() {
        return request;
    }

    @Override
    public InferenceResponse response() {
        return response;
    }

    @Override
    public void setResponse(InferenceResponse response) {
        this.response = response;
    }

    @Override
    public Map<String, Object> attributes() {
        return attributes;
    }

    @Override
    public void fail(Throwable error) {
        this.error = error;
    }

    @Override
    public boolean isFailed() {
        return error != null;
    }

    @Override
    public Throwable error() {
        return error;
    }
}
```

📌 **Key properties**

* Immutable request
* Mutable response
* Thread-safe attributes
* Explicit failure flag

---

## 5️⃣ `DefaultInferenceEngine` (Minimal Orchestrator)

📍 `engine/DefaultInferenceEngine.java`

```java
package com.kayys.golek.inference.kernel.engine;

import com.kayys.golek.inference.kernel.context.DefaultInferenceContext;
import com.kayys.golek.inference.kernel.context.InferenceContext;
import com.kayys.golek.inference.kernel.exception.InferenceException;
import com.kayys.golek.inference.kernel.model.InferenceRequest;
import com.kayys.golek.inference.kernel.model.InferenceResponse;

import java.util.Objects;

public final class DefaultInferenceEngine
        implements InferenceEngine {

    private final EngineContext engineContext;

    public DefaultInferenceEngine(EngineContext engineContext) {
        this.engineContext = Objects.requireNonNull(engineContext);
    }

    @Override
    public InferenceResponse infer(InferenceRequest request) {
        Objects.requireNonNull(request, "InferenceRequest must not be null");

        InferenceContext context = new DefaultInferenceContext(request);

        try {
            engineContext.observer().onStart(context);

            // ⚠️ pipeline not implemented yet (Step 3)
            engineContext.pipeline().execute(context);

            if (context.isFailed()) {
                throw new InferenceException(
                    "Inference failed",
                    context.error()
                );
            }

            engineContext.observer().onSuccess(context);
            return context.response();

        } catch (Throwable t) {
            engineContext.observer().onFailure(t, context);
            throw t instanceof InferenceException
                ? (InferenceException) t
                : new InferenceException("Kernel failure", t);
        }
    }
}
```

🚫 No plugin logic
🚫 No phase logic
🚫 No provider logic

This is **pure control flow only**.

---

## 6️⃣ Execution Flow (What Exists Now)

```
infer(request)
  ├── create InferenceContext
  ├── observer.onStart()
  ├── pipeline.execute(context)   ← stub (next step)
  ├── observer.onSuccess()
  └── return response
```

At this stage:

* Kernel compiles
* Engine is deterministic
* No hidden behavior

---

## 7️⃣ What We DO NOT Implement Yet

❌ InferencePipeline logic
❌ Phases
❌ Plugins
❌ Providers
❌ Streaming

Those start **STEP 3**.

---

## ✅ Step 2 Status

✔ Entry point frozen
✔ Context lifecycle correct
✔ Failure semantics explicit
✔ No hallucinated responsibilities

---

### Say **“step 3”**

and I will implement **InferencePipeline + InferencePhase execution**
with zero deviation from this kernel.


