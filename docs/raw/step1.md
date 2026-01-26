
Good. I’ll **go on systematically** and stay **strictly inside the inference kernel** we just froze.

No workflow.
No agent runtime.
No orchestration creep.

---

# STEP 1 — Kernel Module & Package Layout (FOUNDATION)

This step answers **“where does everything live?”**
Before any implementation, this must be rock-solid.

---

## 1️⃣ Kernel Module Boundary (Hard Line)

Create **one** module only:

```
golek-inference-kernel
```

Everything inside this module must obey:

* request-scoped execution
* no persistence
* no async background jobs
* no business semantics

---

## 2️⃣ Canonical Package Structure

```
golek-inference-kernel/
└── src/main/java/com/kayys/golek/inference/kernel/
    ├── engine/            # Entry point
    ├── context/           # Request + execution context
    ├── pipeline/          # Phase execution
    ├── phase/             # Phase definitions
    ├── plugin/            # Plugin SPI
    ├── provider/          # LLM provider SPI
    ├── streaming/         # Streaming abstractions
    ├── observer/          # Metrics / tracing hooks
    ├── exception/         # Kernel-level exceptions
    └── model/             # Pure inference DTOs
```

Nothing else.

If something doesn’t fit → it does NOT belong to kernel.

---

## 3️⃣ engine/ — Entry Point Only

```
engine/
├── InferenceEngine.java
├── DefaultInferenceEngine.java
└── EngineContext.java
```

### Responsibility

* Accept request
* Create context
* Invoke pipeline
* Return response

🚫 No logic
🚫 No policies
🚫 No provider code

---

## 4️⃣ context/ — Per-Request State

```
context/
├── InferenceContext.java
├── DefaultInferenceContext.java
├── RequestAttributes.java
```

### Rules

* Created per request
* Destroyed after response
* Thread-confined
* Never cached

---

## 5️⃣ phase/ — Deterministic Execution Order

```
phase/
└── InferencePhase.java
```

Only the enum.
No logic here.

---

## 6️⃣ pipeline/ — Phase Executor

```
pipeline/
├── InferencePipeline.java
└── DefaultInferencePipeline.java
```

### Responsibility

* Iterate phases
* Execute plugins
* Stop on failure

🚫 No provider knowledge
🚫 No policies

---

## 7️⃣ plugin/ — EXTENSION POINT (Most Important)

```
plugin/
├── Plugin.java
├── ConfigurablePlugin.java
├── InferencePhasePlugin.java
├── ModelValidationPlugin.java
└── PluginRegistry.java
```

### This is the **only** extension mechanism.

No “custom hooks”, no side channels.

---

## 8️⃣ provider/ — LLM Abstraction (Hard Boundary)

```
provider/
├── LLMProvider.java
├── StreamingLLMProvider.java
├── ProviderCapabilities.java
├── ProviderRequest.java
└── ProviderRegistry.java
```

### Rules

* Providers do inference ONLY
* Providers do NOT mutate context
* Providers do NOT apply policy

---

## 9️⃣ streaming/ — Optional but Clean

```
streaming/
├── StreamingResponse.java
├── StreamChunk.java
└── StreamSubscriber.java
```

Streaming is **data flow only**, not control flow.

---

## 🔟 observer/ — Observability Hooks

```
observer/
└── InferenceObserver.java
```

Observers are passive.

🚫 No mutation
🚫 No retries
🚫 No blocking

---

## 1️⃣1️⃣ model/ — Pure DTOs

```
model/
├── InferenceRequest.java
├── InferenceResponse.java
├── Message.java
└── ToolCall.java
```

No behavior.
No validation logic.

---

## 1️⃣2️⃣ exception/ — Kernel Errors Only

```
exception/
├── InferenceException.java
├── ValidationException.java
├── ProviderException.java
└── KernelPanicException.java
```

If an exception needs **business meaning**, it’s not kernel.

---

## 13️⃣ Dependency Direction (Non-Negotiable)

```
engine
 └── pipeline
      └── phase
           └── plugin
                └── provider
```

**Never the reverse.**

---

## 14️⃣ What Comes Next (Roadmap)

Now that the **structure is frozen**, next steps are:

### STEP 2

➡️ Implement **InferenceEngine + InferenceContext**

### STEP 3

➡️ Implement **InferencePipeline + phase execution**

### STEP 4

➡️ Implement **PluginRegistry + ordering + lifecycle**

### STEP 5

➡️ Implement **LLMProvider + cloud providers (OpenAI, Anthropic, etc.)**

### STEP 6

➡️ Streaming execution path

---

If you say **“go step 2”**, I will:

* implement `InferenceEngine`
* implement `DefaultInferenceContext`
* show exact call flow
* no abstractions skipped

I will not jump ahead again.
