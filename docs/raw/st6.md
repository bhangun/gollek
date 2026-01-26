You’re right to call this out — and thank you for stopping it **now**.
Let’s reset cleanly and **remove the hallucination vector**.

I mixed **workflow-runtime vocabulary** into what is supposed to be an **LLM inference engine core**.
That is **not acceptable** for what you’re building.

Below is a **hard correction**.

---

## 🔴 What Went Wrong (Explicit Admission)

I incorrectly imported **workflow / BPMN mental models**:

* `WorkflowEngine`
* `NodeDescriptor`
* `ExecutionToken` (as a workflow cursor)
* Human approval, compensation, retry semantics

Those belong to:

* BPMN engines
* ESB runtimes
* Long-running business processes

❌ **NOT** to an **Inference Server / LLM Runtime Kernel**

You are correct.

---

## ✅ What We Are ACTUALLY Building

Let’s restate the **true scope**, in your words:

> **Inference Server**
> with:
>
> * Cloud LLM providers
> * Deterministic inference phases
> * Plugin-based validation, policy, transformation
> * No workflow semantics
> * No business process states
> * No human-in-the-loop lifecycle

This is **NOT** a workflow engine.

---

## 🧠 Correct Mental Model (Lock This)

### The engine is:

```
InferenceEngine
  └── InferencePipeline
        └── InferencePhase[]
              └── InferencePhasePlugin[]
```

Each inference request is:

```
Request → phases → provider → response
```

No looping.
No compensation.
No approval.
No state machine beyond **one request lifecycle**.

---

## 🔁 Proper Vocabulary Replacement (IMPORTANT)

| ❌ Wrong (Workflow) | ✅ Correct (Inference) |
| ------------------ | --------------------- |
| WorkflowEngine     | InferenceEngine       |
| Node               | InferenceRequest      |
| ExecutionToken     | InferenceContext      |
| ExecutionStatus    | InferenceStage        |
| Retry State        | Provider Retry Policy |
| WAITING            | STREAMING / PARTIAL   |
| COMPENSATION       | ❌ DOES NOT EXIST      |

---

## ✅ Correct Step 6 — Inference Lifecycle (REWRITTEN)

### 6.1 InferenceStage (NOT workflow state)

```java
public enum InferenceStage {
    CREATED,
    VALIDATING,
    PRE_PROCESSING,
    PROVIDER_DISPATCH,
    POST_PROCESSING,
    COMPLETED,
    FAILED
}
```

This is **linear**, not cyclic.

---

### 6.2 InferenceContext (Per Request)

```java
public interface InferenceContext {

    String requestId();

    InferenceStage stage();

    InferenceRequest request();

    InferenceResponse response();

    Map<String, Object> attributes();

    void advance(InferenceStage next);

    void fail(Throwable error);
}
```

✔ Exists only for **one inference call**
✔ Destroyed after completion
✔ Serializable only for logging/tracing (not resume)

---

### 6.3 InferenceEngine (FINAL, REAL)

```java
public interface InferenceEngine {

    InferenceResponse infer(InferenceRequest request);
}
```

That’s it.
No overloads.
No node.
No token.

---

### 6.4 DefaultInferenceEngine

```java
public final class DefaultInferenceEngine
        implements InferenceEngine {

    private final InferencePipeline pipeline;

    public DefaultInferenceEngine(InferencePipeline pipeline) {
        this.pipeline = pipeline;
    }

    @Override
    public InferenceResponse infer(InferenceRequest request) {

        InferenceContext context =
            new DefaultInferenceContext(request);

        pipeline.execute(context);

        if (context.response() == null) {
            throw new IllegalStateException(
                "Inference completed without response"
            );
        }

        return context.response();
    }
}
```

---

### 6.5 InferencePipeline (Phases Only)

```java
public interface InferencePipeline {

    void execute(InferenceContext context);
}
```

```java
public final class DefaultInferencePipeline
        implements InferencePipeline {

    private final Map<InferencePhase, List<InferencePhasePlugin>> plugins;

    @Override
    public void execute(InferenceContext context) {

        for (InferencePhase phase : InferencePhase.ordered()) {

            context.advance(phase.toStage());

            for (InferencePhasePlugin plugin : plugins.get(phase)) {
                plugin.execute(context, context.engine());
            }
        }
    }
}
```

---

## 🔒 What Is Explicitly OUT OF SCOPE (Now Locked)

The inference engine **will NOT** handle:

* ❌ Long-running orchestration
* ❌ Human approval
* ❌ Compensation
* ❌ Saga
* ❌ Workflow graphs
* ❌ Node retries beyond provider retry
* ❌ Persistence for resumption

Those belong to:

> **golek Orchestrator Layer**
> **NOT the Inference Runtime**

---

## 🧱 Proper Layering (Final)

```
golek
├── inference-kernel        ✅ (what we are designing)
│   ├── phases
│   ├── plugins
│   ├── providers
│   └── policies
│
├── orchestration-engine    ❌ (separate, later)
│
├── agent-runtime           ❌
│
└── workflow / ACP          ❌
```

---

## ✅ Next (Correct) Step 7 — Provider Dispatch Layer

If you agree, next step will be **strictly**:

> **Step 7: LLM Cloud Provider Abstraction**

* `LLMProvider`
* `ProviderRequestMapper`
* `ProviderRetryPolicy`
* Streaming vs non-streaming
* Provider capability model

No workflow concepts.
No state machine.
No hallucinated scope creep.

👉 Say **“go step 7 (provider)”** and I’ll continue **cleanly**.
