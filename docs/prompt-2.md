

How about with these:

InferencePhase:

PRE_VALIDATE
VALIDATE
AUTHORIZE
ROUTE
PRE_PROCESSING
PROVIDER_DISPATCH
POST_PROCESSING
AUDIT
OBSERVABILITY   
CLEANUP



# 🔹 Step 6 — Execution Lifecycle & State Machine (Engine-Grade)

This step makes your engine:

✔ deterministic
✔ resumable
✔ retry-safe
✔ human-aware
✔ agent-friendly
✔ **not hallucinated / not contradictory**

---

## 6.1 Core Principle (Read This First)

Your engine **does NOT execute nodes**.
It **advances execution state**.

Execution = **state transition**, not function calls.

That’s what separates:

* toy workflow engines ❌
* real BPMN / ESB / agent runtimes ✅

---

## 6.2 Canonical Execution State Model

```java
public enum ExecutionStatus {

    CREATED,        // compiled, not started
    RUNNING,        // actively executing
    WAITING,        // external event / human / agent
    SUSPENDED,      // paused by policy
    RETRYING,       // retry backoff
    COMPLETED,      // success
    FAILED,         // terminal failure
    COMPENSATED,    // rollback completed
    CANCELLED
}
```

⚠️ These states are **global truth**
Executors must **never invent their own states**.

---

## 6.3 ExecutionToken (Single Source of Truth)

```java
public final class ExecutionToken {

    private final String executionId;
    private final String nodeId;
    private final ExecutionStatus status;
    private final int attempt;
    private final Instant lastUpdated;

    private final Map<String, Object> variables;
    private final Map<String, Object> metadata;

    // immutable
}
```

✔ Serializable
✔ Persistable
✔ Rehydration-safe
✔ Distributed-safe

---

## 6.4 ExecutionContext vs EngineContext (No Confusion)

| Context            | Scope             |
| ------------------ | ----------------- |
| `EngineContext`    | engine kernel     |
| `ExecutionContext` | **one execution** |

```java
public interface ExecutionContext {

    EngineContext engine();

    ExecutionToken token();

    Map<String, Object> variables();

    void updateStatus(ExecutionStatus status);

    void putVariable(String key, Object value);
}
```

⚠️ Executors only see `ExecutionContext`, **never Engine internals**.

---

## 6.5 State Machine (Deterministic)

```java
public interface ExecutionStateMachine {

    ExecutionStatus next(
        ExecutionStatus current,
        ExecutionSignal signal
    );
}
```

### Signals (Not States)

```java
public enum ExecutionSignal {

    START,
    EXECUTOR_SUCCESS,
    EXECUTOR_FAILURE,
    RETRY_EXHAUSTED,
    WAIT_REQUESTED,
    HUMAN_APPROVED,
    HUMAN_REJECTED,
    COMPENSATION_DONE,
    CANCEL
}
```

✔ Clean
✔ Formal
✔ Testable

---

## 6.6 Default State Machine (Engine-Owned)

```java
public final class DefaultExecutionStateMachine
        implements ExecutionStateMachine {

    @Override
    public ExecutionStatus next(
        ExecutionStatus current,
        ExecutionSignal signal
    ) {
        return switch (current) {

            case CREATED -> signal == ExecutionSignal.START
                ? ExecutionStatus.RUNNING
                : current;

            case RUNNING -> switch (signal) {
                case EXECUTOR_SUCCESS -> ExecutionStatus.COMPLETED;
                case EXECUTOR_FAILURE -> ExecutionStatus.RETRYING;
                case WAIT_REQUESTED -> ExecutionStatus.WAITING;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case RETRYING -> signal == ExecutionSignal.RETRY_EXHAUSTED
                ? ExecutionStatus.FAILED
                : ExecutionStatus.RUNNING;

            case WAITING -> switch (signal) {
                case HUMAN_APPROVED -> ExecutionStatus.RUNNING;
                case HUMAN_REJECTED -> ExecutionStatus.FAILED;
                default -> current;
            };

            default -> current;
        };
    }
}
```

---

## 6.7 WorkflowEngine (FINAL, NON-CONTRADICTORY)

This resolves your earlier complaint 👇

> *“which one is the main? class vs interface?”*

### ✅ THIS is the **only engine contract**

```java
public interface WorkflowEngine {

    ExecutionResult execute(
        ExecutionContext context,
        NodeDescriptor node,
        ExecutionToken token
    );
}
```

✔ Node-level
✔ Stateless
✔ Idempotent
✔ Re-entrant

---

## 6.8 DefaultWorkflowEngine (State-Driven)

```java
public final class DefaultWorkflowEngine
        implements WorkflowEngine {

    private final ExecutionStateMachine stateMachine;

    public DefaultWorkflowEngine(ExecutionStateMachine stateMachine) {
        this.stateMachine = stateMachine;
    }

    @Override
    public ExecutionResult execute(
        ExecutionContext context,
        NodeDescriptor node,
        ExecutionToken token
    ) {

        context.updateStatus(
            stateMachine.next(token.status(), ExecutionSignal.START)
        );

        ExecutionResult result =
            context.engine()
                   .executorDispatcher()
                   .dispatch(context, node.executor(), token);

        ExecutionSignal signal = result.toSignal();

        context.updateStatus(
            stateMachine.next(context.token().status(), signal)
        );

        return result;
    }
}
```

⚠️ No business logic
⚠️ No retry logic here
⚠️ No infra logic

Pure state advancement.

---

## 6.9 Retry & Backoff (Policy-Driven)

```java
public interface RetryPolicy {

    boolean shouldRetry(int attempt, Throwable error);

    Duration backoff(int attempt);
}
```

Retries are **externalized**, not hardcoded.

---

## 6.10 Human-in-the-Loop (First-Class)

Executors may return:

```java
ExecutionResult.waiting("approval_required");
```

Which triggers:

```
RUNNING → WAITING
```

Later:

```
WAITING → RUNNING
```

without replaying side effects.

---

## 6.11 Why This Works for Agentic AI

✔ Long negotiations
✔ Partial agreement
✔ Tool calls
✔ ACP / MCP
✔ Distributed agents
✔ Offline rehydration

---
