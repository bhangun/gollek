

1️⃣ **What Golek Control Layer must provide (future-proof features)**
2️⃣ **Where each feature maps: Golek vs llama.cpp vs LiteRT**

---

# 1️⃣ Golek Control Layer — required features

Think of Golek as:

> **LLM Operating System**

Not just “call model”, but **control + reason + adapt**.

## 🧠 A. Prompt & Context Control

Golek must handle:

* ✅ System / user / assistant / tool roles
* ✅ Context window management (truncate, summarize, chunk)
* ✅ Prompt templating (agent prompt, tool prompt, RAG prompt)
* ✅ Multi-message conversation history
* ✅ Instruction injection (policies, schemas, tools)

👉 llama.cpp & LiteRT only see:
`String prompt`

---

## 🔁 B. Multi-turn Reasoning Loop

Golek must implement:

* ✅ Iterative reasoning loop
* ✅ Stop on:

  * final answer
  * tool call
  * max steps
* ✅ Retry on malformed output
* ✅ Self-repair JSON/tool call

Example responsibility:

```text
LLM → tool call → execute → feed back → LLM → final answer
```

Not:

```text
LLM → done
```

---

## 🛠️ C. Function / Tool Calling

Golek must provide:

* ✅ Tool registry
* ✅ JSON schema for tools
* ✅ Tool call detection
* ✅ Argument validation
* ✅ Tool execution
* ✅ Tool result injection
* ✅ Multiple tool calls per turn (future)

This is **not** in llama.cpp or LiteRT.

They don’t know what a “tool” is.

---

## 📦 D. Backend Abstraction (critical)

Golek must hide:

* llama.cpp
* LiteRT
* CUDA
* Metal
* OpenAI
* Gemini
* Groq
* etc.

So Golek exposes:

```java
generate(Request) -> Response
```

and internally decides:

```text
llama.cpp OR LiteRT OR remote LLM
```

---

## 🎛️ E. Sampling & Decoding Control

Golek must expose:

* temperature
* top-k
* top-p
* repetition penalty
* presence penalty
* mirostat
* stop tokens
* grammar / JSON mode

But backend does actual math.

Golek = policy
llama.cpp/LiteRT = execution

---

## 📚 F. Memory & State

Golek must support:

* ✅ Short-term memory (chat history)
* ✅ Long-term memory (vector DB, disk)
* ✅ Episodic memory (tool results)
* ✅ Summarization / compression
* ✅ Agent state

llama.cpp: ❌ none of this

---

## 🔌 G. RAG / External Knowledge

Golek must handle:

* chunking
* embeddings
* retrieval
* reranking
* context injection

llama.cpp just predicts tokens.

---

## 🧩 H. Agent & Workflow Integration

Golek must support:

* agent roles
* delegation
* multi-agent calls
* DAG / state machine integration (Gamelan)
* A2A protocol
* MCP protocol

llama.cpp does not coordinate agents.

---

## 📡 I. Streaming & Partial Output

Golek should provide:

* token streaming
* partial tool call streaming
* cancel / interrupt
* backpressure
* timeout

llama.cpp only streams tokens;
Golek turns that into:

* text
* tool calls
* events

---

## 🔐 J. Safety & Policy

Golek must implement:

* output filters
* tool allow/deny
* tenant isolation
* quota
* rate limit
* logging
* audit

llama.cpp has zero concept of tenants or security.

---

## 🧪 K. Observability

Golek should provide:

* token usage
* latency
* step count
* tool usage
* traces
* error causes

So later you can plug:

* OpenTelemetry
* Prometheus
* logs
* billing

---

# 2️⃣ Feature mapping: Golek vs llama.cpp vs LiteRT

Here is the important mapping table:

| Feature             | Golek (control) | llama.cpp   | LiteRT      |
| ------------------- | --------------- | ----------- | ----------- |
| Tokenization        | ❌ (delegate)    | ✅           | ✅           |
| Forward pass        | ❌               | ✅           | ✅           |
| KV cache            | ❌               | ✅           | ✅           |
| Quantization        | ❌               | ✅           | ✅           |
| GPU / NPU           | ❌               | ✅           | ✅           |
| Temperature / top-k | policy          | ✅ execution | ✅ execution |
| Grammar / JSON mode | policy          | ✅           | ⚠️ limited  |
| Prompt roles        | ✅               | ❌           | ❌           |
| Multi-turn loop     | ✅               | ❌           | ❌           |
| Function calling    | ✅               | ❌           | ❌           |
| Tool registry       | ✅               | ❌           | ❌           |
| Tool execution      | ✅               | ❌           | ❌           |
| Retry / repair      | ✅               | ❌           | ❌           |
| Memory              | ✅               | ❌           | ❌           |
| RAG                 | ✅               | ❌           | ❌           |
| Agents              | ✅               | ❌           | ❌           |
| Workflow            | via Gamelan     | ❌           | ❌           |
| Streaming API       | ✅               | partial     | partial     |
| Safety / policy     | ✅               | ❌           | ❌           |
| Observability       | ✅               | ❌           | ❌           |

Key insight:

> llama.cpp & LiteRT = **numerical engines**
> Golek = **cognitive control plane**

---

# 3️⃣ How this keeps Golek future-proof

Tomorrow you might swap:

* llama.cpp → vLLM
* LiteRT → Edge TPU
* Local → Groq
* CPU → NPU

If function calling was inside llama.cpp:
❌ rewrite everything

If function calling is in Golek:
✅ backend swap is trivial

This is **the same separation** as:

| OS    | CPU       |
| ----- | --------- |
| Linux | x86 / ARM |

Golek = Linux
llama.cpp/LiteRT = CPU

---

# 4️⃣ Minimal “must have” Golek modules

For v1 future-proof Golek:

Core:

* `InferenceBackend` interface
* `ConversationState`
* `PromptBuilder`
* `ToolRegistry`
* `ToolExecutor`
* `ReasoningLoop`
* `OutputParser`
* `MemoryStore`
* `StreamingController`

Later:

* `RAGService`
* `AgentRouter`
* `PolicyEngine`
* `MetricsCollector`

---

# Final takeaway

✅ **Golek should control:**

* reasoning
* tools
* memory
* loops
* policy
* abstraction

✅ **llama.cpp / LiteRT should only do:**

* token math
* sampling
* kv cache
* device optimization

If you violate this boundary:
👉 you lose portability
👉 you lose agent support
👉 you lose future models
👉 you lose multi-backend

---

