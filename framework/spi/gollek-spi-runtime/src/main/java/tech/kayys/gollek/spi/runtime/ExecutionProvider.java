/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

import tech.kayys.alkhawarizm.spi.model.HealthStatus;

/**
 * Unified provider abstraction that bridges <b>local runners</b>
 * (torch, gguf, safetensor, onnx, litert) and <b>remote API providers</b>
 * (OpenAI, Gemini, Anthropic) behind a single contract.
 *
 * <p>
 * The intelligent router scores candidates of this type using their
 * {@link #capabilities()}, {@link #costProfile()}, and {@link #health()}
 * to produce an optimal {@code RoutingDecision}.
 * </p>
 *
 * <h3>Relationship to {@code LLMProvider}</h3>
 * <p>
 * {@code ExecutionProvider} is a <b>parallel</b> abstraction — it does not
 * extend {@code LLMProvider}. Existing {@code LLMProvider} implementations
 * can be adapted via a thin wrapper that delegates
 * {@link #createSession(SessionConfig)} to the provider's {@code infer()}
 * method.
 * </p>
 */
public interface ExecutionProvider {

    /**
     * Unique provider identifier (e.g. "libtorch-local", "openai-gpt4").
     */
    String id();

    /**
     * Whether this provider executes inference locally (same JVM / node)
     * or delegates to a remote API.
     */
    boolean isLocal();

    /**
     * Set of capabilities supported by this provider.
     */
    CapabilityProfile capabilities();

    /**
     * Proposes a sequence of tokens (speculative drafting).
     * 
     * @param context   the current KV session context
     * @param lookahead number of tokens to propose
     * @return a future resolving to a SpeculativeBatch
     */
    default java.util.concurrent.CompletableFuture<SpeculativeBatch> speculate(RuntimeSession context, int lookahead) {
        return java.util.concurrent.CompletableFuture.failedFuture(
                new UnsupportedOperationException("Speculation not supported by " + id()));
    }

    /**
     * Cost and performance characteristics for scoring.
     */
    CostProfile costProfile();

    /**
     * Current health status.
     */
    HealthStatus health();

    /**
     * Information about the node where this provider is hosted.
     * This is used for locality-aware routing in distributed environments.
     */
    default NodeInfo node() {
        return NodeInfo.LOCAL;
    }

    /**
     * Create a new execution session for the given configuration.
     *
     * <p>
     * The returned session holds runtime state (loaded model weights,
     * KV cache, native memory) and must be {@link RuntimeSession#close() closed}
     * when no longer needed.
     * </p>
     *
     * @param config model and device configuration
     * @return a new session ready for prefill + decode
     */
    RuntimeSession createSession(SessionConfig config);
}
