/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Enumerates the runtime capabilities that an {@link ExecutionProvider} may support.
 * Used by the intelligent router to match request requirements against available providers.
 *
 * <p>The router filters candidates whose {@link CapabilityProfile} satisfies all
 * required capabilities before scoring them on cost, latency, and quality.</p>
 */
public enum Capability {

    /** Provider maintains a KV cache across decode steps. */
    KV_CACHE,

    /** Provider supports speculative (draft+verify) decoding. */
    SPECULATIVE,

    /** Provider can emit tokens as a stream. */
    STREAMING,

    /** Provider accepts multimodal inputs (text + image/audio). */
    MULTIMODAL,

    /** Provider supports continuous batching of concurrent requests. */
    BATCHING,

    /** Provider can execute on GPU hardware. */
    GPU,

    /** Provider supports tool/function calling. */
    TOOL_CALLING,

    /** Provider can produce vector embeddings. */
    EMBEDDINGS,

    /** Provider supports separate prefill and decode execution phases. */
    PREFILL_DECODE_SPLIT,

    /** Provider supports structured (JSON schema) output generation. */
    STRUCTURED_OUTPUTS,

    /** Provider natively supports Mixture-of-Experts (MoE) architectures and sparse routing. */
    MOE
}
