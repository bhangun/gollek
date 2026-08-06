/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

/**
 * A stateful execution session that bridges the scheduler to the actual
 * inference kernel (torch, gguf, onnx, safetensor, etc.).
 *
 * <p>The session splits inference into two explicit phases:</p>
 * <ol>
 *   <li><b>Prefill</b> — processes the full prompt in parallel, populates the KV cache,
 *       and returns the first predicted token.</li>
 *   <li><b>Decode</b> — iteratively generates one token at a time using the KV cache
 *       built during prefill.</li>
 * </ol>
 *
 * <p>For runners that do not natively separate prefill from decode (e.g., simple
 * GGUF wrappers), the default {@link #prefill} implementation delegates to
 * a combined generate call that returns the first token.</p>
 *
 * <p>Sessions are <b>not thread-safe</b>. The scheduler must ensure that at most
 * one thread interacts with a session at any given time.</p>
 */
public interface RuntimeSession extends AutoCloseable {

    /**
     * Execute the prefill phase: process the full input prompt and produce
     * the first output token.
     *
     * @param request the tokenised prompt and generation parameters
     * @return the first predicted token and KV cache metadata
     */
    PrefillResult prefill(PrefillRequest request);

    /**
     * Execute a single decode step: consume the previously predicted token
     * and produce the next one.
     *
     * @param request the last token and current sequence position
     * @return the next predicted token
     */
    DecodeResult decode(DecodeRequest request);

    /**
     * Release all resources held by this session (KV cache, native memory, etc.).
     */
    @Override
    void close();
}
