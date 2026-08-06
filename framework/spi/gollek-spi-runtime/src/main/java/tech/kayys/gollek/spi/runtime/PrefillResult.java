/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Objects;

/**
 * Result of a prefill execution inside a {@link RuntimeSession}.
 *
 * <p>Contains the first predicted token and metadata about the
 * KV cache state after prompt processing.</p>
 *
 * @param firstTokenId   the token predicted after processing the prompt
 * @param kvCacheLength  number of KV entries now stored (= prompt length)
 * @param prefillTimeMs  wall-clock time spent on prefill
 */
public record PrefillResult(
        int firstTokenId,
        int kvCacheLength,
        double prefillTimeMs) {

    public PrefillResult {
        if (kvCacheLength < 0) {
            throw new IllegalArgumentException("kvCacheLength must be >= 0");
        }
    }
}
