/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

import java.util.List;
import java.util.Objects;

/**
 * Request carrier for the prefill phase of a {@link RuntimeSession}.
 *
 * <p>Prefill processes the full input prompt in parallel, producing the
 * initial KV cache entries and the first set of logits.</p>
 *
 * @param requestId   unique identifier for tracing
 * @param tokenIds    the tokenised input prompt
 * @param maxNewTokens maximum tokens to generate in the subsequent decode phase
 */
public record PrefillRequest(
        String requestId,
        List<Integer> tokenIds,
        int maxNewTokens) {

    public PrefillRequest {
        Objects.requireNonNull(requestId, "requestId");
        Objects.requireNonNull(tokenIds, "tokenIds");
        tokenIds = List.copyOf(tokenIds);
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be > 0");
        }
    }
}
