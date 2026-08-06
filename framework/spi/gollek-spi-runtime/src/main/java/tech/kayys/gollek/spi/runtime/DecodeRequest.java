/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Objects;

/**
 * Request carrier for a single decode step inside a {@link RuntimeSession}.
 *
 * <p>Each decode call consumes the previously predicted token and produces
 * the next one, extending the KV cache by one entry.</p>
 *
 * @param previousTokenId the token produced by the last prefill or decode step
 * @param currentPosition the current sequence position (0-based)
 */
public record DecodeRequest(
        int previousTokenId,
        int currentPosition) {

    public DecodeRequest {
        if (currentPosition < 0) {
            throw new IllegalArgumentException("currentPosition must be >= 0");
        }
    }
}
