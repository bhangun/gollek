/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Result of a single decode step inside a {@link RuntimeSession}.
 *
 * @param tokenId       the predicted next token
 * @param isEos         {@code true} if the model signalled end-of-sequence
 * @param decodeTimeMs  wall-clock time for this decode step
 */
public record DecodeResult(
        int tokenId,
        boolean isEos,
        double decodeTimeMs) {
}
