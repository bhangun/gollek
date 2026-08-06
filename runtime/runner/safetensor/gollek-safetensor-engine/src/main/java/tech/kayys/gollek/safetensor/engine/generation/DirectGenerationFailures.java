/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.jboss.logging.Logger;

import java.util.Objects;

/**
 * Builds consistent direct-generation runtime failures.
 */
final class DirectGenerationFailures {

    private DirectGenerationFailures() {
    }

    static RuntimeException wrap(Logger log, String logMessage, String failurePrefix, Exception cause) {
        Objects.requireNonNull(cause, "cause");
        String safePrefix = blankToDefault(failurePrefix, "Direct generation failed");
        if (log != null) {
            log.error(blankToDefault(logMessage, safePrefix), cause);
        }
        String detail = cause.getMessage();
        if (detail == null || detail.isBlank()) {
            return new RuntimeException(safePrefix, cause);
        }
        return new RuntimeException(safePrefix + ": " + detail, cause);
    }

    private static String blankToDefault(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }
}
