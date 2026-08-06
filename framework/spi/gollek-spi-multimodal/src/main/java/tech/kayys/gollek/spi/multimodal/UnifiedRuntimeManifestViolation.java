/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.spi.multimodal;

/**
 * Machine-readable contract problem for unified multimodal runtime manifests.
 */
public record UnifiedRuntimeManifestViolation(
        String runtimeId,
        String code,
        String message) {

    public UnifiedRuntimeManifestViolation {
        runtimeId = runtimeId == null || runtimeId.isBlank() ? "unknown-unified-runtime" : runtimeId.trim();
        code = code == null || code.isBlank() ? "manifest_violation" : code.trim();
        message = message == null || message.isBlank() ? code : message.trim();
    }

    public String summary() {
        return runtimeId + "[" + code + "]: " + message;
    }
}
