/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.core;

/**
 * Machine-readable contract problem for detachable extension availability providers.
 */
public record ExtensionAvailabilityContractViolation(
        String extensionId,
        String code,
        String message) {

    public ExtensionAvailabilityContractViolation {
        extensionId = textOrDefault(extensionId, "unknown");
        code = textOrDefault(code, "contract_violation");
        message = textOrDefault(message, code);
    }

    public String summary() {
        return extensionId + "[" + code + "]: " + message;
    }

    private static String textOrDefault(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value.trim();
    }
}
