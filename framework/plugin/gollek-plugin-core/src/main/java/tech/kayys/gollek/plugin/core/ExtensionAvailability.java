/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.core;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Shared production-readiness report for detachable Gollek extensions.
 */
public record ExtensionAvailability(
        String id,
        String name,
        String kind,
        boolean attached,
        boolean detached,
        boolean healthy,
        boolean productionReady,
        String status,
        List<String> capabilities,
        List<String> formats,
        Map<String, String> attributes,
        String diagnostics,
        List<String> remediationHints) {

    public ExtensionAvailability {
        id = textOrDefault(id, "unknown");
        name = textOrDefault(name, id);
        kind = textOrDefault(kind, "extension");
        status = textOrDefault(status, "unknown");
        capabilities = List.copyOf(capabilities == null ? List.of() : capabilities);
        formats = List.copyOf(formats == null ? List.of() : formats);
        attributes = attributes == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(attributes));
        diagnostics = textOrDefault(diagnostics, "unavailable");
        remediationHints = List.copyOf(remediationHints == null ? List.of() : remediationHints);
    }

    public String compactSummary() {
        return "%s(attached=%s, detached=%s, productionReady=%s, formats=%s)"
                .formatted(
                        status,
                        attached,
                        detached,
                        productionReady,
                        formats.isEmpty() ? "none" : String.join("+", formats));
    }

    public boolean attributeBoolean(String key) {
        return Boolean.parseBoolean(attributes.getOrDefault(key, "false"));
    }

    public String attribute(String key, String fallback) {
        String value = attributes.get(key);
        return value == null || value.isBlank() ? fallback : value;
    }

    private static String textOrDefault(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }
}
