/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.spi.multimodal;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Provider-neutral input envelope for unified multimodal runtime plugins.
 */
public record UnifiedMultimodalRequest(
        String modelId,
        String modelType,
        List<UnifiedInputModality> modalities,
        Map<String, Object> inputs,
        Map<String, String> metadata) {

    public UnifiedMultimodalRequest {
        modelId = textOrDefault(modelId, "unknown-model");
        modelType = normalize(modelType, true);
        modalities = copyModalities(modalities);
        inputs = copyInputs(inputs);
        metadata = copyMetadata(metadata);
    }

    private static List<UnifiedInputModality> copyModalities(List<UnifiedInputModality> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .filter(value -> value != null)
                .distinct()
                .toList();
    }

    private static Map<String, Object> copyInputs(Map<String, Object> values) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        values.forEach((key, value) -> {
            String normalizedKey = normalize(key, false);
            if (!normalizedKey.isBlank()) {
                copy.put(normalizedKey, value);
            }
        });
        return Collections.unmodifiableMap(copy);
    }

    private static Map<String, String> copyMetadata(Map<String, String> values) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, String> copy = new LinkedHashMap<>();
        values.forEach((key, value) -> {
            String normalizedKey = normalize(key, false);
            if (!normalizedKey.isBlank() && value != null) {
                copy.put(normalizedKey, value.trim());
            }
        });
        return Collections.unmodifiableMap(copy);
    }

    private static String textOrDefault(String value, String fallback) {
        String normalized = normalize(value, false);
        return normalized.isBlank() ? fallback : normalized;
    }

    private static String normalize(String value, boolean lowerCase) {
        if (value == null) {
            return "";
        }
        String trimmed = value.trim();
        return lowerCase ? trimmed.toLowerCase(Locale.ROOT) : trimmed;
    }
}
