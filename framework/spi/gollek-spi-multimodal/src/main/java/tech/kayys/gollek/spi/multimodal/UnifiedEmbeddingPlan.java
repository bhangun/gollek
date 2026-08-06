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
 * Normalized embedding/preprocessor plan emitted by unified runtime plugins.
 */
public record UnifiedEmbeddingPlan(
        String runtimeId,
        String modelType,
        List<UnifiedInputModality> modalities,
        List<String> requiredProcessorFiles,
        List<String> requiredTokenizerFiles,
        Map<String, String> metadata) {

    public UnifiedEmbeddingPlan {
        runtimeId = textOrDefault(runtimeId, "unknown-unified-runtime");
        modelType = normalize(modelType, true);
        modalities = copyModalities(modalities);
        requiredProcessorFiles = copyTextList(requiredProcessorFiles, false);
        requiredTokenizerFiles = copyTextList(requiredTokenizerFiles, false);
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

    private static List<String> copyTextList(List<String> values, boolean lowerCase) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .map(value -> normalize(value, lowerCase))
                .filter(value -> !value.isBlank())
                .distinct()
                .toList();
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
