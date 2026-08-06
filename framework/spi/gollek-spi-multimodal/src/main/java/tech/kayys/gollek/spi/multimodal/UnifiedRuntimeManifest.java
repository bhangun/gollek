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
 * Stable capabilities advertised by a detachable unified multimodal runtime.
 */
public record UnifiedRuntimeManifest(
        String runtimeId,
        String displayName,
        List<String> modelFamilyIds,
        List<String> modelTypes,
        List<UnifiedInputModality> inputModalities,
        UnifiedRuntimeReadiness readiness,
        String readinessReason,
        List<String> requiredProcessorFiles,
        List<String> requiredTokenizerFiles,
        Map<String, String> metadata) {

    public UnifiedRuntimeManifest {
        runtimeId = textOrDefault(runtimeId, "unknown-unified-runtime");
        displayName = textOrDefault(displayName, runtimeId);
        modelFamilyIds = copyTextList(modelFamilyIds, false);
        modelTypes = copyTextList(modelTypes, true);
        inputModalities = copyModalities(inputModalities);
        readiness = readiness == null ? UnifiedRuntimeReadiness.UNAVAILABLE : readiness;
        readinessReason = textOrDefault(readinessReason, readiness.statusLabel());
        requiredProcessorFiles = copyTextList(requiredProcessorFiles, false);
        requiredTokenizerFiles = copyTextList(requiredTokenizerFiles, false);
        metadata = copyMetadata(metadata);
    }

    public boolean supportsFamily(String familyId) {
        String normalized = normalize(familyId, false);
        return !normalized.isBlank() && modelFamilyIds.contains(normalized);
    }

    public boolean supportsModelType(String modelType) {
        String normalized = normalize(modelType, true);
        return !normalized.isBlank() && modelTypes.contains(normalized);
    }

    public boolean productionReady() {
        return readiness.productionReady();
    }

    public boolean attached() {
        return readiness.attached();
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

    private static List<UnifiedInputModality> copyModalities(List<UnifiedInputModality> modalities) {
        if (modalities == null || modalities.isEmpty()) {
            return List.of();
        }
        return modalities.stream()
                .filter(modality -> modality != null)
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
