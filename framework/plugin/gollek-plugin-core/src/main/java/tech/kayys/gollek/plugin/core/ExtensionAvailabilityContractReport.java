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
 * Aggregate release-gate view for detachable extension provider contracts.
 */
public record ExtensionAvailabilityContractReport(
        boolean passed,
        String status,
        int violationCount,
        List<ExtensionAvailabilityContractViolation> violations,
        Map<String, List<ExtensionAvailabilityContractViolation>> byExtensionId,
        List<String> summaries) {

    public ExtensionAvailabilityContractReport {
        violations = List.copyOf(violations == null ? List.of() : violations);
        byExtensionId = copyByExtensionId(byExtensionId);
        summaries = List.copyOf(summaries == null ? List.of() : summaries);
        violationCount = Math.max(0, violationCount);
        status = status == null || status.isBlank() ? (passed ? "passed" : "failed") : status.trim();
    }

    public static ExtensionAvailabilityContractReport fromViolations(
            List<ExtensionAvailabilityContractViolation> violations) {
        List<ExtensionAvailabilityContractViolation> normalized =
                List.copyOf(violations == null ? List.of() : violations);
        Map<String, List<ExtensionAvailabilityContractViolation>> byExtensionId = new LinkedHashMap<>();
        for (ExtensionAvailabilityContractViolation violation : normalized) {
            byExtensionId.computeIfAbsent(violation.extensionId(), ignored -> new java.util.ArrayList<>())
                    .add(violation);
        }
        boolean passed = normalized.isEmpty();
        return new ExtensionAvailabilityContractReport(
                passed,
                passed ? "passed" : "failed",
                normalized.size(),
                normalized,
                byExtensionId,
                normalized.stream()
                        .map(ExtensionAvailabilityContractViolation::summary)
                        .toList());
    }

    public boolean failed() {
        return !passed;
    }

    private static Map<String, List<ExtensionAvailabilityContractViolation>> copyByExtensionId(
            Map<String, List<ExtensionAvailabilityContractViolation>> values) {
        if (values == null || values.isEmpty()) {
            return Map.of();
        }
        Map<String, List<ExtensionAvailabilityContractViolation>> copy = new LinkedHashMap<>();
        values.forEach((extensionId, violations) -> copy.put(
                extensionId == null || extensionId.isBlank() ? "unknown" : extensionId.trim(),
                List.copyOf(violations == null ? List.of() : violations)));
        return Collections.unmodifiableMap(copy);
    }
}
