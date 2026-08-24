/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.alkhawarizm.spi.model.ModelFamilyContractViolation;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibilitySummary;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Combined release gate for packaged model-family bundle readiness.
 */
public record ModelFamilyBundleGate(
        boolean passed,
        String status,
        int violationCount,
        List<String> violations,
        Map<String, Integer> contractCategoryCounts,
        List<String> contractRemediationHints) {

    public ModelFamilyBundleGate {
        status = status == null || status.isBlank() ? (passed ? "passed" : "failed") : status;
        violations = List.copyOf(violations == null ? List.of() : violations);
        violationCount = Math.max(0, violationCount);
        contractCategoryCounts = Collections.unmodifiableMap(new LinkedHashMap<>(
                contractCategoryCounts == null ? Map.of() : contractCategoryCounts));
        contractRemediationHints = List.copyOf(
                contractRemediationHints == null ? List.of() : contractRemediationHints);
    }

    public ModelFamilyBundleGate(
            boolean passed,
            String status,
            int violationCount,
            List<String> violations) {
        this(passed, status, violationCount, violations, Map.of(), List.of());
    }

    public static ModelFamilyBundleGate evaluate(
            PluginAvailabilityChecker.ModelFamilyBundleAvailability availability,
            List<ModelFamilyContractViolation> contractViolations) {
        return evaluate(availability, contractViolations, null, false);
    }

    public static ModelFamilyBundleGate evaluate(
            PluginAvailabilityChecker.ModelFamilyBundleAvailability availability,
            List<ModelFamilyContractViolation> contractViolations,
            ModelFamilyRuntimeCompatibilitySummary runtimeCompatibility,
            boolean requireDirectSafetensorRuntime) {
        List<String> violations = new ArrayList<>();
        boolean availabilityPassed = availability != null && availability.healthy();
        if (!availabilityPassed) {
            if (availability == null) {
                violations.add("availability: model-family bundle availability was not evaluated");
            } else if (availability.problems().isEmpty()) {
                violations.add("availability: " + availability.status());
            } else {
                availability.problems().stream()
                        .map(problem -> "availability: " + problem)
                        .forEach(violations::add);
            }
        }

        List<ModelFamilyContractViolation> normalizedContracts =
                contractViolations == null ? List.of() : contractViolations;
        Map<String, Integer> contractCategoryCounts =
                ModelFamilyContractViolationReports.categories(normalizedContracts);
        List<String> contractRemediationHints =
                ModelFamilyContractViolationReports.remediationHints(normalizedContracts);
        normalizedContracts.stream()
                .map(violation -> "contract: " + violation.summary())
                .forEach(violations::add);

        boolean runtimePassed = true;
        if (requireDirectSafetensorRuntime) {
            runtimePassed = runtimeCompatibility != null
                    && !runtimeCompatibility.empty()
                    && runtimeCompatibility.problemCounts().isEmpty();
            if (runtimeCompatibility == null || runtimeCompatibility.empty()) {
                violations.add("runtime: direct SafeTensor runtime is required but no model-family "
                        + "compatibility summary was available");
            } else if (!runtimeCompatibility.problemCounts().isEmpty()) {
                violations.add("runtime: direct SafeTensor runtime has "
                        + runtimeCompatibility.problemCounts().size()
                        + " problem categories preventing compatibility: "
                        + String.join(", ", runtimeCompatibility.problemCounts().keySet()));
            }
        }

        boolean contractPassed = normalizedContracts.isEmpty();
        String status = gateStatus(availability, availabilityPassed, contractPassed, runtimePassed);
        return new ModelFamilyBundleGate(
                availabilityPassed && contractPassed && runtimePassed,
                status,
                violations.size(),
                violations,
                contractCategoryCounts,
                contractRemediationHints);
    }

    public boolean failed() {
        return !passed;
    }

    public String failureMessage() {
        String details = violations.isEmpty()
                ? ""
                : "\n  - " + String.join("\n  - ", violations);
        String recommendations = contractRemediationHints.isEmpty()
                ? ""
                : "\n  Recommendations:\n  - " + String.join("\n  - ", contractRemediationHints);
        return "Model-family bundle release gate failed: " + status + details + recommendations;
    }

    private static String gateStatus(
            PluginAvailabilityChecker.ModelFamilyBundleAvailability availability,
            boolean availabilityPassed,
            boolean contractPassed,
            boolean runtimePassed) {
        List<String> failed = new ArrayList<>();
        if (!availabilityPassed) {
            failed.add(availabilityFailureScope(availability));
        }
        if (!contractPassed) {
            failed.add("contract");
        }
        if (!runtimePassed) {
            failed.add("runtime");
        }
        if (failed.isEmpty()) {
            return "passed";
        }
        return String.join("_and_", failed) + "_failed";
    }

    private static String availabilityFailureScope(
            PluginAvailabilityChecker.ModelFamilyBundleAvailability availability) {
        if (availability == null) {
            return "availability";
        }
        return switch (availability.status()) {
            case "production_safety_failed" -> "production_safety";
            case "catalog_readiness_failed" -> "catalog_readiness";
            default -> "availability";
        };
    }
}
