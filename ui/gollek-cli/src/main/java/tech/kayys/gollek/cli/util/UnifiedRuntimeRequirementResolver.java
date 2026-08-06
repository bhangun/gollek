/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeManifest;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyUnifiedRuntimeRequirement;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeManifest;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeRegistry;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Bridges model-family runtime requirements with attached unified multimodal runtimes.
 */
public final class UnifiedRuntimeRequirementResolver {
    private UnifiedRuntimeRequirementResolver() {
    }

    public static List<UnifiedRuntimeRequirementCompatibility> evaluate(
            ModelFamilyPluginRegistry modelFamilies,
            UnifiedRuntimeRegistry unifiedRuntimes) {
        return evaluate(modelFamilies, unifiedRuntimes, List.of());
    }

    public static List<UnifiedRuntimeRequirementCompatibility> evaluate(
            ModelFamilyPluginRegistry modelFamilies,
            UnifiedRuntimeRegistry unifiedRuntimes,
            Collection<String> selectedFamilyIds) {
        if (modelFamilies == null) {
            return List.of();
        }
        UnifiedRuntimeRegistry runtimeRegistry = unifiedRuntimes == null
                ? UnifiedRuntimeRegistry.of(List.of())
                : unifiedRuntimes;
        Set<String> selected = normalizedSet(selectedFamilyIds);
        return modelFamilies.runtimeManifests().stream()
                .filter(manifest -> selected.isEmpty() || selected.contains(normalize(manifest.familyId())))
                .flatMap(manifest -> manifest.unifiedRuntimeRequirements().stream()
                        .map(requirement -> evaluateRequirement(manifest, requirement, runtimeRegistry)))
                .toList();
    }

    private static UnifiedRuntimeRequirementCompatibility evaluateRequirement(
            ModelFamilyRuntimeManifest family,
            ModelFamilyUnifiedRuntimeRequirement requirement,
            UnifiedRuntimeRegistry runtimeRegistry) {
        List<UnifiedRuntimeRegistry.UnifiedRuntimeReport> reports =
                runtimeRegistry.reportsSupportingModelType(requirement.modelType());
        if (reports.isEmpty()) {
            return attention(
                    family,
                    requirement,
                    UnifiedRuntimeRequirementIssueKind.MISSING_RUNTIME,
                    List.of(),
                    List.of(),
                    "Attach one unified runtime plugin that claims model_type=" + requirement.modelType() + ".");
        }
        List<String> runtimeIds = reports.stream()
                .map(UnifiedRuntimeRegistry.UnifiedRuntimeReport::runtimeId)
                .distinct()
                .toList();
        if (reports.size() > 1) {
            return attention(
                    family,
                    requirement,
                    UnifiedRuntimeRequirementIssueKind.CONFLICTING_RUNTIME,
                    runtimeIds,
                    reports.stream()
                            .filter(UnifiedRuntimeRegistry.UnifiedRuntimeReport::manifestAvailable)
                            .flatMap(report -> modalityLabels(report.manifest()).stream())
                            .distinct()
                            .sorted()
                            .toList(),
                    "Detach duplicate unified runtime plugins for model_type=" + requirement.modelType() + ".");
        }
        UnifiedRuntimeRegistry.UnifiedRuntimeReport report = reports.getFirst();
        if (!report.valid()) {
            return attention(
                    family,
                    requirement,
                    UnifiedRuntimeRequirementIssueKind.INVALID_RUNTIME,
                    runtimeIds,
                    report.manifestAvailable() ? modalityLabels(report.manifest()) : List.of(),
                    "Fix unified runtime manifest violations for " + report.runtimeId() + ".");
        }
        UnifiedRuntimeManifest manifest = report.manifest();
        List<String> availableModalities = modalityLabels(manifest);
        List<String> missingModalities = requirement.requiredInputModalities().stream()
                .filter(modality -> !availableModalities.contains(modality))
                .toList();
        if (!missingModalities.isEmpty()) {
            return attention(
                    family,
                    requirement,
                    UnifiedRuntimeRequirementIssueKind.INSUFFICIENT_MODALITIES,
                    runtimeIds,
                    availableModalities,
                    "Attach a unified runtime for model_type=" + requirement.modelType()
                            + " with modalities: " + String.join(", ", missingModalities) + ".");
        }
        if (requirement.productionReadyRequired() && !manifest.productionReady()) {
            return attention(
                    family,
                    requirement,
                    UnifiedRuntimeRequirementIssueKind.NOT_PRODUCTION_READY,
                    runtimeIds,
                    availableModalities,
                    "Keep " + family.familyId()
                            + " out of production bundles until its unified runtime reports ready.");
        }
        return UnifiedRuntimeRequirementCompatibility.ready(
                family.familyId(),
                requirement.modelType(),
                requirement.requiredInputModalities(),
                requirement.productionReadyRequired(),
                runtimeIds,
                availableModalities);
    }

    private static UnifiedRuntimeRequirementCompatibility attention(
            ModelFamilyRuntimeManifest family,
            ModelFamilyUnifiedRuntimeRequirement requirement,
            UnifiedRuntimeRequirementIssueKind issueKind,
            List<String> runtimeIds,
            List<String> availableInputModalities,
            String remediationHint) {
        return UnifiedRuntimeRequirementCompatibility.attention(
                family.familyId(),
                requirement.modelType(),
                requirement.requiredInputModalities(),
                requirement.productionReadyRequired(),
                issueKind,
                runtimeIds,
                availableInputModalities,
                remediationHint);
    }

    private static List<String> modalityLabels(UnifiedRuntimeManifest manifest) {
        if (manifest == null) {
            return List.of();
        }
        return manifest.inputModalities().stream()
                .map(modality -> modality.name().toLowerCase(Locale.ROOT))
                .distinct()
                .sorted()
                .toList();
    }

    private static Set<String> normalizedSet(Collection<String> values) {
        if (values == null || values.isEmpty()) {
            return Set.of();
        }
        LinkedHashSet<String> normalized = new LinkedHashSet<>();
        values.stream()
                .map(UnifiedRuntimeRequirementResolver::normalize)
                .filter(value -> !value.isBlank())
                .forEach(normalized::add);
        return Set.copyOf(normalized);
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }
}
