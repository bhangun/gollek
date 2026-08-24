/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.cli.util.ModelFamilyRuntimeCompatibilityReportFields.Compatibility;
import tech.kayys.gollek.cli.util.ModelFamilyRuntimeCompatibilityReportFields.DirectSafetensorCompatibility;
import tech.kayys.gollek.cli.util.ModelFamilyRuntimeCompatibilityReportFields.Summary;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibility;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibilitySummary;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds model-family runtime compatibility sections for bundle CI reports.
 */
final class ModelFamilyRuntimeCompatibilityReports {
    private ModelFamilyRuntimeCompatibilityReports() {
    }

    static Map<String, Object> compatibility(
            ModelFamilyBundleManifest manifest,
            ModelFamilyPluginRegistry registry) {
        Map<String, Object> report = new LinkedHashMap<>();
        List<String> selectedFamilies = manifest == null ? List.of() : manifest.families();
        report.put(Compatibility.REQUIRES_DIRECT_SAFETENSOR_RUNTIME,
                manifest != null && manifest.requiresDirectSafetensorRuntime());
        report.put(Compatibility.SELECTED_FAMILY_IDS, selectedFamilies);
        report.put(Compatibility.SELECTED_DIRECT_SAFETENSOR_SUMMARY, summary(
                registry.directSafetensorCompatibilitySummaryForFamilies(selectedFamilies)));
        report.put(Compatibility.SELECTED_DIRECT_SAFETENSOR,
                registry.directSafetensorCompatibilities().stream()
                        .filter(c -> selectedFamilies.isEmpty()
                                || selectedFamilies.stream().anyMatch(id ->
                                        c.modelFamily().familyIds().contains(id)))
                        .map(ModelFamilyRuntimeCompatibilityReports::directSafetensor)
                        .toList());
        report.put(Compatibility.DIRECT_SAFETENSOR_SUMMARY, summary(
                registry.directSafetensorCompatibilitySummary()));
        report.put(Compatibility.DIRECT_SAFETENSOR, registry.directSafetensorCompatibilities().stream()
                .map(ModelFamilyRuntimeCompatibilityReports::directSafetensor)
                .toList());
        return report;
    }

    private static Map<String, Object> summary(ModelFamilyRuntimeCompatibilitySummary summary) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(Summary.FAMILY_COUNT, summary.familyCount());
        report.put(Summary.COMPATIBLE_FAMILY_IDS, summary.compatibleFamilyIds());
        report.put(Summary.PROBLEM_COUNTS, summary.problemCounts());
        report.put(Summary.EMPTY, summary.empty());
        return report;
    }

    private static Map<String, Object> directSafetensor(ModelFamilyRuntimeCompatibility compatibility) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(DirectSafetensorCompatibility.COMPATIBLE, compatibility.compatible());
        report.put(DirectSafetensorCompatibility.FAMILY_IDS, compatibility.modelFamily().familyIds());
        report.put(DirectSafetensorCompatibility.MODEL_TYPE, compatibility.modelFamily().modelType());
        report.put(DirectSafetensorCompatibility.ARCHITECTURE_CLASS_NAME,
                compatibility.modelFamily().architectureClassName());
        report.put(DirectSafetensorCompatibility.SELECTED_ARCHITECTURE_ADAPTER_ID,
                compatibility.selectedArchitectureAdapterId());
        report.put(DirectSafetensorCompatibility.SELECTED_ARCHITECTURE_ADAPTER_BY,
                compatibility.selectedArchitectureAdapterBy());
        report.put(DirectSafetensorCompatibility.PROBLEM_CODES, compatibility.problemCodes());
        report.put(DirectSafetensorCompatibility.REMEDIATION_HINTS, compatibility.remediationHints());
        return report;
    }
}
