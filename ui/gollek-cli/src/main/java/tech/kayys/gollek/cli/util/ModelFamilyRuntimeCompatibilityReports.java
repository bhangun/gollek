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
                registry.directSafetensorCompatibilitiesForFamilies(selectedFamilies).stream()
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
        report.put(Summary.RUNTIME_ID, summary.runtimeId());
        report.put(Summary.FAMILY_COUNT, summary.familyCount());
        report.put(Summary.COMPATIBLE_FAMILY_COUNT, summary.compatibleFamilyCount());
        report.put(Summary.BLOCKED_FAMILY_COUNT, summary.blockedFamilyCount());
        report.put(Summary.ATTENTION_FAMILY_COUNT, summary.attentionFamilyCount());
        report.put(Summary.ARCHITECTURE_ADAPTER_READY_COUNT, summary.architectureAdapterReadyCount());
        report.put(Summary.TOKENIZER_READY_COUNT, summary.tokenizerReadyCount());
        report.put(Summary.TOKENIZER_FILE_INSPECTION_AVAILABLE_COUNT,
                summary.tokenizerFileInspectionAvailableCount());
        report.put(Summary.COMPATIBLE_FAMILY_IDS, summary.compatibleFamilyIds());
        report.put(Summary.BLOCKED_FAMILY_IDS, summary.blockedFamilyIds());
        report.put(Summary.PROBLEM_COUNTS, summary.problemCounts());
        report.put(Summary.ALL_COMPATIBLE, summary.allCompatible());
        report.put(Summary.EMPTY, summary.empty());
        return report;
    }

    private static Map<String, Object> directSafetensor(ModelFamilyRuntimeCompatibility compatibility) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(DirectSafetensorCompatibility.RUNTIME_ID, compatibility.runtimeId());
        report.put(DirectSafetensorCompatibility.COMPATIBLE, compatibility.compatible());
        report.put(DirectSafetensorCompatibility.REQUIRES_ATTENTION, compatibility.requiresAttention());
        report.put(DirectSafetensorCompatibility.SUMMARY, compatibility.summary());
        report.put(DirectSafetensorCompatibility.FAMILY_IDS, compatibility.modelFamily().familyIds());
        report.put(DirectSafetensorCompatibility.MODEL_TYPE, compatibility.modelFamily().modelType());
        report.put(DirectSafetensorCompatibility.ARCHITECTURE_CLASS_NAME,
                compatibility.modelFamily().architectureClassName());
        report.put(DirectSafetensorCompatibility.SELECTED_ARCHITECTURE_ADAPTER_ID,
                compatibility.selectedArchitectureAdapterId());
        report.put(DirectSafetensorCompatibility.SELECTED_ARCHITECTURE_ADAPTER_BY,
                compatibility.selectedArchitectureAdapterBy());
        report.put(DirectSafetensorCompatibility.ARCHITECTURE_ADAPTER_IDS,
                compatibility.architectureAdapterIds());
        report.put(DirectSafetensorCompatibility.ARCHITECTURE_ADAPTER_READY,
                compatibility.architectureAdapterReady());
        report.put(DirectSafetensorCompatibility.TOKENIZER_READY, compatibility.tokenizerReady());
        report.put(DirectSafetensorCompatibility.TOKENIZER_FILE_INSPECTION_AVAILABLE,
                compatibility.tokenizerFileInspectionAvailable());
        report.put(DirectSafetensorCompatibility.USABLE_TOKENIZER_IDS, compatibility.usableTokenizerIds());
        report.put(DirectSafetensorCompatibility.PROBLEM_CODES, compatibility.problemCodes());
        report.put(DirectSafetensorCompatibility.REMEDIATION_HINTS, compatibility.remediationHints());
        return report;
    }
}
