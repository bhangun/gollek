/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.cli.util.ModelFamilyBundleInventoryReportFields.Family;
import tech.kayys.gollek.cli.util.ModelFamilyBundleInventoryReportFields.RuntimeManifest;
import tech.kayys.gollek.cli.util.ModelFamilyBundleInventoryReportFields.RuntimeRequirement;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeManifest;
import tech.kayys.alkhawarizm.spi.model.ModelFamilySupportReport;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyUnifiedRuntimeRequirement;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Builds the reusable model-family inventory section for bundle CI reports.
 */
final class ModelFamilyBundleInventoryReports {
    private ModelFamilyBundleInventoryReports() {
    }

    static Map<String, Object> family(ModelFamilySupportReport family) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(Family.ID, family.id());
        report.put(Family.DISPLAY_NAME, family.displayName());
        report.put(Family.BUNDLE_PROFILE, family.bundleProfile().key());
        report.put(Family.DEFAULT_BUNDLE, family.defaultBundle());
        report.put(Family.MODEL_TYPES, family.modelTypes());
        report.put(Family.ARCHITECTURE_CLASS_NAMES, family.architectureClassNames());
        report.put(Family.ARCHITECTURE_ADAPTER_IDS, family.architectureAdapterIds());
        report.put(Family.TOKENIZER_PROFILE_IDS, family.tokenizerProfileIds());
        report.put(Family.TOKENIZER_KINDS, family.tokenizerKinds().stream()
                .map(kind -> kind.name().toLowerCase())
                .toList());
        report.put(Family.CAPABILITIES, family.capabilities().stream()
                .map(capability -> capability.name().toLowerCase())
                .toList());
        report.put(Family.DIRECT_SAFETENSOR_STATUS, family.directSafetensorStatus().label());
        report.put(Family.DIRECT_SAFETENSOR_READY, family.directSafetensorReady());
        report.put(Family.DIRECT_SAFETENSOR_REASON, family.directSafetensorReason());
        report.put(Family.DIRECT_SAFETENSOR_CAVEATS, family.directSafetensorCaveats());
        report.put(Family.METADATA, family.metadata());
        return report;
    }

    static Map<String, Object> runtimeManifest(ModelFamilyRuntimeManifest manifest) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(RuntimeManifest.FAMILY_ID, manifest.familyId());
        report.put(RuntimeManifest.DISPLAY_NAME, manifest.displayName());
        report.put(RuntimeManifest.MODEL_TYPES, manifest.modelTypes());
        report.put(RuntimeManifest.ARCHITECTURE_CLASS_NAMES, manifest.architectureClassNames());
        report.put(RuntimeManifest.ARCHITECTURE_ADAPTER_IDS, manifest.architectureAdapterIds());
        report.put(RuntimeManifest.TOKENIZER_PROFILE_IDS, manifest.tokenizerProfileIds());
        report.put(RuntimeManifest.TOKENIZER_KINDS, manifest.tokenizerKinds().stream()
                .map(kind -> kind.name().toLowerCase())
                .toList());
        report.put(RuntimeManifest.TOKENIZER_READY, manifest.tokenizerReady());
        report.put(RuntimeManifest.CHAT_TEMPLATE_IDS, manifest.chatTemplateIds());
        report.put(RuntimeManifest.CHAT_TEMPLATE_READY, manifest.chatTemplateReady());
        report.put(RuntimeManifest.BUNDLE_PROFILE, manifest.bundleProfile().key());
        report.put(RuntimeManifest.CAPABILITIES, manifest.capabilities().stream()
                .map(capability -> capability.name().toLowerCase())
                .toList());
        report.put(RuntimeManifest.DIRECT_SAFETENSOR_STATUS, manifest.directSafetensorStatus().label());
        report.put(RuntimeManifest.DIRECT_SAFETENSOR_READY, manifest.directSafetensorReady());
        report.put(RuntimeManifest.DIRECT_SAFETENSOR_REASON, manifest.directSafetensorReason());
        report.put(RuntimeManifest.DIRECT_SAFETENSOR_CAVEATS, manifest.directSafetensorCaveats());
        report.put(RuntimeManifest.UNIFIED_RUNTIME_REQUIREMENTS, manifest.unifiedRuntimeRequirements().stream()
                .map(ModelFamilyBundleInventoryReports::runtimeRequirement)
                .toList());
        report.put(RuntimeManifest.METADATA, manifest.metadata());
        return report;
    }

    private static Map<String, Object> runtimeRequirement(ModelFamilyUnifiedRuntimeRequirement requirement) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(RuntimeRequirement.MODEL_TYPE, requirement.modelType());
        report.put(RuntimeRequirement.REQUIRED_INPUT_MODALITIES, requirement.requiredInputModalities());
        report.put(RuntimeRequirement.PRODUCTION_READY_REQUIRED, requirement.productionReadyRequired());
        report.put(RuntimeRequirement.REASON, requirement.reason());
        report.put(RuntimeRequirement.METADATA, requirement.metadata());
        return report;
    }
}
