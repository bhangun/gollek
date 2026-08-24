/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.DirectArchitecture;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.DirectSafetensorCompatibility;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.Resolution;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.RuntimeCompatibility;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.RuntimeManifest;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.SupportReport;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.Tokenizer;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyDirectSupport;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyResolution;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibility;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeManifest;
import tech.kayys.alkhawarizm.spi.model.ModelFamilySupportReport;
import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Builds reusable model-family resolution reports for CLI, docs, and plugin diagnostics.
 */
public final class ModelFamilyResolutionReports {
    private ModelFamilyResolutionReports() {
    }

    public static Map<String, Object> report(
            ModelFamilyResolution resolution,
            Optional<Path> modelDir,
            ModelFamilyPluginRegistry registry) {
        Optional<Path> effectiveModelDir = modelDir == null ? Optional.empty() : modelDir;
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(Resolution.STATUS, resolution.status().name());
        report.put(Resolution.RESOLVED, resolution.resolved());
        report.put(Resolution.AMBIGUOUS, resolution.ambiguous());
        report.put(Resolution.MODEL_TYPE, resolution.modelType());
        report.put(Resolution.ARCHITECTURE_CLASS_NAME, resolution.architectureClassName());
        report.put(Resolution.FAMILY_IDS, resolution.familyIds());
        report.put(Resolution.SUMMARY, resolution.summary());
        report.put(Resolution.REQUIRES_ATTENTION,
                !problemCodes(resolution, effectiveModelDir, effectiveRegistry).isEmpty());
        report.put(Resolution.PROBLEM_CODES, problemCodes(resolution, effectiveModelDir, effectiveRegistry));
        report.put(Resolution.REMEDIATION_HINTS, remediationHints(resolution, effectiveModelDir, effectiveRegistry));
        report.put(Resolution.SUPPORT_REPORTS, resolution.supportReports().stream()
                .map(ModelFamilyResolutionReports::supportReport)
                .toList());
        report.put(Resolution.RUNTIME_MANIFESTS, resolution.runtimeManifests().stream()
                .map(ModelFamilyResolutionReports::runtimeManifestReport)
                .toList());
        report.put(Resolution.RUNTIME_COMPATIBILITY, runtimeCompatibilityReport(
                resolution,
                effectiveModelDir,
                effectiveRegistry));
        report.put(Resolution.DIRECT_ARCHITECTURE, directArchitectureReport(resolution, effectiveRegistry));
        report.put(Resolution.TOKENIZERS, resolution.tokenizerDescriptors().stream()
                .map(descriptor -> tokenizerReport(descriptor, effectiveModelDir))
                .toList());
        return report;
    }

    public static Map<String, Object> supportReport(ModelFamilySupportReport report) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(SupportReport.ID, report.id());
        out.put(SupportReport.DISPLAY_NAME, report.displayName());
        out.put(SupportReport.BUNDLE_PROFILE, report.bundleProfile().name());
        out.put(SupportReport.CAPABILITIES, report.capabilities().stream().map(Enum::name).toList());
        out.put(SupportReport.ARCHITECTURE_ADAPTER_IDS, report.architectureAdapterIds());
        out.put(SupportReport.TOKENIZER_PROFILE_IDS, report.tokenizerProfileIds());
        out.put(SupportReport.TOKENIZER_KINDS, report.tokenizerKinds().stream().map(Enum::name).toList());
        out.put(SupportReport.DIRECT_SAFETENSOR_STATUS, report.directSafetensorStatus().name());
        out.put(SupportReport.DIRECT_SAFETENSOR_REASON, report.directSafetensorReason());
        out.put(SupportReport.DIRECT_SAFETENSOR_CAVEATS, report.directSafetensorCaveats());
        return out;
    }

    public static Map<String, Object> runtimeManifestReport(ModelFamilyRuntimeManifest manifest) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(RuntimeManifest.FAMILY_ID, manifest.familyId());
        out.put(RuntimeManifest.MODEL_TYPES, manifest.modelTypes());
        out.put(RuntimeManifest.ARCHITECTURE_ADAPTER_IDS, manifest.architectureAdapterIds());
        out.put(RuntimeManifest.TOKENIZER_PROFILE_IDS, manifest.tokenizerProfileIds());
        out.put(RuntimeManifest.TOKENIZER_READY, manifest.tokenizerReady());
        out.put(RuntimeManifest.CHAT_TEMPLATE_IDS, manifest.chatTemplateIds());
        out.put(RuntimeManifest.CHAT_TEMPLATE_READY, manifest.chatTemplateReady());
        out.put(RuntimeManifest.BUNDLE_PROFILE, manifest.bundleProfile().name());
        out.put(RuntimeManifest.DIRECT_SAFETENSOR_STATUS, manifest.directSafetensorStatus().name());
        out.put(RuntimeManifest.DIRECT_SAFETENSOR_READY, manifest.directSafetensorReady());
        return out;
    }

    public static Map<String, Object> runtimeCompatibilityReport(
            ModelFamilyResolution resolution,
            Optional<Path> modelDir,
            ModelFamilyPluginRegistry registry) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(RuntimeCompatibility.DIRECT_SAFETENSOR, runtimeCompatibilityReport(
                directSafetensorCompatibility(resolution, modelDir, registry)));
        return out;
    }

    public static Map<String, Object> runtimeCompatibilityReport(
            ModelFamilyRuntimeCompatibility compatibility) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(DirectSafetensorCompatibility.COMPATIBLE, compatibility.compatible());
        out.put(DirectSafetensorCompatibility.SELECTED_ARCHITECTURE_ADAPTER_ID,
                compatibility.selectedArchitectureAdapterId());
        out.put(DirectSafetensorCompatibility.SELECTED_ARCHITECTURE_ADAPTER_BY,
                compatibility.selectedArchitectureAdapterBy());
        out.put(DirectSafetensorCompatibility.PROBLEM_CODES, compatibility.problemCodes());
        out.put(DirectSafetensorCompatibility.REMEDIATION_HINTS, compatibility.remediationHints());
        return out;
    }

    public static Map<String, Object> tokenizerReport(
            ModelTokenizerDescriptor descriptor,
            Optional<Path> modelDir) {
        Optional<Path> effectiveModelDir = modelDir == null ? Optional.empty() : modelDir;
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(Tokenizer.ID, descriptor.id());
        out.put(Tokenizer.KIND, descriptor.kind().name());
        out.put(Tokenizer.REQUIRED_FILE_GROUPS, descriptor.requiredFileGroups());
        out.put(Tokenizer.OPTIONS, descriptor.options());
        out.put(Tokenizer.FILE_STATUS_AVAILABLE, effectiveModelDir.isPresent());
        out.put(Tokenizer.USABLE, effectiveModelDir
                .map(dir -> descriptor.firstExistingFileGroup(dir).isPresent())
                .orElse(false));
        out.put(Tokenizer.EXISTING_FILE_GROUP, effectiveModelDir
                .flatMap(descriptor::firstExistingFileGroup)
                .map(paths -> paths.stream()
                        .map(path -> effectiveModelDir.orElseThrow().relativize(path).toString())
                        .toList())
                .orElse(List.of()));
        out.put(Tokenizer.MISSING_FILE_GROUPS, effectiveModelDir
                .map(dir -> missingFileGroups(dir, descriptor))
                .orElse(List.of()));
        return out;
    }

    public static List<String> problemCodes(
            ModelFamilyResolution resolution,
            Optional<Path> modelDir,
            ModelFamilyPluginRegistry registry) {
        Optional<Path> effectiveModelDir = modelDir == null ? Optional.empty() : modelDir;
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        LinkedHashSet<String> problemCodes = new LinkedHashSet<>(resolution.problemCodes());
        problemCodes.addAll(directArchitectureProblemCodes(resolution, effectiveRegistry));
        if (effectiveModelDir.isPresent()
                && resolution.resolved()
                && !resolution.tokenizerDescriptors().isEmpty()
                && resolution.tokenizerDescriptors().stream()
                        .noneMatch(descriptor ->
                                descriptor.firstExistingFileGroup(effectiveModelDir.orElseThrow()).isPresent())) {
            problemCodes.add("model_family_tokenizer_files_missing");
        }
        return List.copyOf(problemCodes);
    }

    public static List<String> remediationHints(
            ModelFamilyResolution resolution,
            Optional<Path> modelDir,
            ModelFamilyPluginRegistry registry) {
        Optional<Path> effectiveModelDir = modelDir == null ? Optional.empty() : modelDir;
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        List<String> hints = new ArrayList<>(resolution.remediationHints());
        List<String> architectureProblems = directArchitectureProblemCodes(resolution, effectiveRegistry);
        if (architectureProblems.contains("model_family_architecture_adapters_missing")) {
            hints.add("Publish an architecture adapter from the matched model-family plugin or remove the direct SafeTensor capability until the adapter is ready.");
        }
        if (architectureProblems.contains("model_family_architecture_adapter_unmatched")) {
            hints.add("Update the matched architecture adapter claims for model_type="
                    + resolution.modelType() + ", architecture=" + resolution.architectureClassName() + ".");
        }
        if (problemCodes(resolution, effectiveModelDir, effectiveRegistry)
                .contains("model_family_tokenizer_files_missing")) {
            String requirements = resolution.tokenizerDescriptors().stream()
                    .map(descriptor -> descriptor.id() + " requires " + descriptor.requiredFileGroups())
                    .reduce((left, right) -> left + "; " + right)
                    .orElse("no tokenizer descriptor requirements were available");
            hints.add("Add one required tokenizer file group for the matched family: " + requirements + ".");
        }
        return List.copyOf(hints);
    }

    public static Map<String, Object> directArchitectureReport(
            ModelFamilyResolution resolution,
            ModelFamilyPluginRegistry registry) {
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        List<ModelArchitecture> adapters =
                effectiveRegistry.architectureAdaptersFor(resolution.modelType(), resolution.architectureClassName());
        Optional<ArchitectureSelection> selected = selectArchitectureAdapter(
                adapters,
                resolution.modelType(),
                resolution.architectureClassName());
        Map<String, Object> report = new LinkedHashMap<>();
        report.put(DirectArchitecture.DIRECT_SUPPORT_EXPECTED, directAdapterExpected(resolution));
        report.put(DirectArchitecture.DIRECT_SUPPORT_STATUSES, resolution.supportReports().stream()
                .map(support -> support.directSafetensorStatus().name())
                .distinct()
                .toList());
        report.put(DirectArchitecture.ADAPTER_IDS, adapters.stream()
                .map(ModelFamilyResolutionReports::safeAdapterId)
                .filter(id -> !id.isBlank())
                .toList());
        report.put(DirectArchitecture.SELECTED_ADAPTER_ID, selected.map(ArchitectureSelection::adapterId).orElse(null));
        report.put(DirectArchitecture.SELECTED_BY, selected.map(ArchitectureSelection::selectedBy).orElse(null));
        report.put(DirectArchitecture.PROBLEM_CODES, directArchitectureProblemCodes(resolution, effectiveRegistry));
        return report;
    }

    public static List<String> directArchitectureProblemCodes(
            ModelFamilyResolution resolution,
            ModelFamilyPluginRegistry registry) {
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        if (!directAdapterExpected(resolution)) {
            return List.of();
        }
        List<ModelArchitecture> adapters =
                effectiveRegistry.architectureAdaptersFor(resolution.modelType(), resolution.architectureClassName());
        if (adapters.isEmpty()) {
            return List.of("model_family_architecture_adapters_missing");
        }
        if (selectArchitectureAdapter(adapters, resolution.modelType(), resolution.architectureClassName()).isEmpty()) {
            return List.of("model_family_architecture_adapter_unmatched");
        }
        return List.of();
    }

    public static boolean directAdapterExpected(ModelFamilyResolution resolution) {
        return resolution.supportReports().stream().anyMatch(report ->
                report.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE)
                        || report.directSafetensorStatus() == ModelFamilyDirectSupport.READY
                        || report.directSafetensorStatus() == ModelFamilyDirectSupport.EXPERIMENTAL
                        || report.directSafetensorStatus() == ModelFamilyDirectSupport.DECLARED_NO_ADAPTER);
    }

    public static ModelFamilyRuntimeCompatibility directSafetensorCompatibility(
            ModelFamilyResolution resolution,
            Optional<Path> modelDir,
            ModelFamilyPluginRegistry registry) {
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        Optional<Path> effectiveModelDir = modelDir == null ? Optional.empty() : modelDir;
        return effectiveRegistry.directSafetensorCompatibility(
                resolution,
                effectiveModelDir.orElse(null));
    }

    private static List<List<String>> missingFileGroups(Path modelDir, ModelTokenizerDescriptor descriptor) {
        return descriptor.requiredFileGroups().stream()
                .filter(group -> group.stream().anyMatch(relative -> !Files.exists(modelDir.resolve(relative))))
                .map(List::copyOf)
                .toList();
    }

    private static Optional<ArchitectureSelection> selectArchitectureAdapter(
            List<ModelArchitecture> adapters,
            String modelType,
            String architectureClassName) {
        for (ModelArchitecture adapter : adapters) {
            if (modelType != null && supportedModelTypes(adapter).contains(modelType)) {
                return Optional.of(new ArchitectureSelection(safeAdapterId(adapter), "model_type"));
            }
        }
        for (ModelArchitecture adapter : adapters) {
            if (architectureClassName != null
                    && supportedArchitectureClassNames(adapter).contains(architectureClassName)) {
                return Optional.of(new ArchitectureSelection(safeAdapterId(adapter), "architecture"));
            }
        }
        return Optional.empty();
    }

    private static String safeAdapterId(ModelArchitecture adapter) {
        try {
            String id = adapter.id();
            return id == null ? "" : id.trim();
        } catch (RuntimeException error) {
            return "";
        }
    }

    private static List<String> supportedModelTypes(ModelArchitecture adapter) {
        try {
            List<String> modelTypes = adapter.supportedModelTypes();
            return modelTypes == null ? List.of() : modelTypes;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private static List<String> supportedArchitectureClassNames(ModelArchitecture adapter) {
        try {
            List<String> architectureClassNames = adapter.supportedArchClassNames();
            return architectureClassNames == null ? List.of() : architectureClassNames;
        } catch (RuntimeException error) {
            return List.of();
        }
    }

    private record ArchitectureSelection(String adapterId, String selectedBy) {
    }
}
