/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.core.ExtensionAvailability;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityContractReport;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityContractViolation;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Plugin availability checker for CLI commands.
 * Checks both high-level providers (LLM providers like GGUF, Gemini)
 * and low-level runner plugins (ONNX Runtime, SafeTensor, etc.).
 */
@ApplicationScoped
public class PluginAvailabilityChecker {

    private static final Logger LOG = Logger.getLogger(PluginAvailabilityChecker.class);

    public record ModelFamilyBundleAvailability(
            boolean present,
            boolean detached,
            boolean healthy,
            String status,
            int selectedFamilyCount,
            int discoveredSelectedFamilyCount,
            int missingSelectedFamilyCount,
            int omittedFamilyCount,
            String policyStatus,
            int policyViolationCount,
            String productionSafetyStatus,
            boolean productionSafetyPassed,
            List<String> productionPendingTokenizerFamilies,
            String catalogReadinessStatus,
            boolean catalogReadinessPassed,
            int productionReadinessPendingCount,
            int directSafetensorPendingCount,
            List<String> productionReadinessPendingFamilies,
            List<String> directSafetensorPendingFamilies,
            String fixtureStatus,
            Boolean fixturePassed,
            int fixtureMissingRequiredCount,
            int fixtureProblemFamilyCount,
            String presetConformanceStatus,
            List<String> problems,
            List<String> remediationHints,
            List<String> missingSelectedFamilies,
            List<String> omittedFamilies,
            List<String> fixtureMissingRequiredFamilies,
            List<String> fixtureProblemFamilies) {

        public ModelFamilyBundleAvailability {
            status = status == null || status.isBlank() ? "unknown" : status;
            policyStatus = policyStatus == null || policyStatus.isBlank() ? "unknown" : policyStatus;
            productionSafetyStatus = productionSafetyStatus == null || productionSafetyStatus.isBlank()
                    ? "unknown"
                    : productionSafetyStatus;
            catalogReadinessStatus = catalogReadinessStatus == null || catalogReadinessStatus.isBlank()
                    ? "unknown"
                    : catalogReadinessStatus;
            productionReadinessPendingCount = Math.max(0, productionReadinessPendingCount);
            directSafetensorPendingCount = Math.max(0, directSafetensorPendingCount);
            fixtureStatus = fixtureStatus == null || fixtureStatus.isBlank() ? "unknown" : fixtureStatus;
            fixtureMissingRequiredCount = Math.max(0, fixtureMissingRequiredCount);
            fixtureProblemFamilyCount = Math.max(0, fixtureProblemFamilyCount);
            presetConformanceStatus = presetConformanceStatus == null || presetConformanceStatus.isBlank()
                    ? "none"
                    : presetConformanceStatus;
            problems = List.copyOf(problems == null ? List.of() : problems);
            remediationHints = List.copyOf(remediationHints == null ? List.of() : remediationHints);
            productionPendingTokenizerFamilies = List.copyOf(productionPendingTokenizerFamilies == null
                    ? List.of()
                    : productionPendingTokenizerFamilies);
            productionReadinessPendingFamilies = List.copyOf(productionReadinessPendingFamilies == null
                    ? List.of()
                    : productionReadinessPendingFamilies);
            directSafetensorPendingFamilies = List.copyOf(directSafetensorPendingFamilies == null
                    ? List.of()
                    : directSafetensorPendingFamilies);
            missingSelectedFamilies = List.copyOf(missingSelectedFamilies == null
                    ? List.of()
                    : missingSelectedFamilies);
            omittedFamilies = List.copyOf(omittedFamilies == null ? List.of() : omittedFamilies);
            fixtureMissingRequiredFamilies = List.copyOf(fixtureMissingRequiredFamilies == null
                    ? List.of()
                    : fixtureMissingRequiredFamilies);
            fixtureProblemFamilies = List.copyOf(fixtureProblemFamilies == null ? List.of() : fixtureProblemFamilies);
        }

        public String compactSummary() {
            return "%s(selected=%d, discovered=%d, missing=%d, omitted=%d, policy=%s, production=%s, catalog=%s, "
                    + "fixture=%s, presetConformance=%s)"
                    .formatted(
                            status,
                            selectedFamilyCount,
                            discoveredSelectedFamilyCount,
                            missingSelectedFamilyCount,
                            omittedFamilyCount,
                            policyStatus,
                            productionSafetyStatus,
                            catalogReadinessStatus,
                            fixtureStatus,
                            presetConformanceStatus);
        }
    }

    
    @Inject
    Instance<RunnerPlugin> runnerPlugins;

    @Inject
    Instance<ModelFamilyPlugin> modelFamilyPlugins;

    /**
     * Check if any LLM providers are registered.
     */
    public boolean hasProviders() {
        try {
            var providers = java.util.List.of();
            return providers != null && !providers.isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking providers: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Check if a specific provider is available by ID.
     */
    public boolean hasProvider(String providerId) {
        try {
            return false;
        } catch (Exception e) {
            LOG.debugf("Error checking provider %s: %s", providerId, e.getMessage());
            return false;
        }
    }

    /**
     * Check if any runner plugins are discovered.
     * Runner plugins are low-level execution backends (ONNX, GGUF native, etc.)
     */
    public boolean hasRunnerPlugins() {
        try {
            return !getRunnerPluginIds().isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking runner plugins: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Get a list of all discovered runner plugin IDs.
     */
    public List<String> getRunnerPluginIds() {
        Set<String> ids = new LinkedHashSet<>();
        try {
            if (runnerPlugins != null && !runnerPlugins.isUnsatisfied()) {
                ids.addAll(runnerPlugins.stream()
                        .map(RunnerPlugin::id)
                        .collect(Collectors.toList()));
            }
        } catch (Exception e) {
            LOG.debugf("Error listing CDI runner plugins: %s", e.getMessage());
        }

        try {
            ids.addAll(RunnerPluginManager.getInstance().getAvailablePlugins().stream()
                    .map(RunnerPlugin::id)
                    .collect(Collectors.toList()));
        } catch (Exception e) {
            LOG.debugf("Error listing ServiceLoader runner plugins: %s", e.getMessage());
        }

        return List.copyOf(ids);
    }

    /**
     * Check if any model-family plugins are discovered.
     * Model-family plugins describe reusable architecture/tokenizer metadata.
     */
    public boolean hasModelFamilyPlugins() {
        try {
            return !getModelFamilyPluginIds().isEmpty();
        } catch (Exception e) {
            LOG.debugf("Error checking model-family plugins: %s", e.getMessage());
            return false;
        }
    }

    /**
     * Get a list of all discovered model-family plugin IDs.
     */
    public List<String> getModelFamilyPluginIds() {
        return getModelFamilyPluginIds(null);
    }

    /**
     * Get all discovered model-family plugin IDs, including families exposed by
     * an explicit plugin classloader.
     */
    public List<String> getModelFamilyPluginIds(ClassLoader pluginClassLoader) {
        return modelFamilyPluginIds(getModelFamilyPluginRegistry(pluginClassLoader));
    }

    public ModelFamilyPluginRegistry getModelFamilyPluginRegistry(ClassLoader pluginClassLoader) {
        ModelFamilyPluginRegistry packagedRegistry = ModelFamilyPluginRegistry.global();
        try {
            if (modelFamilyPlugins != null && !modelFamilyPlugins.isUnsatisfied()) {
                modelFamilyPlugins.forEach(plugin -> {
                    packagedRegistry.register(plugin);
                });
            }
        } catch (Exception e) {
            LOG.debugf("Error listing CDI model-family plugins: %s", e.getMessage());
        }

        try {
            packagedRegistry.discoverServiceLoaderPlugins();
        } catch (Exception e) {
            LOG.debugf("Error listing ServiceLoader model-family plugins: %s", e.getMessage());
        }

        if (pluginClassLoader == null) {
            return packagedRegistry;
        }

        ModelFamilyPluginRegistry scopedRegistry = packagedRegistry.snapshot();
        try {
            scopedRegistry.discoverServiceLoaderPlugins(pluginClassLoader);
        } catch (Exception e) {
            LOG.debugf("Error listing external ServiceLoader model-family plugins: %s", e.getMessage());
        }
        return scopedRegistry;
    }

    /**
     * Get model-family IDs with direct SafeTensor readiness for diagnostics.
     */
    public List<String> getModelFamilySupportSummaries() {
        try {
            return getModelFamilyPluginRegistry(null).supportReports().stream()
                    .map(report -> report.id() + "[" + report.bundleProfile().key() + "]("
                            + report.shortDirectSafetensorSummary() + ")")
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family support summaries: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get compact capability-matrix rows for packaged model-family plugins.
     */
    public List<String> getModelFamilyCapabilityMatrixSummaries() {
        try {
            return getModelFamilyPluginRegistry(null).capabilityMatrix().stream()
                    .map(entry -> entry.compactSummary())
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family capability matrix summaries: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get duplicate model-type claims across packaged model-family plugins.
     */
    public List<String> getModelFamilyClaimConflicts() {
        try {
            return getModelFamilyPluginRegistry(null).modelTypeConflicts().stream()
                    .map(conflict -> conflict.summary())
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family claim conflicts: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get model-family plugin contract violations across packaged families.
     */
    public List<String> getModelFamilyContractViolations() {
        try {
            return getModelFamilyPluginRegistry(null).contractViolations().stream()
                    .map(violation -> violation.summary())
                    .collect(Collectors.toList());
        } catch (Exception e) {
            LOG.debugf("Error listing model-family contract violations: %s", e.getMessage());
            return List.of();
        }
    }

    /**
     * Get the model-family bundle manifest packaged into this CLI build.
     */
    public ModelFamilyBundleManifest getModelFamilyBundleManifest() {
        return ModelFamilyBundleManifest.load();
    }

    /**
     * Get build selector aliases packaged into this CLI build.
     */
    public List<String> getModelFamilyBundleAliasSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.bundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAlias::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get build bundle presets packaged into this CLI build.
     */
    public List<String> getModelFamilyBundlePresetSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.bundlePresets().stream()
                .map(ModelFamilyBundleManifest.BundlePreset::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get the active build preset packaged into this CLI build, including policy status.
     */
    public String getActiveModelFamilyBundlePresetSummary() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present() || !manifest.hasBundlePreset()) {
            return "";
        }
        return manifest.activeBundlePreset()
                .map(ModelFamilyBundleManifest.BundlePreset::compactSummary)
                .orElse(manifest.bundlePreset() + "(metadata=missing)");
    }

    /**
     * Get selector aliases fully covered by this CLI build.
     */
    public List<String> getCompleteModelFamilyBundleAliasSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.completeBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get selector aliases partially covered by this CLI build.
     */
    public List<String> getPartialModelFamilyBundleAliasSummaries() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.partialBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::compactSummary)
                .collect(Collectors.toList());
    }

    /**
     * Get build-selected model families that were not discovered at runtime.
     */
    public List<String> getMissingBundledModelFamilies() {
        ModelFamilyBundleManifest manifest = getModelFamilyBundleManifest();
        if (!manifest.present()) {
            return List.of();
        }
        return manifest.missingDiscovered(new LinkedHashSet<>(getModelFamilyPluginIds()));
    }

    /**
     * Get the packaged model-family bundle availability relative to runtime discovery.
     */
    public ModelFamilyBundleAvailability getModelFamilyBundleAvailability() {
        return modelFamilyBundleAvailability(
                getModelFamilyBundleManifest(),
                new LinkedHashSet<>(getModelFamilyPluginIds()));
    }

    /**
     * Get the combined model-family bundle release gate.
     */
    public ModelFamilyBundleGate getModelFamilyBundleGate() {
        return modelFamilyBundleGate(
                getModelFamilyBundleManifest(),
                new LinkedHashSet<>(getModelFamilyPluginIds()));
    }

    /**
     * Audio-specific compatibility view for the optional Suling extension.
     */
    public record AudioExtensionAvailability(
            ExtensionAvailability extensionAvailability,
            boolean flacAvailable,
            String flacVersion,
            boolean mp3EncodingAvailable) {

        public AudioExtensionAvailability {
            extensionAvailability = Objects.requireNonNull(extensionAvailability, "extensionAvailability");
            flacVersion = flacVersion == null || flacVersion.isBlank() ? "unavailable" : flacVersion;
        }

        public static AudioExtensionAvailability fromExtensionAvailability(ExtensionAvailability availability) {
            return new AudioExtensionAvailability(
                    availability,
                    availability.attributeBoolean("flacAvailable"),
                    availability.attribute("flacVersion", "unavailable"),
                    availability.attributeBoolean("mp3EncodingAvailable"));
        }

        public String id() {
            return extensionAvailability.id();
        }

        public String name() {
            return extensionAvailability.name();
        }

        public String kind() {
            return extensionAvailability.kind();
        }

        public boolean attached() {
            return extensionAvailability.attached();
        }

        public boolean detached() {
            return extensionAvailability.detached();
        }

        public boolean healthy() {
            return extensionAvailability.healthy();
        }

        public boolean productionReady() {
            return extensionAvailability.productionReady();
        }

        public String status() {
            return extensionAvailability.status();
        }

        public List<String> capabilities() {
            return extensionAvailability.capabilities();
        }

        public List<String> formats() {
            return extensionAvailability.formats();
        }

        public Map<String, String> attributes() {
            return extensionAvailability.attributes();
        }

        public String diagnostics() {
            return extensionAvailability.diagnostics();
        }

        public List<String> remediationHints() {
            return extensionAvailability.remediationHints();
        }
    }

    /**
     * Get the generic plugin-core availability report for the Suling audio extension.
     */
    public ExtensionAvailability sulingExtensionAvailability() {
        return getExtensionAvailabilityReports().stream()
                .filter(report -> SulingAudioExtensionAvailabilityProvider.ID.equals(report.id()))
                .findFirst()
                .orElseGet(() -> new SulingAudioExtensionAvailabilityProvider().availability());
    }

    /**
     * Get the audio-specific compatibility report for the Suling audio extension.
     */
    public AudioExtensionAvailability sulingAudioAvailability() {
        return AudioExtensionAvailability.fromExtensionAvailability(sulingExtensionAvailability());
    }

    /**
     * Get all extension availability reports known to plugin-core plus CLI built-ins.
     */
    public List<ExtensionAvailability> getExtensionAvailabilityReports() {
        return getGlobalExtensionAvailabilityReports();
    }

    /**
     * Get extension availability reports using an explicit extension classloader.
     */
    public List<ExtensionAvailability> getExtensionAvailabilityReports(ClassLoader classLoader) {
        return getGlobalExtensionAvailabilityReports(classLoader);
    }

    /**
     * Get extension availability provider contract violations for discovered extensions.
     */
    public List<ExtensionAvailabilityContractViolation> getExtensionAvailabilityContractViolations() {
        return getGlobalExtensionAvailabilityContractViolations();
    }

    /**
     * Get aggregate extension provider contract status for discovered extensions.
     */
    public ExtensionAvailabilityContractReport getExtensionAvailabilityContractReport() {
        return getGlobalExtensionAvailabilityContractReport();
    }

    /**
     * Get the combined extension release gate for discovered extensions.
     */
    public ExtensionAvailabilityGate getExtensionAvailabilityGate() {
        return getGlobalExtensionAvailabilityGate();
    }

    public static List<ExtensionAvailability> getGlobalExtensionAvailabilityReports() {
        return getGlobalExtensionAvailabilityReports(null);
    }

    public static List<ExtensionAvailability> getGlobalExtensionAvailabilityReports(ClassLoader classLoader) {
        ExtensionAvailabilityRegistry registry = getExtensionAvailabilityRegistry(classLoader);
        return withDiscoveryClassLoader(classLoader, registry::availabilityReports);
    }

    public static List<ExtensionAvailabilityContractViolation> getGlobalExtensionAvailabilityContractViolations() {
        return getGlobalExtensionAvailabilityContractViolations(null);
    }

    public static List<ExtensionAvailabilityContractViolation> getGlobalExtensionAvailabilityContractViolations(
            ClassLoader classLoader) {
        ExtensionAvailabilityRegistry registry = getExtensionAvailabilityRegistry(classLoader);
        return withDiscoveryClassLoader(classLoader, registry::contractViolations);
    }

    public static ExtensionAvailabilityContractReport getGlobalExtensionAvailabilityContractReport() {
        return getGlobalExtensionAvailabilityContractReport(null);
    }

    public static ExtensionAvailabilityContractReport getGlobalExtensionAvailabilityContractReport(
            ClassLoader classLoader) {
        ExtensionAvailabilityRegistry registry = getExtensionAvailabilityRegistry(classLoader);
        return withDiscoveryClassLoader(classLoader, registry::contractReport);
    }

    public static ExtensionAvailabilityGate getGlobalExtensionAvailabilityGate() {
        return getGlobalExtensionAvailabilityGate(null);
    }

    public static ExtensionAvailabilityGate getGlobalExtensionAvailabilityGate(ClassLoader classLoader) {
        ExtensionAvailabilityRegistry registry = getExtensionAvailabilityRegistry(classLoader);
        return withDiscoveryClassLoader(classLoader, () -> {
            ExtensionAvailabilityPolicy.Result policy = ExtensionAvailabilityPolicy.fromRuntimeConfiguration()
                    .evaluate(registry.availabilityReports());
            return ExtensionAvailabilityGate.evaluate(policy, registry.contractReport());
        });
    }

    private static <T> T withDiscoveryClassLoader(ClassLoader classLoader, Supplier<T> action) {
        if (classLoader == null) {
            return action.get();
        }
        Thread thread = Thread.currentThread();
        ClassLoader previous = thread.getContextClassLoader();
        try {
            thread.setContextClassLoader(classLoader);
            return action.get();
        } finally {
            thread.setContextClassLoader(previous);
        }
    }

    public static List<String> getGlobalModelFamilyPluginIds() {
        return getGlobalModelFamilyPluginIds(null);
    }

    public static List<String> getGlobalModelFamilyPluginIds(ClassLoader classLoader) {
        return new PluginAvailabilityChecker().getModelFamilyPluginIds(classLoader);
    }

    static List<String> modelFamilyPluginIds(ModelFamilyPluginRegistry registry) {
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        return effectiveRegistry.all().stream()
                .map(PluginAvailabilityChecker::modelFamilyPluginId)
                .filter(id -> !id.isBlank())
                .collect(Collectors.toList());
    }

    private static String modelFamilyPluginId(ModelFamilyPlugin plugin) {
        try {
            return plugin.descriptor().id();
        } catch (Exception error) {
            try {
                return plugin.id();
            } catch (Exception ignored) {
                return "";
            }
        }
    }

    public static ModelFamilyBundleAvailability getGlobalModelFamilyBundleAvailability() {
        return getGlobalModelFamilyBundleAvailability(null);
    }

    public static ModelFamilyBundleAvailability getGlobalModelFamilyBundleAvailability(ClassLoader classLoader) {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyPluginRegistry registry = new PluginAvailabilityChecker()
                .getModelFamilyPluginRegistry(classLoader);
        return modelFamilyBundleAvailability(
                manifest,
                new LinkedHashSet<>(modelFamilyPluginIds(registry)));
    }

    public static ModelFamilyBundleGate getGlobalModelFamilyBundleGate() {
        return getGlobalModelFamilyBundleGate(null);
    }

    public static ModelFamilyBundleGate getGlobalModelFamilyBundleGate(ClassLoader classLoader) {
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyPluginRegistry registry = new PluginAvailabilityChecker()
                .getModelFamilyPluginRegistry(classLoader);
        return modelFamilyBundleGate(
                manifest,
                new LinkedHashSet<>(modelFamilyPluginIds(registry)),
                registry);
    }

    public static ExtensionAvailabilityRegistry getExtensionAvailabilityRegistry(ClassLoader classLoader) {
        ExtensionAvailabilityRegistry packagedRegistry = ExtensionAvailabilityRegistry.global();
        try {
            packagedRegistry.discoverServiceLoaderProviders();
        } catch (Exception e) {
            LOG.debugf("Error discovering extension availability providers: %s", e.getMessage());
        }
        ensureBuiltInSulingAvailabilityProvider(packagedRegistry);

        if (classLoader == null) {
            return packagedRegistry;
        }

        ExtensionAvailabilityRegistry scopedRegistry = packagedRegistry.snapshot();
        try {
            scopedRegistry.discoverServiceLoaderProviders(classLoader);
        } catch (Exception e) {
            LOG.debugf("Error discovering external extension availability providers: %s", e.getMessage());
        }
        ensureBuiltInSulingAvailabilityProvider(scopedRegistry);
        return scopedRegistry;
    }

    private static void ensureBuiltInSulingAvailabilityProvider(ExtensionAvailabilityRegistry registry) {
        boolean hasSulingProvider = registry.providers().stream()
                .anyMatch(provider -> SulingAudioExtensionAvailabilityProvider.ID.equals(provider.extensionId()));
        if (!hasSulingProvider) {
            registry.register(new SulingAudioExtensionAvailabilityProvider());
        }
    }

    /**
     * Build a reusable availability summary for model-family bundle diagnostics.
     */
    public static ModelFamilyBundleAvailability modelFamilyBundleAvailability(
            ModelFamilyBundleManifest manifest,
            Set<String> discoveredFamilyIds) {
        if (manifest == null || !manifest.present()) {
            return new ModelFamilyBundleAvailability(
                    false,
                    false,
                    false,
                    "manifest_missing",
                    0,
                    0,
                    0,
                    0,
                    "unknown",
                    0,
                    "unknown",
                    false,
                    List.of(),
                    "unknown",
                    false,
                    0,
                    0,
                    List.of(),
                    List.of(),
                    "unknown",
                    null,
                    0,
                    0,
                    "none",
                    List.of("model-family bundle manifest is missing"),
                    List.of("Run 'gollek modules --all' in a packaged CLI, or rebuild the CLI so "
                            + ModelFamilyBundleManifest.RESOURCE_PATH + " is generated."),
                    List.of(),
                    List.of(),
                    List.of(),
                    List.of());
        }

        List<String> missing = manifest.missingDiscovered(discoveredFamilyIds);
        List<String> omitted = manifest.omittedFamilies();
        List<String> problems = new ArrayList<>();
        List<String> hints = new ArrayList<>();
        List<String> countProblems = manifest.countConsistencyProblems();
        int selectedCount = manifest.families().size();
        int discoveredSelectedCount = Math.max(0, selectedCount - missing.size());
        boolean policyFailed = manifest.bundlePolicy().statusKnown()
                && !Boolean.TRUE.equals(manifest.bundlePolicy().passed());
        boolean productionSafetyFailed = !manifest.productionSafetyPassed();
        boolean catalogReadinessFailed = manifest.catalogReadinessStatusKnown()
                && !manifest.catalogReadinessPassed();
        ModelFamilyBundleManifest.FixtureStatus fixtureStatus = manifest.fixtureStatus();
        boolean fixtureFailed = fixtureStatus.statusKnown() && !Boolean.TRUE.equals(fixtureStatus.passed());
        ModelFamilyBundleManifest.BundlePresetConformance conformance = manifest.activeBundlePresetConformance();
        boolean presetDrifted = conformance.hasPreset()
                && (!conformance.presetMetadataPresent() || "drifted".equals(conformance.statusLabel()));

        if (!countProblems.isEmpty()) {
            problems.add("model-family bundle manifest counts are inconsistent: "
                    + String.join(", ", countProblems));
            hints.add("Regenerate the packaged model-family bundle manifest with "
                    + ":ui:gollek-cli:generateModelFamilyBundleManifest.");
        }
        if (manifest.detached()) {
            hints.add("Attach model-family plugins by rebuilding with -Pgollek.modelFamilies=<selector> "
                    + "or by using --plugin-dir, --plugin-classpath, or GOLLEK_PLUGIN_DIRS when this CLI "
                    + "needs local tokenizer or architecture metadata.");
        } else {
            if (!missing.isEmpty()) {
                problems.add("selected model-family plugins were not discovered: " + String.join(", ", missing));
                hints.add("Ensure the selected model-family plugin artifacts are packaged and visible to CDI or "
                        + "ServiceLoader: " + String.join(", ", missing)
                        + ". For detachable local model farms, attach them with --plugin-dir, --plugin-classpath, "
                        + "or GOLLEK_PLUGIN_DIRS.");
            }
            if (policyFailed) {
                problems.add("model-family bundle policy failed with "
                        + manifest.bundlePolicy().violationCount() + " violation(s)");
                hints.add("Adjust -Pgollek.requiredModelFamilies, -Pgollek.forbiddenModelFamilies, "
                        + "alias policy flags, or choose a bundle preset whose policy matches this build.");
            }
            if (productionSafetyFailed) {
                problems.add("model-family production safety failed: "
                        + manifest.displayProductionSafetyStatus());
                hints.add("Keep pending-tokenizer families out of prod_* model-family bundle presets, "
                        + "or finish the tokenizer adapter metadata before using this build for production inference.");
            }
            if (catalogReadinessFailed) {
                problems.add("model-family catalog readiness failed: "
                        + manifest.displayCatalogReadinessStatus());
                hints.add("Run :ui:gollek-cli:validateModelFamilyModuleCatalog and fix production or direct "
                        + "SafeTensor readiness blockers before packaging this build for production inference.");
            }
            if (fixtureFailed) {
                problems.add("model-family fixture gate failed: " + fixtureStatus.compactStatus());
                hints.add("Run :ui:gollek-cli:validateModelFamilyFixtures, repair missing/problem fixture families, "
                        + "then regenerate the bundle manifest and fixture fingerprint lock if intentional.");
            }
            if (presetDrifted) {
                problems.add("active model-family bundle preset conformance is " + conformance.statusLabel());
                hints.add("Rebuild with the reviewed -Pgollek.modelFamilyBundlePreset="
                        + conformance.presetId()
                        + " without selector/policy overrides, or intentionally update the preset and bundle lock.");
            }
        }

        String status;
        boolean healthy;
        if (!countProblems.isEmpty()) {
            status = "manifest_inconsistent";
            healthy = false;
        } else if (manifest.detached()) {
            status = "detached";
            healthy = true;
        } else if (!missing.isEmpty()) {
            status = "missing_plugins";
            healthy = false;
        } else if (policyFailed) {
            status = "policy_failed";
            healthy = false;
        } else if (productionSafetyFailed) {
            status = "production_safety_failed";
            healthy = false;
        } else if (catalogReadinessFailed) {
            status = "catalog_readiness_failed";
            healthy = false;
        } else if (fixtureFailed) {
            status = "fixture_failed";
            healthy = false;
        } else if (presetDrifted) {
            status = "preset_drifted";
            healthy = false;
        } else {
            status = "ready";
            healthy = true;
        }

        return new ModelFamilyBundleAvailability(
                true,
                manifest.detached(),
                healthy,
                status,
                selectedCount,
                discoveredSelectedCount,
                missing.size(),
                omitted.size(),
                manifest.bundlePolicy().statusLabel(),
                manifest.bundlePolicy().violationCount(),
                manifest.productionSafetyStatusLabel(),
                manifest.productionSafetyPassed(),
                manifest.selectedTokenizerMetadataPendingFamilies(),
                manifest.catalogReadinessStatusLabel(),
                manifest.catalogReadinessPassed(),
                manifest.productionReadinessPendingCount(),
                manifest.directSafetensorPendingCount(),
                manifest.productionReadinessPendingFamilies(),
                manifest.directSafetensorPendingFamilies(),
                fixtureStatus.statusLabel(),
                fixtureStatus.passed(),
                fixtureStatus.missingRequiredCount(),
                fixtureStatus.problemFamilyCount(),
                conformance.statusLabel(),
                problems,
                hints,
                missing,
                omitted,
                fixtureStatus.missingRequiredFamilies(),
                fixtureStatus.problemFamilies());
    }

    public static ModelFamilyBundleGate modelFamilyBundleGate(
            ModelFamilyBundleManifest manifest,
            Set<String> discoveredFamilyIds) {
        return modelFamilyBundleGate(manifest, discoveredFamilyIds, ModelFamilyPluginRegistry.global());
    }

    public static ModelFamilyBundleGate modelFamilyBundleGate(
            ModelFamilyBundleManifest manifest,
            Set<String> discoveredFamilyIds,
            ModelFamilyPluginRegistry registry) {
        ModelFamilyPluginRegistry effectiveRegistry = registry == null
                ? ModelFamilyPluginRegistry.global()
                : registry;
        return ModelFamilyBundleGate.evaluate(
                modelFamilyBundleAvailability(manifest, discoveredFamilyIds),
                effectiveRegistry.contractViolations(),
                effectiveRegistry.directSafetensorCompatibilitySummaryForFamilies(
                        manifest == null ? List.of() : manifest.families()),
                manifest != null && manifest.requiresDirectSafetensorRuntime());
    }

    /**
     * Get an error message when no plugins are available at all.
     */
    public String getNoPluginsError() {
        StringBuilder sb = new StringBuilder("""
            Error: No inference providers or runner plugins are available.
            
            This typically means no runtime extensions are packaged in this CLI build.
            
            To fix this, ensure at least one of these dependencies is on the classpath:
              - gollek-ext-runner-gguf       (GGUF/llama.cpp runtime)
              - gollek-runner-onnx           (ONNX Runtime)
              - gollek-ext-runner-safetensor  (SafeTensor runtime)
              - gollek-ext-cloud-gemini      (Gemini cloud provider)
              - gollek-ext-cloud-cerebras    (Cerebras cloud provider)
            
            Run 'gollek modules --all' for packaged runtime details.
            """);
        sb.append(modelFamilyBundleDiagnostics());
        return sb.toString();
    }

    /**
     * Get an error message when a specific provider is not found.
     */
    public String getProviderNotFoundError(String providerId) {
        StringBuilder sb = new StringBuilder();
        sb.append("Error: Provider '").append(providerId).append("' is not available.\n\n");

        try {
            var available = java.util.List.of();
            if (available != null && !available.isEmpty()) {
                sb.append("Available providers:\n");
                for (var p : available) {
                    sb.append("  - ").append("").append("\n");
                }
            } else {
                sb.append("No providers are currently available.\n");
            }
        } catch (Exception e) {
            sb.append("Could not list available providers.\n");
        }

        List<String> runners = getRunnerPluginIds();
        if (!runners.isEmpty()) {
            sb.append("\nAvailable runner plugins: ").append(String.join(", ", runners)).append("\n");
        }

        List<String> families = getModelFamilySupportSummaries();
        if (!families.isEmpty()) {
            sb.append("Available model-family plugins: ").append(String.join(", ", families)).append("\n");
        }

        sb.append(modelFamilyBundleDiagnostics());

        List<String> conflicts = getModelFamilyClaimConflicts();
        if (!conflicts.isEmpty()) {
            sb.append("Model-family claim conflicts: ").append(String.join("; ", conflicts)).append("\n");
        }

        sb.append("\nRun 'gollek modules --all' for packaged runtime details.\n");
        return sb.toString();
    }

    private String modelFamilyBundleDiagnostics() {
        return ModelFamilyBundleDiagnosticsRenderer.render(
                getModelFamilyBundleManifest(),
                new LinkedHashSet<>(getModelFamilyPluginIds()));
    }
}
