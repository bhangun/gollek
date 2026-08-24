package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.cli.util.RouteBenchmarkCacheReportContract;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.cli.util.ExtensionAvailabilityGate;
import tech.kayys.gollek.cli.util.ExtensionAvailabilityPolicy;
import tech.kayys.gollek.cli.util.ExternalPluginClasspath;
import tech.kayys.gollek.cli.util.ExternalPluginClasspathScope;
import tech.kayys.gollek.cli.util.ModelFamilyBundleGate;
import tech.kayys.gollek.cli.util.ModelFamilyBundleManifest;
import tech.kayys.gollek.cli.util.ModelFamilyContractViolationReports;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportContract;
import tech.kayys.gollek.cli.util.PluginAvailabilityChecker;
import tech.kayys.gollek.cli.util.PluginDirectoryReadinessReports;
import tech.kayys.gollek.cli.util.PluginGateViolationReports;
import tech.kayys.gollek.cli.util.PluginGates;
import tech.kayys.gollek.cli.util.RouteBenchmarkCacheReportContract;
import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticContract;
import tech.kayys.gollek.sdk.route.RouteReportPayloadContract;
import tech.kayys.gollek.sdk.route.RunnerRouteReportContract;
import tech.kayys.gollek.cli.util.SulingAudioExtensionAvailabilityProvider;
import tech.kayys.gollek.cli.util.UnifiedRuntimeRequirementCompatibility;
import tech.kayys.gollek.cli.util.UnifiedRuntimeRequirementRecommendations;
import tech.kayys.gollek.cli.util.UnifiedRuntimeRequirementReportFields;
import tech.kayys.gollek.cli.util.UnifiedRuntimeRequirementReports;
import tech.kayys.gollek.cli.util.UnifiedRuntimeRequirementResolver;
import tech.kayys.gollek.plugin.core.ExtensionAvailability;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityContractReport;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityContractViolation;
import tech.kayys.gollek.plugin.core.ExtensionAvailabilityRegistry;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;
import tech.kayys.gollek.plugin.kernel.KernelPlatformDetector;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapabilityMatrixEntry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyContractViolation;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibility;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibilitySummary;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeManifest;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyUnifiedRuntimeRequirement;
import tech.kayys.alkhawarizm.spi.model.ModelFamilySupportReport;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeManifest;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeManifestViolation;
import tech.kayys.gollek.spi.multimodal.UnifiedRuntimeRegistry;


import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Predicate;

/**
 * Show built-in modules, runner plugins, and kernel platform info.
 * Provides a complete view of the multilevel plugin system.
 */
@Dependent
@Unremovable
@Command(
        name = "modules",
        aliases = { "components", "runtime-extensions" },
        description = "Show packaged runtime modules, built-in plugins, and kernel info")
public class ExtensionsCommand implements Runnable {
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    private static final int JSON_SCHEMA_VERSION = 1;

    private record ExtensionDef(String type, String name, String profile, String markerClass, String providerId) {
    }

    private record PluginDoctorSnapshot(
            ModelFamilyBundleManifest manifest,
            ModelFamilyPluginRegistry modelFamilyRegistry,
            Map<String, ModelFamilyPlugin> modelFamilies,
            ExtensionAvailabilityRegistry extensionRegistry,
            List<ExtensionAvailability> extensions,
            ExtensionAvailabilityGate extensionGate,
            ModelFamilyBundleGate modelFamilyGate,
            PluginGates pluginGates,
            PluginAvailabilityChecker.ModelFamilyBundleAvailability bundleAvailability,
            ModelFamilyRuntimeCompatibilitySummary directSafetensorSummary,
            Map<String, Object> routeBenchmarkCache,
            UnifiedRuntimeRegistry unifiedRuntimeRegistry,
            List<UnifiedRuntimeRequirementCompatibility> unifiedRuntimeRequirementCompatibility,
            ExternalPluginClasspath.PluginDirectoryInspection pluginDirectoryInspection) {
    }

    private static final List<ExtensionDef> EXTENSIONS = List.of(
            // Local runtimes
            new ExtensionDef("runtime", "GGUF", "base", "tech.kayys.gollek.inference.llamacpp.LlamaCppProvider", "gguf"),
            new ExtensionDef("runtime", "ONNX", "base", "tech.kayys.gollek.onnx.runner.OnnxRuntimeRunner", "onnx"),
            new ExtensionDef("runtime", "SafeTensor", "base",
                    "tech.kayys.gollek.safetensor.engine.warmup.SafetensorProvider", "safetensor"),
            new ExtensionDef("runtime", "LibTorch", "experimental",
                    "tech.kayys.gollek.inference.libtorch.LibTorchProvider", "libtorch"),
            new ExtensionDef("runtime", "LiteRT", "optional",
                    "tech.kayys.gollek.provider.litert.LiteRTProvider", "litert"),
            new ExtensionDef("runtime", "TensorRT", "optional",
                    "tech.kayys.gollek.inference.tensorrt.TensorRTProvider", "tensorrt"),

            // Cloud providers
            new ExtensionDef("cloud", "Gemini", "ext-cloud-gemini",
                    "tech.kayys.gollek.provider.gemini.GeminiProvider", "gemini"),
            new ExtensionDef("cloud", "Cerebras", "ext-cloud-cerebras",
                    "tech.kayys.gollek.provider.cerebras.CerebrasProvider", "cerebras"),
            new ExtensionDef("cloud", "Mistral", "ext-cloud-mistral",
                    "tech.kayys.gollek.provider.mistral.MistralProvider", "mistral"),
            new ExtensionDef("cloud", "OpenAI", "optional",
                    "tech.kayys.gollek.provider.openai.OpenAiProvider", "openai"),
            new ExtensionDef("cloud", "Anthropic", "optional",
                    "tech.kayys.gollek.provider.anthropic.AnthropicProvider", "anthropic"),

            // Tool/integration providers
            new ExtensionDef("tool", "MCP", "base",
                    "tech.kayys.gollek.provider.mcp.McpProvider", "mcp"));

    @Inject
    GollekSdk sdk;

    @Inject
    Instance<RunnerPlugin> runnerPluginInstances;

    @Inject
    Instance<ModelFamilyPlugin> modelFamilyPluginInstances;

    @Option(names = { "-a", "--all" }, description = "Show missing runtime modules too")
    boolean showAll;

    @Option(names = { "--json" }, description = "Print a machine-readable module report")
    boolean jsonOutput;

    @Option(names = { "--doctor", "--plugin-doctor" },
            description = "Print a focused detachable plugin validation report")
    boolean pluginDoctor;

    @Option(names = {
            ExternalPluginClasspath.OPTION_PLUGIN_CLASSPATH,
            ExternalPluginClasspath.OPTION_EXTERNAL_PLUGIN_CLASSPATH },
            split = ",",
            description = ExternalPluginClasspath.MODEL_FAMILY_AND_EXTENSION_OPTION_DESCRIPTION)
    List<String> externalPluginClasspath = new ArrayList<>();

    @Option(names = {
            ExternalPluginClasspath.OPTION_PLUGIN_DIR,
            ExternalPluginClasspath.OPTION_EXTERNAL_PLUGIN_DIR },
            split = ",",
            description = ExternalPluginClasspath.PLUGIN_DIRECTORY_OPTION_DESCRIPTION)
    List<String> externalPluginDirectories = new ArrayList<>();

    @Option(names = { "--fail-on-extension-gate" },
            description = "Fail after printing the report when extension policy or provider contracts fail")
    boolean failOnExtensionGate;

    @Option(names = { "--fail-on-model-family-gate" },
            description = "Fail after printing the report when model-family bundle availability or contracts fail")
    boolean failOnModelFamilyGate;

    @Option(names = { "--fail-on-plugin-gates" },
            description = "Fail after printing the report when any plugin release gate fails")
    boolean failOnPluginGates;

    @Option(names = { "--fail-on-route-benchmark-cache" },
            description = "Fail after printing the report when route benchmark cache readiness has problems")
    boolean failOnRouteBenchmarkCache;

    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String CYAN = "\u001B[36m";
    private static final String DIM = "\u001B[2m";
    private static final String BOLD = "\u001B[1m";

    @Override
    public void run() {
        try (ExternalPluginClasspathScope pluginScope = pluginClasspathScope()) {
            run(pluginScope);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("Failed to close external plugin classpath", e);
        }
    }

    private void run(ExternalPluginClasspathScope pluginScope) {
        if (jsonOutput) {
            if (pluginDoctor) {
                printJson(buildPluginDoctorReport(pluginScope), "plugin doctor JSON report");
            } else {
                printJsonReport(pluginScope);
            }
            enforcePluginGatesIfRequested(pluginScope);
            enforceExtensionGateIfRequested(pluginScope);
            enforceModelFamilyGateIfRequested(pluginScope);
            enforceRouteBenchmarkCacheIfRequested();
            return;
        }

        if (pluginDoctor) {
            printPluginDoctor(pluginScope);
            enforcePluginGatesIfRequested(pluginScope);
            enforceExtensionGateIfRequested(pluginScope);
            enforceModelFamilyGateIfRequested(pluginScope);
            enforceRouteBenchmarkCacheIfRequested();
            return;
        }

        // === Section 1: Kernel Platform ===
        printKernelInfo();
        System.out.println();

        // === Section 2: Runtime Modules ===
        printExtensions();
        System.out.println();

        // === Section 3: Runner Plugins ===
        printRunnerPlugins();
        System.out.println();

        // === Section 4: Model Families ===
        printModelFamilies(pluginScope);
        System.out.println();

        // === Section 5: Unified Multimodal Runtimes ===
        printUnifiedRuntimes(pluginScope);
        System.out.println();

        // === Section 6: Extension Availability ===
        printExtensionAvailability(pluginScope);
        System.out.println();

        // === Section 7: Plugin Gates ===
        printPluginGates(pluginScope);
        System.out.println();

        // === Section 8: Dynamic Plugins ===
        printDynamicPlugins();
        enforcePluginGatesIfRequested(pluginScope);
        enforceExtensionGateIfRequested(pluginScope);
        enforceModelFamilyGateIfRequested(pluginScope);
        enforceRouteBenchmarkCacheIfRequested();
    }

    private ExternalPluginClasspathScope pluginClasspathScope() {
        return ExternalPluginClasspathScope.open(
                externalPluginClasspath,
                externalPluginDirectories,
                ExtensionsCommand.class);
    }

    private void printJsonReport(ExternalPluginClasspathScope pluginScope) {
        printJson(buildJsonReport(pluginScope), "modules JSON report");
    }

    private void printJson(Map<String, Object> report, String reportName) {
        try {
            System.out.println(JSON.writeValueAsString(report));
        } catch (Exception e) {
            throw new IllegalStateException("Failed to render " + reportName, e);
        }
    }

    private Map<String, Object> buildJsonReport(ExternalPluginClasspathScope pluginScope) {
        ModelFamilyPluginRegistry modelFamilyRegistry =
                collectModelFamilyPluginRegistry(pluginScope.discoveryClassLoader());
        Map<String, ModelFamilyPlugin> families = collectModelFamilyPlugins(modelFamilyRegistry);
        ExtensionAvailabilityRegistry extensionRegistry = extensionAvailabilityRegistry(pluginScope.discoveryClassLoader());
        UnifiedRuntimeRegistry unifiedRuntimeRegistry =
                collectUnifiedRuntimeRegistry(pluginScope.discoveryClassLoader());
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schemaVersion", JSON_SCHEMA_VERSION);
        report.put("showAll", showAll);
        report.put("externalPluginClasspath", pluginScope.displayClasspath());
        report.put("pluginDirectories", pluginDirectoryReport());
        report.put("kernel", kernelReport());
        report.put("runtimeModules", runtimeModuleReports());
        report.put("runnerPlugins", runnerPluginSectionReport());
        Map<String, Object> extensionAvailability =
                extensionAvailabilitySectionReport(extensionRegistry, pluginScope.registryScope());
        Map<String, Object> modelFamilyBundle = modelFamilyBundleReport(
                manifest,
                families,
                modelFamilyRegistry,
                pluginScope.registryScope());
        report.put("extensionAvailability", extensionAvailability);
        report.put("audioExtensions", audioExtensionSectionReport(extensionRegistry));
        report.put("modelFamilyBundle", modelFamilyBundle);
        PluginGates pluginGates = currentPluginGates(
                extensionRegistry,
                manifest,
                families.keySet(),
                modelFamilyRegistry,
                unifiedRuntimeRegistry,
                ExternalPluginClasspath.inspectPluginDirectories(externalPluginDirectories));
        report.put("pluginGates", pluginGatesReport(pluginGates));
        report.put("pluginDoctor", buildPluginDoctorReport(
                pluginScope,
                manifest,
                families,
                modelFamilyRegistry,
                extensionRegistry,
                unifiedRuntimeRegistry));
        report.put("unifiedRuntimes", unifiedRuntimeSectionReport(
                unifiedRuntimeRegistry,
                pluginScope.registryScope(),
                UnifiedRuntimeRequirementResolver.evaluate(
                        modelFamilyRegistry,
                        unifiedRuntimeRegistry,
                        manifest.families())));
        report.put("modelFamilyPlugins", modelFamilyPluginSectionReport(
                families,
                modelFamilyRegistry,
                pluginScope.registryScope()));
        report.put("dynamicPlugins", dynamicPluginSectionReport());
        return report;
    }

    private Map<String, Object> buildPluginDoctorReport(ExternalPluginClasspathScope pluginScope) {
        return buildPluginDoctorReport(pluginScope, pluginDoctorSnapshot(pluginScope));
    }

    private Map<String, Object> buildPluginDoctorReport(
            ExternalPluginClasspathScope pluginScope,
            ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> modelFamilies,
            ModelFamilyPluginRegistry modelFamilyRegistry,
            ExtensionAvailabilityRegistry extensionRegistry,
            UnifiedRuntimeRegistry unifiedRuntimeRegistry) {
        return buildPluginDoctorReport(pluginScope, pluginDoctorSnapshot(
                manifest,
                modelFamilies,
                modelFamilyRegistry,
                extensionRegistry,
                unifiedRuntimeRegistry,
                ExternalPluginClasspath.inspectPluginDirectories(externalPluginDirectories)));
    }

    private Map<String, Object> buildPluginDoctorReport(
            ExternalPluginClasspathScope pluginScope,
            PluginDoctorSnapshot snapshot) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("schemaVersion", JSON_SCHEMA_VERSION);
        report.put("kind", "pluginDoctor");
        report.put("passed", snapshot.pluginGates().passed());
        report.put("status", snapshot.pluginGates().status());
        report.put("summary", pluginDoctorSummary(pluginScope, snapshot));
        report.put("recommendations", pluginDoctorRecommendations(pluginScope, snapshot));
        report.put("externalPluginClasspath", pluginScope.displayClasspath());
        report.put("registryScope", pluginScope.registryScope());
        report.put("pluginDirectories", pluginDirectoryReport());

        Map<String, Object> classpath = new LinkedHashMap<>();
        classpath.put("active", pluginScope.active());
        classpath.put("registryScope", pluginScope.registryScope());
        classpath.put("entries", pluginScope.displayClasspath());
        classpath.put("pluginDirectories", pluginDirectoryReport());
        report.put("classpath", classpath);

        report.put("pluginGates", pluginGatesReport(snapshot.pluginGates()));
        report.put("extensionGate", extensionAvailabilityGateReport(snapshot.extensionGate()));
        report.put("modelFamilyGate", modelFamilyBundleGateReport(snapshot.modelFamilyGate()));
        report.put("pluginDirectoryReadiness", PluginDirectoryReadinessReports.report(
                snapshot.pluginDirectoryInspection()));
        putRouteBenchmarkCacheReadiness(report, snapshot.routeBenchmarkCache());
        report.put("unifiedRuntimes", unifiedRuntimeSectionReport(
                snapshot.unifiedRuntimeRegistry(),
                pluginScope.registryScope(),
                snapshot.unifiedRuntimeRequirementCompatibility()));

        Map<String, Object> extensionSummary = new LinkedHashMap<>();
        extensionSummary.put("registryScope", pluginScope.registryScope());
        extensionSummary.put("totals", extensionAvailabilityTotals(snapshot.extensions()));
        extensionSummary.put("ids", extensionIds(snapshot.extensions()));
        extensionSummary.put("contractViolations", snapshot.extensionRegistry().contractReport().summaries());
        report.put("extensions", extensionSummary);

        Map<String, Object> modelFamilySummary = new LinkedHashMap<>();
        modelFamilySummary.put("registryScope", pluginScope.registryScope());
        modelFamilySummary.put("discoveredCount", snapshot.modelFamilies().size());
        modelFamilySummary.put("selectedCount", snapshot.manifest().families().size());
        modelFamilySummary.put("selectedDiscoveredCount", selectedDiscoveredFamilyCount(snapshot));
        modelFamilySummary.put("ids", snapshot.modelFamilies().keySet().stream().sorted().toList());
        modelFamilySummary.put("selectedFamilyIds", snapshot.manifest().families());
        modelFamilySummary.put("missingSelectedFamilies",
                snapshot.manifest().missingDiscovered(snapshot.modelFamilies().keySet()));
        modelFamilySummary.put("bundlePreset", snapshot.manifest().hasBundlePreset()
                ? snapshot.manifest().bundlePreset()
                : null);
        modelFamilySummary.put("bundlePresetDescription", snapshot.manifest().displayBundlePreset());
        modelFamilySummary.put("bundleAvailability", modelFamilyBundleAvailabilityReport(snapshot.bundleAvailability()));
        modelFamilySummary.put("productionSafety", modelFamilyBundleProductionSafetyReport(snapshot.manifest()));
        modelFamilySummary.put("catalogReadiness", modelFamilyBundleCatalogReadinessReport(snapshot.manifest()));
        modelFamilySummary.put("capabilityTotals",
                modelFamilyCapabilityTotals(snapshot.modelFamilyRegistry().capabilityMatrix()));
        modelFamilySummary.put("directSafetensor",
                modelFamilyRuntimeCompatibilitySummaryReport(snapshot.directSafetensorSummary()));
        report.put("modelFamilies", modelFamilySummary);

        return report;
    }

    private Map<String, Object> kernelReport() {
        KernelPlatform platform = KernelPlatformDetector.detect();

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("platform", platform.name().toLowerCase());
        report.put("displayName", platform.getDisplayName());
        report.put("description", platform.getDescription());
        report.put("gpuAccelerated", platform.isGpu());
        report.put("architecture", System.getProperty("os.arch", "unknown"));
        report.put("osName", System.getProperty("os.name", "unknown"));
        report.put("osVersion", System.getProperty("os.version", ""));
        report.put("javaVersion", System.getProperty("java.version", "unknown"));
        report.put("javaVendor", System.getProperty("java.vendor", "unknown"));
        return report;
    }

    private Map<String, Object> pluginDirectoryReport() {
        ExternalPluginClasspath.PluginDirectoryReport directories =
                ExternalPluginClasspath.pluginDirectoryReport(externalPluginDirectories);
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("commandDirectories", ExternalPluginClasspath.display(directories.commandDirectories()));
        report.put("configuredDirectories", ExternalPluginClasspath.display(directories.configuredDirectories()));
        report.put("activeDirectories", ExternalPluginClasspath.display(directories.activeDirectories()));
        report.put("defaultDirectory", directories.defaultDirectory().toString());
        report.put("defaultDirectoryExists", directories.defaultDirectoryExists());
        report.put("defaultDirectoryAutoloadEnabled", directories.defaultDirectoryAutoloadEnabled());
        report.put("defaultDirectoryActive", directories.defaultDirectoryActive());
        report.put("jarReadiness", PluginDirectoryReadinessReports.report(
                ExternalPluginClasspath.inspectPluginDirectories(externalPluginDirectories)));
        if (directories.defaultDirectoryExists() && !directories.defaultDirectoryAutoloadEnabled()
                && directories.commandDirectories().isEmpty()) {
            report.put("hint", "Pass --plugin-dir " + directories.defaultDirectory()
                    + " or set GOLLEK_PLUGIN_AUTOLOAD_DEFAULT_DIR=true to load the default plugin directory.");
        } else {
            report.put("hint", null);
        }
        return report;
    }

    private Map<String, Object> unifiedRuntimeSectionReport(
            UnifiedRuntimeRegistry registry,
            String registryScope,
            List<UnifiedRuntimeRequirementCompatibility> requirementCompatibilities) {
        UnifiedRuntimeRegistry effectiveRegistry = registry == null
                ? UnifiedRuntimeRegistry.of(List.of())
                : registry;
        List<UnifiedRuntimeRegistry.UnifiedRuntimeReport> runtimes = effectiveRegistry.reports();
        List<UnifiedRuntimeRequirementCompatibility> requirements = requirementCompatibilities == null
                ? List.of()
                : requirementCompatibilities;
        List<UnifiedRuntimeManifestViolation> conflicts = effectiveRegistry.modelTypeConflicts();
        Map<String, Object> section = new LinkedHashMap<>();
        section.put("registryScope", registryScope);
        section.put("runtimeCount", runtimes.size());
        section.put("validCount", runtimes.stream()
                .filter(UnifiedRuntimeRegistry.UnifiedRuntimeReport::valid)
                .count());
        section.put("invalidCount", runtimes.stream()
                .filter(report -> !report.valid())
                .count());
        section.put("productionReadyCount", runtimes.stream()
                .filter(report -> report.manifestAvailable()
                        && report.manifest().productionReady()
                        && report.valid())
                .count());
        section.put("modelTypes", runtimes.stream()
                .filter(UnifiedRuntimeRegistry.UnifiedRuntimeReport::manifestAvailable)
                .flatMap(report -> report.manifest().modelTypes().stream())
                .distinct()
                .sorted()
                .toList());
        section.put("runtimes", runtimes.stream()
                .map(this::unifiedRuntimeReport)
                .toList());
        section.put("conflicts", conflicts.stream()
                .map(this::unifiedRuntimeViolationReport)
                .toList());
        section.put("contractViolations", runtimes.stream()
                .flatMap(report -> report.violations().stream())
                .map(this::unifiedRuntimeViolationReport)
                .toList());
        section.putAll(UnifiedRuntimeRequirementReports.modelFamilyRequirementSection(requirements));
        return section;
    }

    private Map<String, Object> unifiedRuntimeReport(
            UnifiedRuntimeRegistry.UnifiedRuntimeReport runtime) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("runtimeId", runtime.runtimeId());
        report.put("valid", runtime.valid());
        report.put("manifestAvailable", runtime.manifestAvailable());
        report.put("diagnostics", runtime.diagnostics());
        report.put("violations", runtime.violations().stream()
                .map(this::unifiedRuntimeViolationReport)
                .toList());
        UnifiedRuntimeManifest manifest = runtime.manifest();
        if (manifest == null) {
            report.put("displayName", runtime.runtimeId());
            report.put("modelFamilyIds", List.of());
            report.put("modelTypes", List.of());
            report.put("inputModalities", List.of());
            report.put("readiness", "unavailable");
            report.put("readinessReason", runtime.diagnostics());
            report.put("productionReady", false);
            report.put("requiredProcessorFiles", List.of());
            report.put("requiredTokenizerFiles", List.of());
            report.put("metadata", Map.of());
            return report;
        }
        report.put("displayName", manifest.displayName());
        report.put("modelFamilyIds", manifest.modelFamilyIds());
        report.put("modelTypes", manifest.modelTypes());
        report.put("inputModalities", manifest.inputModalities().stream()
                .map(modality -> modality.name().toLowerCase(Locale.ROOT))
                .toList());
        report.put("readiness", manifest.readiness().statusLabel());
        report.put("readinessReason", manifest.readinessReason());
        report.put("productionReady", manifest.productionReady() && runtime.valid());
        report.put("requiredProcessorFiles", manifest.requiredProcessorFiles());
        report.put("requiredTokenizerFiles", manifest.requiredTokenizerFiles());
        report.put("metadata", manifest.metadata());
        return report;
    }

    private Map<String, Object> unifiedRuntimeViolationReport(
            UnifiedRuntimeManifestViolation violation) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("runtimeId", violation.runtimeId());
        report.put("code", violation.code());
        report.put("message", violation.message());
        report.put("summary", violation.summary());
        return report;
    }

    private List<Map<String, Object>> runtimeModuleReports() {
        Set<String> runtimeProviders = getRuntimeProviderIds();
        return EXTENSIONS.stream()
                .filter(extension -> showAll || isClassPresent(extension.markerClass()))
                .map(extension -> runtimeModuleReport(extension, runtimeProviders))
                .toList();
    }

    private Map<String, Object> runtimeModuleReport(ExtensionDef extension, Set<String> runtimeProviders) {
        boolean packaged = isClassPresent(extension.markerClass());

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("type", extension.type());
        report.put("name", extension.name());
        report.put("profile", extension.profile());
        report.put("packaged", packaged);
        report.put("providerAvailable", runtimeProviders.contains(extension.providerId()));
        report.put("providerId", extension.providerId());
        report.put("markerClass", extension.markerClass());
        return report;
    }

    private Map<String, Object> runnerPluginSectionReport() {
        Map<String, Object> section = new LinkedHashMap<>();
        Map<String, Object> routeReportSchema = RunnerRouteReportContract.schema();
        section.put("routeReportSchema", routeReportSchema);
        section.put("routeReportSchemaValidation",
                RunnerRouteReportContract.schemaValidationReport(routeReportSchema));
        Map<String, Object> routeReportPayloadSchema = RouteReportPayloadContract.schema();
        section.put("routeReportPayloadSchema", routeReportPayloadSchema);
        section.put("routeReportPayloadSchemaValidation",
                RouteReportPayloadContract.schemaValidationReport(routeReportPayloadSchema));
        Map<String, Object> routePreflightDiagnosticSchema = RoutePreflightDiagnosticContract.schema();
        section.put("routePreflightDiagnosticSchema", routePreflightDiagnosticSchema);
        section.put("routePreflightDiagnosticSchemaValidation",
                RoutePreflightDiagnosticContract.schemaValidationReport(routePreflightDiagnosticSchema));
        Map<String, Object> routeContractBundle = RunnerRouteContractBundle.report();
        section.put("routeContractBundle", routeContractBundle);
        section.put("routeContractBundleValidation",
                RunnerRouteContractBundle.validationReport(routeContractBundle));
        section.put("selectionPolicy", RunnerRoutePolicyContract.report());
        putRouteBenchmarkCacheReadiness(section, routeBenchmarkCacheReadinessReport());
        try {
            section.put("plugins", collectRunnerPlugins().values().stream()
                    .map(this::runnerPluginReport)
                    .toList());
            section.put("error", null);
        } catch (Exception e) {
            section.put("plugins", List.of());
            section.put("error", e.getMessage());
        }
        return section;
    }

    private void putRouteBenchmarkCacheReadiness(
            Map<String, Object> report,
            Map<String, Object> routeBenchmarkCache) {
        Map<String, Object> routeBenchmarkCacheSchema = RouteBenchmarkCacheReportContract.schema();
        report.put("routeBenchmarkCacheReportSchema", routeBenchmarkCacheSchema);
        report.put("routeBenchmarkCacheReportSchemaValidation",
                RouteBenchmarkCacheReportContract.schemaValidationReport(routeBenchmarkCacheSchema));
        report.put("routeBenchmarkCacheReportValidation",
                RouteBenchmarkCacheReportContract.reportValidationReport(routeBenchmarkCache));
        report.put("routeBenchmarkCache", routeBenchmarkCache);
    }

    private Map<String, Object> runnerPluginReport(RunnerPlugin plugin) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("id", plugin.id());
        report.put("name", plugin.name());
        report.put("version", plugin.version());
        report.put("format", plugin.format());
        report.put("status", "active");
        report.put("priority", plugin.priority());
        report.put("supportedFormats", plugin.supportedFormats());
        report.put("supportedArchitectures", plugin.supportedArchitectures());
        return report;
    }

    private Map<String, Object> modelFamilyBundleReport(
            ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies,
            ModelFamilyPluginRegistry registry,
            String registryScope) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("present", manifest.present());
        report.put("modelFamilyRegistryScope", registryScope);
        report.put("schemaVersion", manifest.schemaVersion());
        report.put("fingerprint", manifest.bundleFingerprint());
        report.put("detached", manifest.detached());
        report.put("countConsistencyProblems", manifest.countConsistencyProblems());
        report.put("bundlePreset", manifest.hasBundlePreset() ? manifest.bundlePreset() : null);
        report.put("availableBundlePresets", manifest.availableBundlePresets());
        report.put("activeBundlePreset", manifest.activeBundlePreset()
                .map(this::modelFamilyBundlePresetReport)
                .orElse(null));
        report.put("requiresDirectSafetensorRuntime", manifest.requiresDirectSafetensorRuntime());
        report.put("productionSafety", modelFamilyBundleProductionSafetyReport(manifest));
        report.put("catalogReadiness", modelFamilyBundleCatalogReadinessReport(manifest));
        report.put("activeBundlePresetConformance",
                modelFamilyBundlePresetConformanceReport(manifest.activeBundlePresetConformance()));
        report.put("selectorSource", manifest.selectorSource());
        report.put("explicitSelectors", manifest.explicitSelectors());
        report.put("presetSelectors", manifest.presetSelectors());
        report.put("defaultSelectors", manifest.defaultSelectors());
        report.put("policySource", manifest.policySource());
        report.put("presetRequiredFamilies", manifest.presetRequiredFamilies());
        report.put("presetForbiddenFamilies", manifest.presetForbiddenFamilies());
        report.put("presetRequiredAliases", manifest.presetRequiredAliases());
        report.put("presetForbiddenAliases", manifest.presetForbiddenAliases());
        report.put("explicitRequiredFamilies", manifest.explicitRequiredFamilies());
        report.put("explicitForbiddenFamilies", manifest.explicitForbiddenFamilies());
        report.put("explicitRequiredAliases", manifest.explicitRequiredAliases());
        report.put("explicitForbiddenAliases", manifest.explicitForbiddenAliases());
        report.put("requiredFamilies", manifest.bundlePolicy().requiredFamilies());
        report.put("forbiddenFamilies", manifest.bundlePolicy().forbiddenFamilies());
        report.put("requiredAliases", manifest.bundlePolicy().requiredAliases());
        report.put("forbiddenAliases", manifest.bundlePolicy().forbiddenAliases());
        report.put("policyStatus", modelFamilyBundlePolicyStatusReport(manifest.bundlePolicy()));
        report.put("policyViolations", modelFamilyBundlePolicyViolationsReport(manifest.bundlePolicy().violations()));
        report.put("fixtureStatus", modelFamilyBundleFixtureStatusReport(manifest.fixtureStatus()));
        PluginAvailabilityChecker.ModelFamilyBundleAvailability availability =
                PluginAvailabilityChecker.modelFamilyBundleAvailability(manifest, discoveredFamilies.keySet());
        report.put("availabilityStatus", modelFamilyBundleAvailabilityReport(availability));
        report.put("gate", modelFamilyBundleGateReport(
                PluginAvailabilityChecker.modelFamilyBundleGate(
                        manifest,
                        discoveredFamilies.keySet(),
                        registry)));
        report.put("runtimeCompatibility", modelFamilyBundleRuntimeCompatibilityReport(manifest, registry));
        report.put("tokenizerCoverage", modelFamilyTokenizerCoverageReport(
                discoveredFamilies,
                manifest.families(),
                manifest.tokenizerMetadataPendingFamilies(),
                manifest.tokenizerMetadataPendingReasons()));
        report.put("selectors", manifest.selectors());
        report.put("requestedFamilies", manifest.requestedFamilies());
        report.put("requestedProfiles", manifest.requestedProfiles());
        report.put("requestedAliases", manifest.requestedAliases());
        report.put("reservedSelectors", manifest.reservedSelectors());
        report.put("unknownSelectors", manifest.unknownSelectors());
        report.put("families", manifest.families());
        report.put("profiles", manifest.profiles());
        report.put("availableFamilies", manifest.availableFamilies());
        report.put("availableProfiles", manifest.availableProfiles());
        report.put("availableSelectors", manifest.availableSelectors());
        report.put("tokenizerMetadataPendingFamilies", manifest.tokenizerMetadataPendingFamilies());
        report.put("tokenizerMetadataPendingReasons", manifest.tokenizerMetadataPendingReasons());
        report.put("bundlePresets", manifest.bundlePresets().stream()
                .map(this::modelFamilyBundlePresetReport)
                .toList());
        report.put("bundleAliases", manifest.bundleAliasCoverage().stream()
                .map(this::modelFamilyBundleAliasReport)
                .toList());
        report.put("completeAliases", manifest.completeBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::id)
                .toList());
        report.put("partialAliases", manifest.partialBundleAliases().stream()
                .map(ModelFamilyBundleManifest.BundleAliasCoverage::id)
                .toList());
        report.put("omittedFamilies", manifest.omittedFamilies());
        report.put("omittedFamiliesWithProfiles", manifest.omittedFamiliesWithProfiles());
        report.put("missingDiscovered", manifest.missingDiscovered(discoveredFamilies.keySet()));
        report.put("selection", modelFamilySelectionReport(manifest, discoveredFamilies));
        return report;
    }

    private Map<String, Object> modelFamilyBundleFixtureStatusReport(
            ModelFamilyBundleManifest.FixtureStatus fixtureStatus) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("known", fixtureStatus.statusKnown());
        report.put("passed", fixtureStatus.passed());
        report.put("status", fixtureStatus.statusLabel());
        report.put("summary", fixtureStatus.compactStatus());
        report.put("requiredSelectors", fixtureStatus.requiredSelectors());
        report.put("requiredFamilies", fixtureStatus.requiredFamilies());
        report.put("requiredFingerprint", fixtureStatus.requiredFingerprint());
        report.put("inventoryFingerprint", fixtureStatus.inventoryFingerprint());
        report.put("availableFamilyCount", fixtureStatus.availableFamilyCount());
        report.put("fixtureFamilyCount", fixtureStatus.fixtureFamilyCount());
        report.put("requiredFamilyCount", fixtureStatus.requiredFamilyCount());
        report.put("requiredPassedCount", fixtureStatus.requiredPassedCount());
        report.put("missingRequiredCount", fixtureStatus.missingRequiredCount());
        report.put("problemFamilyCount", fixtureStatus.problemFamilyCount());
        report.put("missingRequiredFamilies", fixtureStatus.missingRequiredFamilies());
        report.put("problemFamilies", fixtureStatus.problemFamilies());
        return report;
    }

    private Map<String, Object> modelFamilyBundleAvailabilityReport(
            PluginAvailabilityChecker.ModelFamilyBundleAvailability availability) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("present", availability.present());
        report.put("detached", availability.detached());
        report.put("healthy", availability.healthy());
        report.put("status", availability.status());
        report.put("summary", availability.compactSummary());
        report.put("selectedFamilyCount", availability.selectedFamilyCount());
        report.put("discoveredSelectedFamilyCount", availability.discoveredSelectedFamilyCount());
        report.put("missingSelectedFamilyCount", availability.missingSelectedFamilyCount());
        report.put("omittedFamilyCount", availability.omittedFamilyCount());
        report.put("policyStatus", availability.policyStatus());
        report.put("policyViolationCount", availability.policyViolationCount());
        report.put("productionSafetyStatus", availability.productionSafetyStatus());
        report.put("productionSafetyPassed", availability.productionSafetyPassed());
        report.put("productionPendingTokenizerFamilies", availability.productionPendingTokenizerFamilies());
        report.put("catalogReadinessStatus", availability.catalogReadinessStatus());
        report.put("catalogReadinessPassed", availability.catalogReadinessPassed());
        report.put("productionReadinessPendingCount", availability.productionReadinessPendingCount());
        report.put("directSafetensorPendingCount", availability.directSafetensorPendingCount());
        report.put("productionReadinessPendingFamilies", availability.productionReadinessPendingFamilies());
        report.put("directSafetensorPendingFamilies", availability.directSafetensorPendingFamilies());
        report.put("fixtureStatus", availability.fixtureStatus());
        report.put("fixturePassed", availability.fixturePassed());
        report.put("fixtureMissingRequiredCount", availability.fixtureMissingRequiredCount());
        report.put("fixtureProblemFamilyCount", availability.fixtureProblemFamilyCount());
        report.put("presetConformanceStatus", availability.presetConformanceStatus());
        report.put("problems", availability.problems());
        report.put("remediationHints", availability.remediationHints());
        report.put("missingSelectedFamilies", availability.missingSelectedFamilies());
        report.put("omittedFamilies", availability.omittedFamilies());
        report.put("fixtureMissingRequiredFamilies", availability.fixtureMissingRequiredFamilies());
        report.put("fixtureProblemFamilies", availability.fixtureProblemFamilies());
        return report;
    }

    private Map<String, Object> modelFamilyBundleProductionSafetyReport(ModelFamilyBundleManifest manifest) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("tokenizerMetadataRequired", manifest.productionTokenizerMetadataRequired());
        report.put("tokenizerMetadataReady", manifest.productionTokenizerMetadataReady());
        report.put("passed", manifest.productionSafetyPassed());
        report.put("status", manifest.productionSafetyStatusLabel());
        report.put("summary", manifest.displayProductionSafetyStatus());
        report.put("pendingTokenizerFamilyCount", manifest.selectedTokenizerMetadataPendingFamilies().size());
        report.put("pendingTokenizerFamilies", manifest.selectedTokenizerMetadataPendingFamilies());
        report.put("pendingTokenizerReasons", manifest.selectedTokenizerMetadataPendingReasons());
        return report;
    }

    private Map<String, Object> modelFamilyBundleCatalogReadinessReport(ModelFamilyBundleManifest manifest) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("statusKnown", manifest.catalogReadinessStatusKnown());
        report.put("passed", manifest.catalogReadinessPassed());
        report.put("status", manifest.catalogReadinessStatusLabel());
        report.put("summary", manifest.displayCatalogReadinessStatus());
        report.put("productionReadinessPendingCount", manifest.productionReadinessPendingCount());
        report.put("directSafetensorPendingCount", manifest.directSafetensorPendingCount());
        report.put("productionReadinessPendingFamilies", manifest.productionReadinessPendingFamilies());
        report.put("directSafetensorPendingFamilies", manifest.directSafetensorPendingFamilies());
        return report;
    }

    private Map<String, Object> modelFamilyBundleGateReport(ModelFamilyBundleGate gate) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("passed", gate.passed());
        report.put("failed", gate.failed());
        report.put("status", gate.status());
        report.put("violationCount", gate.violationCount());
        report.put("violations", gate.violations());
        report.put("contractCategoryCounts", gate.contractCategoryCounts());
        report.put("contractRemediationHints", gate.contractRemediationHints());
        report.put("failOnModelFamilyGate", failOnModelFamilyGate);
        return report;
    }

    private Map<String, Object> pluginGatesReport(PluginGates gates) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("passed", gates.passed());
        report.put("failed", gates.failed());
        report.put("status", gates.status());
        report.put("violationCount", gates.violationCount());
        report.put("violations", gates.violations());
        report.put("violationCategories", PluginGateViolationReports.categories(gates));
        report.put("extensionStatus", gates.extensionStatus());
        report.put("modelFamilyStatus", gates.modelFamilyStatus());
        report.put("extensionViolationCount", gates.extensionViolationCount());
        report.put("modelFamilyViolationCount", gates.modelFamilyViolationCount());
        report.put("modelFamilyContractCategoryCounts", gates.modelFamilyContractCategoryCounts());
        report.put("modelFamilyContractRemediationHints", gates.modelFamilyContractRemediationHints());
        report.put("failOnPluginGates", failOnPluginGates);
        return report;
    }

    private Map<String, Object> modelFamilyBundlePresetReport(ModelFamilyBundleManifest.BundlePreset preset) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("id", preset.id());
        report.put("description", preset.description());
        report.put("selectors", preset.selectors());
        report.put("requiredFamilies", preset.requiredFamilies());
        report.put("forbiddenFamilies", preset.forbiddenFamilies());
        report.put("requiredAliases", preset.requiredAliases());
        report.put("forbiddenAliases", preset.forbiddenAliases());
        report.put("selectedFamilies", preset.selectedFamilies());
        report.put("selectedCount", preset.selectedCount());
        Map<String, Object> productionSafety = new LinkedHashMap<>();
        productionSafety.put("known", preset.productionSafetyStatusKnown());
        productionSafety.put("tokenizerMetadataRequired", preset.productionTokenizerMetadataRequired());
        productionSafety.put("tokenizerMetadataReady", preset.productionTokenizerMetadataReady());
        productionSafety.put("passed", preset.productionSafetyPassed());
        productionSafety.put("status", preset.productionSafetyStatusLabel());
        productionSafety.put("summary", preset.productionSafetyCompactStatus());
        productionSafety.put("violationCount", preset.productionSafetyViolationCount());
        productionSafety.put("pendingTokenizerFamilyCount", preset.pendingTokenizerFamilies().size());
        productionSafety.put("pendingTokenizerFamilies", preset.pendingTokenizerFamilies());
        productionSafety.put("pendingTokenizerReasons", preset.pendingTokenizerReasons());
        report.put("productionSafety", productionSafety);
        Map<String, Object> policyStatus = new LinkedHashMap<>();
        policyStatus.put("known", preset.policyStatusKnown());
        policyStatus.put("passed", preset.policyPassed());
        policyStatus.put("status", preset.policyStatusLabel());
        policyStatus.put("violationCount", preset.policyViolationCount());
        report.put("policyStatus", policyStatus);
        report.put("policyViolations", modelFamilyBundlePolicyViolationsReport(preset.policyViolations()));
        return report;
    }

    private Map<String, Object> modelFamilyBundlePresetConformanceReport(
            ModelFamilyBundleManifest.BundlePresetConformance conformance) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("presetId", conformance.hasPreset() ? conformance.presetId() : null);
        report.put("presetMetadataPresent", conformance.presetMetadataPresent());
        report.put("status", conformance.statusLabel());
        report.put("summary", conformance.compactSummary());
        report.put("matchesPreset", conformance.matchesPreset());
        report.put("cleanPresetBuild", conformance.cleanPresetBuild());
        report.put("selectorsMatch", conformance.selectorsMatch());
        report.put("policyInputsMatch", conformance.policyInputsMatch());
        report.put("explicitSelectorOverride", conformance.explicitSelectorOverride());
        report.put("explicitPolicyOverride", conformance.explicitPolicyOverride());
        report.put("selectorAdditions", conformance.selectorAdditions());
        report.put("selectorOmissions", conformance.selectorOmissions());
        report.put("requiredFamilyAdditions", conformance.requiredFamilyAdditions());
        report.put("requiredFamilyOmissions", conformance.requiredFamilyOmissions());
        report.put("forbiddenFamilyAdditions", conformance.forbiddenFamilyAdditions());
        report.put("forbiddenFamilyOmissions", conformance.forbiddenFamilyOmissions());
        report.put("requiredAliasAdditions", conformance.requiredAliasAdditions());
        report.put("requiredAliasOmissions", conformance.requiredAliasOmissions());
        report.put("forbiddenAliasAdditions", conformance.forbiddenAliasAdditions());
        report.put("forbiddenAliasOmissions", conformance.forbiddenAliasOmissions());
        return report;
    }

    private Map<String, Object> modelFamilyBundlePolicyStatusReport(ModelFamilyBundleManifest.BundlePolicy policy) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("known", policy.statusKnown());
        report.put("passed", policy.passed());
        report.put("status", policy.statusLabel());
        report.put("violationCount", policy.violationCount());
        return report;
    }

    private Map<String, Object> modelFamilyBundlePolicyViolationsReport(
            ModelFamilyBundleManifest.BundlePolicyViolations violations) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("missingRequired", violations.missingRequiredFamilies());
        report.put("selectedForbidden", violations.selectedForbiddenFamilies());
        report.put("missingRequiredAliases", aliasFamilyMapReport(violations.missingRequiredAliases(), "missingFamilies"));
        report.put("selectedForbiddenAliases",
                aliasFamilyMapReport(violations.selectedForbiddenAliases(), "selectedFamilies"));
        return report;
    }

    private List<Map<String, Object>> aliasFamilyMapReport(Map<String, List<String>> values, String familyField) {
        return values.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(entry -> {
                    Map<String, Object> report = new LinkedHashMap<>();
                    report.put("alias", entry.getKey());
                    report.put(familyField, entry.getValue());
                    return report;
                })
                .toList();
    }

    private Map<String, Object> modelFamilyBundleAliasReport(ModelFamilyBundleManifest.BundleAliasCoverage alias) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("id", alias.id());
        report.put("description", alias.description());
        report.put("families", alias.families());
        report.put("familyCount", alias.familyCount());
        report.put("selectedFamilies", alias.selectedFamilies());
        report.put("selectedCount", alias.selectedCount());
        report.put("missingFamilies", alias.missingFamilies());
        report.put("missingCount", alias.missingCount());
        report.put("complete", alias.complete());
        report.put("partial", alias.partial());
        return report;
    }

    private List<Map<String, Object>> modelFamilySelectionReport(ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies) {
        LinkedHashSet<String> familyIds = new LinkedHashSet<>(manifest.availableFamilies());
        familyIds.addAll(manifest.families());
        familyIds.addAll(discoveredFamilies.keySet());

        return familyIds.stream()
                .map(familyId -> modelFamilySelectionEntry(familyId, manifest, discoveredFamilies))
                .toList();
    }

    private Map<String, Object> modelFamilySelectionEntry(String familyId,
            ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies) {
        String normalizedId = normalizeModelFamilyId(familyId);
        ModelFamilyPlugin discovered = modelFamilyPluginById(discoveredFamilies, normalizedId);
        String profile = manifest.familyProfiles().get(normalizedId);
        if ((profile == null || profile.isBlank()) && discovered != null) {
            profile = discovered.supportReport().bundleProfile().key();
        }

        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("id", normalizedId);
        entry.put("selected", manifest.isFamilySelected(normalizedId));
        entry.put("discovered", discovered != null);
        entry.put("profile", profile);
        entry.put("path", manifest.familyPaths().get(normalizedId));
        return entry;
    }

    private Map<String, Object> modelFamilyPluginSectionReport(
            Map<String, ModelFamilyPlugin> families,
            ModelFamilyPluginRegistry registry,
            String registryScope) {
        Map<String, Object> section = new LinkedHashMap<>();
        List<ModelFamilyCapabilityMatrixEntry> matrix = registry.capabilityMatrix();
        section.put("modelFamilyRegistryScope", registryScope);
        section.put("plugins", families.values().stream()
                .map(this::modelFamilyPluginReport)
                .toList());
        section.put("capabilityMatrix", matrix.stream()
                .map(this::modelFamilyCapabilityMatrixReport)
                .toList());
        section.put("capabilityTotals", modelFamilyCapabilityTotals(matrix));
        section.put("tokenizerCoverage", modelFamilyTokenizerCoverageReport(families));
        section.put("runtimeManifests", registry.all().stream()
                .flatMap(p -> { try { return java.util.stream.Stream.of(p.runtimeManifest()); } catch (Exception e) { return java.util.stream.Stream.empty(); } })
                .map(this::modelFamilyRuntimeManifestReport)
                .toList());
        section.put("runtimeCompatibility", modelFamilyRuntimeCompatibilityReport(registry));
        Map<String, Object> resolutionReportSchema = ModelFamilyResolutionReportContract.schema();
        section.put("resolutionReportSchema", resolutionReportSchema);
        section.put("resolutionReportSchemaValidation",
                ModelFamilyResolutionReportContract.schemaValidationReport(resolutionReportSchema));
        section.put("conflicts", registry.modelTypeConflicts().stream()
                .map(conflict -> {
                    Map<String, Object> report = new LinkedHashMap<>();
                    report.put("claimType", conflict.claimType());
                    report.put("claim", conflict.claim());
                    report.put("familyIds", conflict.familyIds());
                    report.put("summary", conflict.claimType() + " '" + conflict.claim() + "' claimed by " + conflict.familyIds());
                    return report;
                })
                .toList());
        List<ModelFamilyContractViolation> contractViolations = registry.contractViolations();
        Map<String, Object> contractReport = ModelFamilyContractViolationReports.summary(contractViolations);
        section.put("contract", contractReport);
        section.put("contractValidation", ModelFamilyContractViolationReports.validationReport(contractReport));
        section.put("contractViolationCategories",
                ModelFamilyContractViolationReports.categories(contractViolations));
        section.put("contractViolations", registry.contractViolations().stream()
                .map(ModelFamilyContractViolationReports::violationReport)
                .toList());
        return section;
    }

    private Map<String, Object> modelFamilyPluginReport(ModelFamilyPlugin plugin) {
        ModelFamilySupportReport report = plugin.supportReport();
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("id", report.id());
        value.put("displayName", report.displayName());
        value.put("modelTypes", report.modelTypes());
        value.put("architectureClassNames", report.architectureClassNames());
        value.put("architectureAdapterIds", report.architectureAdapterIds());
        value.put("tokenizerProfileIds", report.tokenizerProfileIds());
        value.put("tokenizerKinds", report.tokenizerKinds().stream()
                .map(kind -> kind.name().toLowerCase())
                .toList());
        value.put("bundleProfile", report.bundleProfile().key());
        value.put("defaultBundle", report.defaultBundle());
        value.put("capabilities", report.capabilities().stream()
                .map(capability -> capability.name().toLowerCase())
                .toList());
        value.put("directSafetensorStatus", report.directSafetensorStatus().name().toLowerCase());
        value.put("directSafetensorReady", report.directSafetensorReady());
        value.put("directSafetensorLabel", report.directSafetensorStatus().name().toLowerCase(Locale.ROOT));
        value.put("directSafetensorReason", report.directSafetensorReason());
        value.put("directSafetensorCaveats", report.directSafetensorCaveats());
        value.put("unifiedRuntimeRequirements", plugin.runtimeManifest().unifiedRuntimeRequirements().stream()
                .map(this::modelFamilyUnifiedRuntimeRequirementReport)
                .toList());
        value.put("metadata", report.metadata());
        return value;
    }

    private Map<String, Object> modelFamilyTokenizerCoverageReport(Map<String, ModelFamilyPlugin> families) {
        return modelFamilyTokenizerCoverageReport(families, List.copyOf(families.keySet()));
    }

    private Map<String, Object> modelFamilyTokenizerCoverageReport(
            Map<String, ModelFamilyPlugin> families,
            List<String> familyIds) {
        return modelFamilyTokenizerCoverageReport(families, familyIds, List.of(), Map.of());
    }

    private Map<String, Object> modelFamilyTokenizerCoverageReport(
            Map<String, ModelFamilyPlugin> families,
            List<String> familyIds,
            List<String> pendingMetadataFamilyIds,
            Map<String, String> pendingMetadataReasons) {
        List<String> normalizedFamilyIds = familyIds == null
                ? List.copyOf(families.keySet())
                : familyIds.stream()
                        .map(ExtensionsCommand::normalizeModelFamilyId)
                        .distinct()
                        .toList();
        Set<String> normalizedPendingFamilyIds = new LinkedHashSet<>();
        if (pendingMetadataFamilyIds != null) {
            pendingMetadataFamilyIds.stream()
                    .map(ExtensionsCommand::normalizeModelFamilyId)
                    .filter(id -> !id.isBlank())
                    .forEach(normalizedPendingFamilyIds::add);
        }
        Map<String, String> normalizedPendingReasons = new LinkedHashMap<>();
        if (pendingMetadataReasons != null) {
            pendingMetadataReasons.forEach((familyId, reason) -> {
                String normalizedId = normalizeModelFamilyId(familyId);
                if (!normalizedId.isBlank() && reason != null && !reason.isBlank()) {
                    normalizedPendingReasons.put(normalizedId, reason);
                }
            });
        }
        List<String> readyFamilyIds = new ArrayList<>();
        List<String> pendingFamilyIds = new ArrayList<>();
        List<String> missingFamilyIds = new ArrayList<>();
        List<String> undiscoveredFamilyIds = new ArrayList<>();
        Map<String, Integer> tokenizerKindCounts = new java.util.TreeMap<>();

        for (String familyId : normalizedFamilyIds) {
            boolean tokenizerMetadataPending = normalizedPendingFamilyIds.contains(familyId);
            ModelFamilyPlugin plugin = modelFamilyPluginById(families, familyId);
            if (plugin == null) {
                undiscoveredFamilyIds.add(familyId);
                if (tokenizerMetadataPending) {
                    pendingFamilyIds.add(familyId);
                }
                continue;
            }
            var report = plugin.supportReport();
            if (report.tokenizerKinds().isEmpty()) {
                if (tokenizerMetadataPending) {
                    pendingFamilyIds.add(familyId);
                } else {
                    missingFamilyIds.add(familyId);
                }
                continue;
            }
            readyFamilyIds.add(familyId);
            report.tokenizerKinds().forEach(kind -> {
                String key = kind.name().toLowerCase();
                tokenizerKindCounts.merge(key, 1, Integer::sum);
            });
        }

        Map<String, Object> coverage = new LinkedHashMap<>();
        coverage.put("familyCount", normalizedFamilyIds.size());
        coverage.put("discoveredFamilyCount", normalizedFamilyIds.size() - undiscoveredFamilyIds.size());
        coverage.put("tokenizerMetadataReadyCount", readyFamilyIds.size());
        coverage.put("tokenizerMetadataPendingCount", pendingFamilyIds.size());
        coverage.put("tokenizerMetadataMissingCount", missingFamilyIds.size());
        coverage.put("undiscoveredFamilyCount", undiscoveredFamilyIds.size());
        coverage.put("readyFamilyIds", readyFamilyIds);
        coverage.put("pendingFamilyIds", pendingFamilyIds);
        coverage.put("missingFamilyIds", missingFamilyIds);
        coverage.put("undiscoveredFamilyIds", undiscoveredFamilyIds);
        coverage.put("pendingReasons", pendingReasonsForCoverage(pendingFamilyIds, normalizedPendingReasons));
        coverage.put("tokenizerKindCounts", new LinkedHashMap<>(tokenizerKindCounts));
        return coverage;
    }

    private Map<String, String> pendingReasonsForCoverage(
            List<String> pendingFamilyIds,
            Map<String, String> pendingMetadataReasons) {
        Map<String, String> reasons = new LinkedHashMap<>();
        for (String familyId : pendingFamilyIds) {
            String reason = pendingMetadataReasons.get(familyId);
            if (reason != null && !reason.isBlank()) {
                reasons.put(familyId, reason);
            }
        }
        return reasons;
    }

    private Map<String, Object> modelFamilyRuntimeManifestReport(ModelFamilyRuntimeManifest manifest) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("familyId", manifest.familyId());
        value.put("modelTypes", manifest.modelTypes());
        value.put("architectureAdapterIds", manifest.architectureAdapterIds());
        value.put("tokenizerProfileIds", manifest.tokenizerProfileIds());
        value.put("tokenizerReady", manifest.tokenizerReady());
        value.put("chatTemplateIds", manifest.chatTemplateIds());
        value.put("chatTemplateReady", manifest.chatTemplateReady());
        value.put("bundleProfile", manifest.bundleProfile() != null ? manifest.bundleProfile().key() : "unknown");
        value.put("directSafetensorStatus", manifest.directSafetensorStatus().name().toLowerCase());
        value.put("directSafetensorReady", manifest.directSafetensorReady());
        value.put("unifiedRuntimeRequired", manifest.requiresUnifiedRuntime());
        value.put("unifiedRuntimeRequirements", manifest.unifiedRuntimeRequirements().stream()
                .map(this::modelFamilyUnifiedRuntimeRequirementReport)
                .toList());
        return value;
    }

    private Map<String, Object> modelFamilyUnifiedRuntimeRequirementReport(
            ModelFamilyUnifiedRuntimeRequirement requirement) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("modelType", requirement.modelType());
        report.put("requiredInputModalities", requirement.requiredInputModalities());
        report.put("productionReadyRequired", requirement.productionReadyRequired());
        report.put("reason", requirement.reason());
        report.put("metadata", requirement.metadata());
        return report;
    }

    private Map<String, Object> modelFamilyRuntimeCompatibilityReport(ModelFamilyPluginRegistry registry) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("directSafetensorSummary", modelFamilyRuntimeCompatibilitySummaryReport(
                registry.directSafetensorCompatibilitySummary()));
        value.put("directSafetensor", registry.directSafetensorCompatibilities().stream()
                .map(this::modelFamilyDirectSafetensorCompatibilityReport)
                .toList());
        return value;
    }

    private Map<String, Object> modelFamilyBundleRuntimeCompatibilityReport(
            ModelFamilyBundleManifest manifest,
            ModelFamilyPluginRegistry registry) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("requiresDirectSafetensorRuntime", manifest.requiresDirectSafetensorRuntime());
        value.put("selectedFamilyIds", manifest.families());
        value.put("selectedDirectSafetensorSummary", modelFamilyRuntimeCompatibilitySummaryReport(
                registry.directSafetensorCompatibilitySummaryForFamilies(manifest.families())));
        value.put("selectedDirectSafetensor",
                registry.directSafetensorCompatibilities().stream()
                        .filter(c -> manifest.families().stream().anyMatch(id ->
                                c.modelFamily().familyIds().contains(id)))
                        .map(this::modelFamilyDirectSafetensorCompatibilityReport)
                        .toList());
        return value;
    }

    private Map<String, Object> modelFamilyRuntimeCompatibilitySummaryReport(
            ModelFamilyRuntimeCompatibilitySummary summary) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("familyCount", summary.familyCount());
        value.put("compatibleFamilyIds", summary.compatibleFamilyIds());
        value.put("problemCounts", summary.problemCounts());
        value.put("empty", summary.empty());
        return value;
    }

    private Map<String, Object> modelFamilyDirectSafetensorCompatibilityReport(
            ModelFamilyRuntimeCompatibility compatibility) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("compatible", compatibility.compatible());
        value.put("problemCodes", compatibility.problemCodes());
        value.put("remediationHints", compatibility.remediationHints());
        value.put("familyIds", compatibility.modelFamily().familyIds());
        value.put("modelType", compatibility.modelFamily().modelType());
        value.put("architectureClassName", compatibility.modelFamily().architectureClassName());
        value.put("selectedArchitectureAdapterId", compatibility.selectedArchitectureAdapterId());
        value.put("selectedArchitectureAdapterBy", compatibility.selectedArchitectureAdapterBy());
        return value;
    }

    private Map<String, Object> modelFamilyCapabilityMatrixReport(ModelFamilyCapabilityMatrixEntry entry) {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("id", entry.id());
        value.put("familyId", entry.familyId());
        value.put("bundleProfile", entry.bundleProfile() != null ? entry.bundleProfile().key() : "unknown");
        value.put("causalLm", entry.causalLm());
        value.put("tokenizer", entry.tokenizer());
        value.put("multimodal", entry.multimodal());
        value.put("systemPrompt", entry.systemPrompt());
        value.put("moe", entry.moe());
        value.put("architectureAdapterPresent", entry.architectureAdapterPresent());
        value.put("directSafetensorReady", entry.directSafetensorReady());
        value.put("summary", entry.compactSummary());
        return value;
    }

    private Map<String, Object> modelFamilyCapabilityTotals(List<ModelFamilyCapabilityMatrixEntry> matrix) {
        Map<String, Object> totals = new LinkedHashMap<>();
        totals.put("families", matrix.size());
        totals.put("tokenizer", count(matrix, ModelFamilyCapabilityMatrixEntry::tokenizer));
        totals.put("multimodal", count(matrix, ModelFamilyCapabilityMatrixEntry::multimodal));
        totals.put("causalLm", count(matrix, ModelFamilyCapabilityMatrixEntry::causalLm));
        totals.put("systemPrompt", count(matrix, ModelFamilyCapabilityMatrixEntry::systemPrompt));
        totals.put("moe", count(matrix, ModelFamilyCapabilityMatrixEntry::moe));
        totals.put("architectureAdapterFamilies",
                count(matrix, ModelFamilyCapabilityMatrixEntry::architectureAdapterPresent));
        totals.put("directSafetensorReady", count(matrix, ModelFamilyCapabilityMatrixEntry::directSafetensorReady));
        return totals;
    }

    private long count(List<ModelFamilyCapabilityMatrixEntry> matrix, Predicate<ModelFamilyCapabilityMatrixEntry> test) {
        return matrix.stream().filter(test).count();
    }

    private Map<String, Object> dynamicPluginSectionReport() {
        Map<String, Object> section = new LinkedHashMap<>();
        try {
            List<tech.kayys.gollek.spi.plugin.GollekPlugin.PluginMetadata> plugins = sdk.listPlugins();
            section.put("plugins", plugins == null ? List.of() : plugins.stream()
                    .map(plugin -> {
                        Map<String, Object> report = new LinkedHashMap<>();
                        report.put("id", plugin.id());
                        report.put("version", plugin.version());
                        report.put("implementationClass", plugin.implementationClass());
                        report.put("order", plugin.order());
                        return report;
                    })
                    .toList());
            section.put("error", null);
        } catch (Exception e) {
            section.put("plugins", List.of());
            section.put("error", e.getMessage());
        }
        return section;
    }

    private Map<String, Object> extensionAvailabilitySectionReport(
            ExtensionAvailabilityRegistry registry,
            String registryScope) {
        List<ExtensionAvailability> extensions = registry.availabilityReports();
        Map<String, Object> byId = new LinkedHashMap<>();
        Map<String, List<Map<String, Object>>> byKind = new LinkedHashMap<>();
        List<Map<String, Object>> reports = extensions.stream()
                .map(this::extensionAvailabilityReport)
                .peek(report -> {
                    byId.put(report.get("id").toString(), report);
                    String kind = report.get("kind").toString();
                    byKind.computeIfAbsent(kind, ignored -> new java.util.ArrayList<>()).add(report);
                })
                .toList();

        Map<String, Object> section = new LinkedHashMap<>();
        section.put("extensionRegistryScope", registryScope);
        section.put("extensions", reports);
        section.put("byId", byId);
        section.put("byKind", byKind);
        section.put("totals", extensionAvailabilityTotals(extensions));
        ExtensionAvailabilityPolicy.Result policy =
                ExtensionAvailabilityPolicy.fromRuntimeConfiguration().evaluate(extensions);
        section.put("policy", extensionAvailabilityPolicyReport(policy));
        ExtensionAvailabilityContractReport contract = registry.contractReport();
        section.put("contract", extensionAvailabilityContractReport(contract));
        section.put("contractViolations", contract.violations().stream()
                .map(this::extensionAvailabilityContractViolationReport)
                .toList());
        section.put("gate", extensionAvailabilityGateReport(ExtensionAvailabilityGate.evaluate(policy, contract)));
        return section;
    }

    private Map<String, Object> extensionAvailabilityGateReport(ExtensionAvailabilityGate gate) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("passed", gate.passed());
        report.put("failed", !gate.passed());
        report.put("status", gate.status());
        report.put("violationCount", gate.violationCount());
        report.put("violations", gate.violations());
        report.put("failOnExtensionGate", failOnExtensionGate);
        return report;
    }

    private Map<String, Object> extensionAvailabilityContractReport(ExtensionAvailabilityContractReport contract) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("passed", contract.passed());
        report.put("failed", contract.failed());
        report.put("status", contract.status());
        report.put("violationCount", contract.violationCount());
        report.put("summaries", contract.summaries());
        report.put("byExtensionId", contract.byExtensionId().entrySet().stream()
                .collect(LinkedHashMap::new,
                        (map, entry) -> map.put(entry.getKey(), entry.getValue().stream()
                                .map(this::extensionAvailabilityContractViolationReport)
                                .toList()),
                        LinkedHashMap::putAll));
        return report;
    }

    private Map<String, Object> extensionAvailabilityContractViolationReport(
            ExtensionAvailabilityContractViolation violation) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("extensionId", violation.extensionId());
        report.put("code", violation.code());
        report.put("message", violation.message());
        report.put("summary", violation.summary());
        return report;
    }

    private Map<String, Object> extensionAvailabilityPolicyReport(ExtensionAvailabilityPolicy.Result result) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("configured", result.configured());
        report.put("passed", result.passed());
        report.put("status", result.status());
        report.put("violations", result.violations());
        report.put("requiredExtensions", result.requiredExtensions());
        report.put("requiredProductionExtensions", result.requiredProductionExtensions());
        report.put("forbiddenExtensions", result.forbiddenExtensions());
        report.put("configuration", extensionAvailabilityPolicyConfigurationReport());
        return report;
    }

    private Map<String, Object> extensionAvailabilityPolicyConfigurationReport() {
        Map<String, Object> configuration = new LinkedHashMap<>();
        configuration.put("requiredExtensions", extensionAvailabilityPolicyConfigurationEntry(
                ExtensionAvailabilityPolicy.REQUIRED_PROPERTY,
                ExtensionAvailabilityPolicy.REQUIRED_ENV));
        configuration.put("requiredProductionExtensions", extensionAvailabilityPolicyConfigurationEntry(
                ExtensionAvailabilityPolicy.REQUIRED_PRODUCTION_PROPERTY,
                ExtensionAvailabilityPolicy.REQUIRED_PRODUCTION_ENV));
        configuration.put("forbiddenExtensions", extensionAvailabilityPolicyConfigurationEntry(
                ExtensionAvailabilityPolicy.FORBIDDEN_PROPERTY,
                ExtensionAvailabilityPolicy.FORBIDDEN_ENV));
        return configuration;
    }

    private Map<String, Object> extensionAvailabilityPolicyConfigurationEntry(
            String propertyName,
            String environmentName) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("propertyName", propertyName);
        entry.put("environmentName", environmentName);
        entry.put("source", extensionAvailabilityPolicyConfigurationSource(propertyName, environmentName));
        return entry;
    }

    private String extensionAvailabilityPolicyConfigurationSource(String propertyName, String environmentName) {
        if (hasText(System.getProperty(propertyName))) {
            return "system_property";
        }
        if (hasText(System.getenv(environmentName))) {
            return "environment";
        }
        return "unset";
    }

    private Map<String, Object> extensionAvailabilityTotals(List<ExtensionAvailability> extensions) {
        Map<String, Object> totals = new LinkedHashMap<>();
        totals.put("extensions", extensions.size());
        totals.put("attached", countExtensionAvailability(extensions, ExtensionAvailability::attached));
        totals.put("detached", countExtensionAvailability(extensions, ExtensionAvailability::detached));
        totals.put("healthy", countExtensionAvailability(extensions, ExtensionAvailability::healthy));
        totals.put("unhealthy", countExtensionAvailability(extensions, availability -> !availability.healthy()));
        totals.put("productionReady", countExtensionAvailability(extensions, ExtensionAvailability::productionReady));
        totals.put("notProductionReady",
                countExtensionAvailability(extensions, availability -> !availability.productionReady()));
        totals.put("byStatus", extensionAvailabilityStatusTotals(extensions));
        totals.put("byKind", extensionAvailabilityKindTotals(extensions));
        return totals;
    }

    private Map<String, Long> extensionAvailabilityStatusTotals(List<ExtensionAvailability> extensions) {
        Map<String, Long> totals = new LinkedHashMap<>();
        extensions.stream()
                .map(ExtensionAvailability::status)
                .sorted()
                .forEach(status -> totals.merge(status, 1L, Long::sum));
        return totals;
    }

    private Map<String, Long> extensionAvailabilityKindTotals(List<ExtensionAvailability> extensions) {
        Map<String, Long> totals = new LinkedHashMap<>();
        extensions.stream()
                .map(ExtensionAvailability::kind)
                .sorted()
                .forEach(kind -> totals.merge(kind, 1L, Long::sum));
        return totals;
    }

    private long countExtensionAvailability(
            List<ExtensionAvailability> extensions,
            Predicate<ExtensionAvailability> test) {
        return extensions.stream().filter(test).count();
    }

    private Map<String, Object> audioExtensionSectionReport(ExtensionAvailabilityRegistry registry) {
        PluginAvailabilityChecker.AudioExtensionAvailability suling =
                PluginAvailabilityChecker.AudioExtensionAvailability.fromExtensionAvailability(
                        registry.availability(SulingAudioExtensionAvailabilityProvider.ID)
                                .orElseGet(() -> new SulingAudioExtensionAvailabilityProvider().availability()));

        Map<String, Object> section = new LinkedHashMap<>();
        section.put("suling", audioExtensionAvailabilityReport(suling));
        return section;
    }

    private Map<String, Object> audioExtensionAvailabilityReport(
            PluginAvailabilityChecker.AudioExtensionAvailability availability) {
        Map<String, Object> report = extensionAvailabilityReport(availability.extensionAvailability());
        report.put("flacAvailable", availability.flacAvailable());
        report.put("flacVersion", availability.flacVersion());
        report.put("mp3EncodingAvailable", availability.mp3EncodingAvailable());
        return report;
    }

    private Map<String, Object> extensionAvailabilityReport(ExtensionAvailability availability) {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("id", availability.id());
        report.put("name", availability.name());
        report.put("kind", availability.kind());
        report.put("attached", availability.attached());
        report.put("detached", availability.detached());
        report.put("healthy", availability.healthy());
        report.put("productionReady", availability.productionReady());
        report.put("status", availability.status());
        report.put("summary", availability.compactSummary());
        report.put("capabilities", availability.capabilities());
        report.put("formats", availability.formats());
        report.put("attributes", availability.attributes());
        report.put("diagnostics", availability.diagnostics());
        report.put("remediationHints", availability.remediationHints());
        return report;
    }

    private void printKernelInfo() {
        System.out.println(BOLD + "=== Kernel Platform ===" + RESET);
        KernelPlatform platform = KernelPlatformDetector.detect();
        System.out.printf("  Platform:     %s%s%s%n", CYAN, platform.getDisplayName(), RESET);
        System.out.printf("  GPU Accel:    %s%n", platform.isCpu()
                ? YELLOW + "No (CPU only)" + RESET
                : GREEN + "Yes" + RESET);
        System.out.printf("  Architecture: %s%n", System.getProperty("os.arch", "unknown"));
        System.out.printf("  OS:           %s %s%n",
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.version", ""));
        System.out.printf("  Java:         %s (%s)%n",
                System.getProperty("java.version", "unknown"),
                System.getProperty("java.vendor", "unknown"));
    }

    private void printExtensions() {
        Set<String> runtimeProviders = getRuntimeProviderIds();

        System.out.println(BOLD + "=== Runtime Modules ===" + RESET);
        System.out.printf("  %-8s %-12s %-18s %-10s %-10s%n",
                "TYPE", "NAME", "PROFILE", "PACKAGED", "PROVIDER");
        System.out.println("  " + "-".repeat(62));

        int shown = 0;
        for (ExtensionDef ext : EXTENSIONS) {
            boolean packaged = isClassPresent(ext.markerClass());
            boolean providerAvailable = runtimeProviders.contains(ext.providerId());
            if (!showAll && !packaged) {
                continue;
            }
            String packagedStr = packaged ? GREEN + "yes" + RESET : DIM + "no" + RESET;
            String providerStr = providerAvailable ? GREEN + "yes" + RESET : DIM + "no" + RESET;
            System.out.printf("  %-8s %-12s %-18s %-19s %-19s%n",
                    ext.type(),
                    ext.name(),
                    ext.profile(),
                    packagedStr,
                    providerStr);
            shown++;
        }

        if (shown == 0) {
            System.out.println("  No packaged runtime modules found. Use --all to see all possible modules.");
        }

        if (!runtimeProviders.isEmpty()) {
            System.out.printf("%n  " + DIM + "Active providers: %s" + RESET + "%n",
                    String.join(", ", runtimeProviders));
        }
    }

    private void printRunnerPlugins() {
        System.out.println(BOLD + "=== Runner Plugins ===" + RESET);
        try {
            Map<String, RunnerPlugin> plugins = collectRunnerPlugins();
            if (plugins.isEmpty()) {
                System.out.println("  No runner plugins discovered.");
                return;
            }

            System.out.printf("  %-20s %-12s %-10s%n", "ID", "FORMAT", "STATUS");
            System.out.println("  " + "-".repeat(44));

            int count = 0;
            for (RunnerPlugin plugin : plugins.values()) {
                String id = plugin.id();
                String format = plugin.format() != null ? plugin.format() : "unknown";
                String status = GREEN + "active" + RESET;
                System.out.printf("  %-20s %-12s %-19s%n", id, format, status);
                count++;
            }

            if (count == 0) {
                System.out.println("  No runner plugins active.");
            } else {
                System.out.printf("%n  " + DIM + "Total runners: %d" + RESET + "%n", count);
            }
            Map<String, Object> benchmarkCache = RouteBenchmarkCacheReports.summaryReport(0);
            System.out.printf("  " + DIM + "Route benchmark cache: %s, entries=%s, file=%s" + RESET + "%n",
                    Boolean.TRUE.equals(benchmarkCache.get("enabled")) ? "enabled" : "disabled",
                    benchmarkCache.get("entryCount"),
                    benchmarkCache.get("cacheFile"));
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to enumerate runner plugins: " + e.getMessage() + RESET);
        }
    }

    private Map<String, RunnerPlugin> collectRunnerPlugins() {
        Map<String, RunnerPlugin> plugins = new LinkedHashMap<>();
        if (runnerPluginInstances != null && !runnerPluginInstances.isUnsatisfied()) {
            for (RunnerPlugin plugin : runnerPluginInstances) {
                plugins.putIfAbsent(plugin.id(), plugin);
            }
        }
        for (RunnerPlugin plugin : RunnerPluginManager.getInstance().getAvailablePlugins()) {
            plugins.putIfAbsent(plugin.id(), plugin);
        }
        return plugins;
    }

    private void printModelFamilies(ExternalPluginClasspathScope pluginScope) {
        System.out.println(BOLD + "=== Model Family Plugins ===" + RESET);
        try {
            ModelFamilyPluginRegistry registry =
                    collectModelFamilyPluginRegistry(pluginScope.discoveryClassLoader());
            Map<String, ModelFamilyPlugin> families = collectModelFamilyPlugins(registry);
            ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
            printPluginClasspathScope(pluginScope);
            printModelFamilyBundleManifest(manifest, families, registry);
            if (families.isEmpty()) {
                if (manifest.detached()) {
                    System.out.println("  Model-family plugins intentionally detached by this CLI build.");
                    System.out.println("  Rebuild with -Pgollek.modelFamilies=core,direct,vlm,embedding,research,all or another selector alias to attach them.");
                    System.out.println("  Or attach external plugin jars/classes with --plugin-dir, --plugin-classpath, or GOLLEK_PLUGIN_DIRS.");
                } else {
                    System.out.println("  No model-family plugins discovered.");
                }
                return;
            }

            System.out.printf("  %-14s %-18s %-13s %-16s %-18s %-14s %-24s%n",
                    "ID", "NAME", "PROFILE", "CAPABILITIES", "TOKENIZER", "DIRECT", "REASON");
            System.out.println("  " + "-".repeat(126));

            for (ModelFamilyPlugin plugin : families.values()) {
                var report = plugin.supportReport();
                String direct = report.directSafetensorReady()
                        ? GREEN + report.directSafetensorStatus().name().toLowerCase(Locale.ROOT) + RESET
                        : DIM + report.directSafetensorStatus().name().toLowerCase(Locale.ROOT) + RESET;
                String capabilities = report.capabilities().stream()
                        .limit(3)
                        .map(Enum::name)
                        .map(String::toLowerCase)
                        .reduce((a, b) -> a + "," + b)
                        .orElse("none");
                String tokenizers = report.tokenizerKinds().stream()
                        .limit(2)
                        .map(kind -> kind.name().toLowerCase())
                        .reduce((a, b) -> a + "," + b)
                        .orElse(DIM + "none" + RESET);
                System.out.printf("  %-14s %-18s %-13s %-16s %-27s %-23s %-24s%n",
                        report.id(),
                        report.displayName(),
                        report.bundleProfile().key(),
                        capabilities,
                        tokenizers,
                        direct,
                        shortText(report.directSafetensorReason(), 24));
            }

            List<ModelFamilyCapabilityMatrixEntry> matrix = registry.capabilityMatrix();
            printModelFamilyCapabilitySummary(matrix);
            printModelFamilyTokenizerCoverageSummary(families);

            List<String> directCaveats = families.values().stream()
                    .map(ModelFamilyPlugin::supportReport)
                    .filter(report -> !report.directSafetensorCaveats().isEmpty())
                    .map(report -> "  - " + report.id() + ": " + report.shortDirectSafetensorSummary())
                    .toList();
            if (!directCaveats.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Partial direct SafeTensor caveats:" + RESET);
                directCaveats.forEach(System.out::println);
            }

            var conflicts = registry.modelTypeConflicts();
            if (!conflicts.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Model family claim conflicts:" + RESET);
                for (var conflict : conflicts) {
                    System.out.println("  - " + conflict.claimType() + " '" + conflict.claim() + "' claimed by " + conflict.familyIds());
                }
            }

            var contractViolations = registry.contractViolations();
            if (!contractViolations.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Model family contract violations:" + RESET);
                System.out.println("  " + CYAN + "Model family contract categories:" + RESET + " "
                        + modelFamilyContractCategorySummary(contractViolations));
                for (var violation : contractViolations) {
                    System.out.println("  - ["
                            + ModelFamilyContractViolationReports.category(violation)
                            + "] "
                            + violation.summary());
                }
                System.out.println("  " + CYAN + "Model family contract recommendations:" + RESET);
                ModelFamilyContractViolationReports.remediationHints(contractViolations)
                        .forEach(hint -> System.out.println("    - " + hint));
            }

            System.out.printf("%n  " + DIM + "Total model families: %d" + RESET + "%n", families.size());
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to enumerate model families: " + e.getMessage() + RESET);
        }
    }

    private void printUnifiedRuntimes(ExternalPluginClasspathScope pluginScope) {
        System.out.println(BOLD + "=== Unified Multimodal Runtimes ===" + RESET);
        try {
            UnifiedRuntimeRegistry registry =
                    collectUnifiedRuntimeRegistry(pluginScope.discoveryClassLoader());
            ModelFamilyPluginRegistry modelFamilyRegistry =
                    collectModelFamilyPluginRegistry(pluginScope.discoveryClassLoader());
            ModelFamilyBundleManifest bundleManifest = ModelFamilyBundleManifest.load();
            List<UnifiedRuntimeRequirementCompatibility> requirements =
                    UnifiedRuntimeRequirementResolver.evaluate(
                            modelFamilyRegistry,
                            registry,
                            bundleManifest.families());
            printPluginClasspathScope(pluginScope);
            if (registry.reports().isEmpty()) {
                System.out.println("  No unified multimodal runtimes discovered.");
                System.out.println("  Attach runtime jars/classes with --plugin-dir, --plugin-classpath, or GOLLEK_PLUGIN_DIRS.");
                printUnifiedRuntimeRequirements(requirements);
                return;
            }

            System.out.printf("  %-28s %-14s %-28s %-20s %-8s%n",
                    "RUNTIME", "READINESS", "MODEL_TYPES", "MODALITIES", "VALID");
            System.out.println("  " + "-".repeat(108));
            for (UnifiedRuntimeRegistry.UnifiedRuntimeReport runtime : registry.reports()) {
                UnifiedRuntimeManifest manifest = runtime.manifest();
                String readiness = manifest == null ? "unavailable" : manifest.readiness().statusLabel();
                String modelTypes = manifest == null
                        ? "-"
                        : joinedLimited(manifest.modelTypes(), 3);
                String modalities = manifest == null
                        ? "-"
                        : joinedLimited(manifest.inputModalities().stream()
                                .map(modality -> modality.name().toLowerCase(Locale.ROOT))
                                .toList(), 4);
                String valid = runtime.valid() ? GREEN + "yes" + RESET : YELLOW + "no" + RESET;
                System.out.printf("  %-28s %-14s %-28s %-20s %-17s%n",
                        runtime.runtimeId(),
                        readiness,
                        modelTypes,
                        modalities,
                        valid);
            }

            var conflicts = registry.modelTypeConflicts();
            if (!conflicts.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Unified runtime claim conflicts:" + RESET);
                for (var conflict : conflicts) {
                    System.out.println("  - " + conflict.summary());
                }
            }

            List<UnifiedRuntimeManifestViolation> violations = registry.reports().stream()
                    .flatMap(report -> report.violations().stream())
                    .toList();
            if (!violations.isEmpty()) {
                System.out.println();
                System.out.println("  " + YELLOW + "Unified runtime manifest violations:" + RESET);
                for (UnifiedRuntimeManifestViolation violation : violations) {
                    System.out.println("  - " + violation.summary());
                }
            }

            printUnifiedRuntimeRequirements(requirements);

            System.out.printf("%n  " + DIM + "Total unified runtimes: %d" + RESET + "%n",
                    registry.reports().size());
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to enumerate unified runtimes: " + e.getMessage() + RESET);
        }
    }

    private void printUnifiedRuntimeRequirements(
            List<UnifiedRuntimeRequirementCompatibility> requirements) {
        System.out.println();
        Map<String, Object> section = UnifiedRuntimeRequirementReports.modelFamilyRequirementSection(requirements);
        if (requirements == null || requirements.isEmpty()) {
            System.out.println("  Selected model-family requirements: none");
            printUnifiedRuntimeRequirementContract(section);
            return;
        }

        long compatibleCount = requirements.stream()
                .filter(UnifiedRuntimeRequirementCompatibility::compatible)
                .count();
        long attentionCount = requirements.stream()
                .filter(UnifiedRuntimeRequirementCompatibility::requiresAttention)
                .count();
        System.out.printf("  Selected model-family requirements: %s (selected=%d, ready=%d, attention=%d)%n",
                attentionCount == 0 ? GREEN + "ready" + RESET : YELLOW + "attention" + RESET,
                requirements.size(),
                compatibleCount,
                attentionCount);
        printUnifiedRuntimeRequirementContract(section);
        Map<String, Object> totals = unifiedRuntimeRequirementTotals(section, requirements);
        Map<String, Long> problemCodeCounts = unifiedRuntimeRequirementProblemCodeCounts(totals);
        if (!problemCodeCounts.isEmpty()) {
            System.out.println("  Attention problem codes: " + problemCodeSummary(problemCodeCounts));
        }
        printUnifiedRuntimeRequirementIssues(totals);
        printUnifiedRuntimeRequirementRecommendations(totals);
        System.out.printf("  %-24s %-26s %-20s %-24s %-24s%n",
                "FAMILY", "MODEL_TYPE", "STATUS", "REQUIRED", "RUNTIMES");
        System.out.println("  " + "-".repeat(122));
        for (UnifiedRuntimeRequirementCompatibility requirement : requirements) {
            System.out.printf("  %-24s %-26s %-20s %-24s %-24s%n",
                    shortText(requirement.familyId(), 24),
                    shortText(requirement.modelType(), 26),
                    colorUnifiedRequirementStatus(requirement),
                    joinedLimited(requirement.requiredInputModalities(), 4),
                    joinedLimited(requirement.runtimeIds(), 3));
            if (requirement.requiresAttention() && !requirement.effectiveRemediationHints().isEmpty()) {
                System.out.println("    Hint: " + requirement.effectiveRemediationHints().getFirst());
            }
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> unifiedRuntimeRequirementTotals(
            Map<String, Object> section,
            List<UnifiedRuntimeRequirementCompatibility> requirements) {
        Object totals = section.get(UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_TOTALS);
        if (totals instanceof Map<?, ?>) {
            return (Map<String, Object>) totals;
        }
        return UnifiedRuntimeRequirementReports.totals(requirements);
    }

    private void printUnifiedRuntimeRequirementContract(Map<String, Object> section) {
        Object value = section.get(
                UnifiedRuntimeRequirementReportFields.Section.MODEL_FAMILY_REQUIREMENT_CONTRACT);
        if (!(value instanceof Map<?, ?> contract)) {
            return;
        }
        boolean passed = Boolean.TRUE.equals(contract.get(UnifiedRuntimeRequirementReportFields.Contract.PASSED));
        String status = passed ? GREEN + "ready" + RESET : YELLOW + "attention" + RESET;
        String schemaVersion = issueText(
                contract,
                UnifiedRuntimeRequirementReportFields.Contract.SCHEMA_VERSION,
                "unknown");
        String problemCount = issueText(
                contract,
                UnifiedRuntimeRequirementReportFields.Contract.PROBLEM_COUNT,
                "0");
        String fingerprint = shortText(issueText(
                contract,
                UnifiedRuntimeRequirementReportFields.Contract.SCHEMA_FINGERPRINT,
                "unknown"), 22);
        System.out.printf("  Requirement report contract: %s (schema=%s, problems=%s, fingerprint=%s)%n",
                status,
                schemaVersion,
                problemCount,
                fingerprint);
        List<String> problems = issueStringList(contract, UnifiedRuntimeRequirementReportFields.Contract.PROBLEMS);
        for (String problem : problems) {
            System.out.println("    Contract problem: " + problem);
        }
    }

    private Map<String, Long> unifiedRuntimeRequirementProblemCodeCounts(
            Map<String, Object> totals) {
        Object counts = totals.get(UnifiedRuntimeRequirementReportFields.Totals.PROBLEM_CODE_COUNTS);
        if (!(counts instanceof Map<?, ?> rawCounts)) {
            return Map.of();
        }
        Map<String, Long> normalized = new TreeMap<>();
        for (Map.Entry<?, ?> entry : rawCounts.entrySet()) {
            if (entry.getKey() instanceof String key && entry.getValue() instanceof Number count) {
                normalized.put(key, count.longValue());
            }
        }
        return new LinkedHashMap<>(normalized);
    }

    private void printUnifiedRuntimeRequirementIssues(Map<String, Object> totals) {
        Object issues = totals.get(UnifiedRuntimeRequirementReportFields.Totals.ISSUES);
        if (!(issues instanceof List<?> issueReports) || issueReports.isEmpty()) {
            return;
        }
        System.out.println("  Attention issues:");
        for (Object issueReport : issueReports) {
            if (!(issueReport instanceof Map<?, ?> issue)) {
                continue;
            }
            String problemCode = issueText(
                    issue,
                    UnifiedRuntimeRequirementReportFields.Issue.PROBLEM_CODE,
                    "unknown");
            String status = issueText(issue, UnifiedRuntimeRequirementReportFields.Issue.STATUS, problemCode);
            String count = issueText(issue, UnifiedRuntimeRequirementReportFields.Issue.COUNT, "0");
            String familyIds = joinedLimited(
                    issueStringList(issue, UnifiedRuntimeRequirementReportFields.Issue.FAMILY_IDS),
                    4);
            String modelTypes = joinedLimited(
                    issueStringList(issue, UnifiedRuntimeRequirementReportFields.Issue.MODEL_TYPES),
                    4);
            System.out.println("    - " + problemCode + " (" + count + ", status=" + status
                    + ", families=" + familyIds + ", model_types=" + modelTypes + ")");
            Object hints = issue.get(UnifiedRuntimeRequirementReportFields.Issue.REMEDIATION_HINTS);
            if (hints instanceof List<?> hintReports && !hintReports.isEmpty()) {
                System.out.println("      Hint: " + hintReports.getFirst());
            }
        }
    }

    private void printUnifiedRuntimeRequirementRecommendations(Map<String, Object> totals) {
        List<String> recommendations = UnifiedRuntimeRequirementRecommendations.fromTotals(totals);
        if (recommendations.isEmpty()) {
            return;
        }
        System.out.println("  Attention recommendations:");
        for (String recommendation : recommendations) {
            System.out.println("    - " + recommendation);
        }
    }

    private String issueText(Map<?, ?> issue, String key, String fallback) {
        Object value = issue.get(key);
        return value == null ? fallback : String.valueOf(value);
    }

    private List<String> issueStringList(Map<?, ?> issue, String key) {
        Object value = issue.get(key);
        if (!(value instanceof List<?> rawValues)) {
            return List.of();
        }
        return rawValues.stream()
                .filter(String.class::isInstance)
                .map(String.class::cast)
                .toList();
    }

    private String problemCodeSummary(Map<String, Long> problemCodeCounts) {
        StringBuilder summary = new StringBuilder();
        for (Map.Entry<String, Long> entry : problemCodeCounts.entrySet()) {
            if (summary.length() > 0) {
                summary.append(", ");
            }
            summary.append(entry.getKey()).append("=").append(entry.getValue());
        }
        return summary.toString();
    }

    private void printModelFamilyCapabilitySummary(List<ModelFamilyCapabilityMatrixEntry> matrix) {
        if (matrix.isEmpty()) {
            return;
        }

        System.out.println();
        System.out.println("  " + CYAN + "Capability matrix summary:" + RESET);
        System.out.printf(
                "  families=%d causal_lm=%d tokenizer=%d adapters-present=%d direct-ready=%d multimodal=%d moe=%d system_prompt=%d%n",
                matrix.size(),
                count(matrix, ModelFamilyCapabilityMatrixEntry::causalLm),
                count(matrix, ModelFamilyCapabilityMatrixEntry::tokenizer),
                count(matrix, ModelFamilyCapabilityMatrixEntry::architectureAdapterPresent),
                count(matrix, ModelFamilyCapabilityMatrixEntry::directSafetensorReady),
                count(matrix, ModelFamilyCapabilityMatrixEntry::multimodal),
                count(matrix, ModelFamilyCapabilityMatrixEntry::moe),
                count(matrix, ModelFamilyCapabilityMatrixEntry::systemPrompt));
    }

    private void printModelFamilyTokenizerCoverageSummary(Map<String, ModelFamilyPlugin> families) {
        if (families.isEmpty()) {
            return;
        }
        Map<String, Object> coverage = modelFamilyTokenizerCoverageReport(families);
        @SuppressWarnings("unchecked")
        Map<String, Integer> kindCounts = (Map<String, Integer>) coverage.get("tokenizerKindCounts");
        @SuppressWarnings("unchecked")
        List<String> missingFamilyIds = (List<String>) coverage.get("missingFamilyIds");
        String kindSummary = kindCounts.isEmpty()
                ? "-"
                : kindCounts.entrySet().stream()
                        .map(entry -> entry.getKey() + "=" + entry.getValue())
                        .reduce((left, right) -> left + ", " + right)
                        .orElse("-");

        System.out.println();
        System.out.println("  " + CYAN + "Tokenizer metadata coverage:" + RESET);
        System.out.printf(
                "  ready=%d pending=%d missing=%d undiscovered=%d kinds=%s%n",
                coverage.get("tokenizerMetadataReadyCount"),
                coverage.get("tokenizerMetadataPendingCount"),
                coverage.get("tokenizerMetadataMissingCount"),
                coverage.get("undiscoveredFamilyCount"),
                kindSummary);
        if (!missingFamilyIds.isEmpty()) {
            System.out.printf("  %sMissing tokenizer metadata:%s %s%n",
                    YELLOW, RESET, String.join(", ", missingFamilyIds));
        }
    }

    private void printModelFamilyBundleManifest(
            ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> discoveredFamilies,
            ModelFamilyPluginRegistry registry) {
        if (!manifest.present()) {
            return;
        }

        System.out.printf("  Build selectors: %s%n", manifest.joinedSelectors());
        System.out.printf("  Build selector source: %s%n", manifest.displaySelectorSource());
        System.out.printf("  Build bundle policy: %s%n", manifest.displayBundlePolicyStatus());
        System.out.printf("  Build policy source: %s%n", manifest.displayPolicySource());
        System.out.printf("  Build fixture status: %s%n", manifest.displayFixtureStatus());
        System.out.printf("  Build availability: %s%n",
                PluginAvailabilityChecker.modelFamilyBundleAvailability(
                        manifest,
                        discoveredFamilies.keySet()).compactSummary());
        ModelFamilyBundleGate gate = PluginAvailabilityChecker.modelFamilyBundleGate(
                manifest,
                discoveredFamilies.keySet(),
                registry);
        System.out.printf("  Build gate: %s%s%s (%d violation(s))%n",
                gate.passed() ? GREEN : YELLOW,
                gate.status(),
                RESET,
                gate.violationCount());
        if (gate.failed()) {
            gate.violations().forEach(violation -> System.out.println("    - " + violation));
        }
        printModelFamilyBundleRuntimeCompatibility(manifest, registry);
        System.out.printf("  Build preset:   %s%n", manifest.displayBundlePreset());
        System.out.printf("  Build preset policy: %s%n", manifest.displayBundlePresetPolicyStatus());
        System.out.printf("  Build production safety: %s%n", manifest.displayProductionSafetyStatus());
        System.out.printf("  Build preset conformance: %s%n", manifest.displayActiveBundlePresetConformance());
        System.out.printf("  Build fingerprint: %s%n", manifest.displayFingerprint());
        System.out.printf("  Build profiles:  %s%n", manifest.displayProfiles());
        System.out.printf("  Build families:  %s%n", manifest.displayFamilies());
        if (!manifest.tokenizerMetadataPendingFamilies().isEmpty()) {
            System.out.printf("  Build tokenizer pending: %s%n",
                    manifest.tokenizerMetadataPendingFamilies().stream()
                            .map(familyId -> {
                                String reason = manifest.tokenizerMetadataPendingReasons().get(familyId);
                                return reason == null || reason.isBlank()
                                        ? familyId
                                        : familyId + " (" + reason + ")";
                            })
                            .reduce((left, right) -> left + ", " + right)
                            .orElse("-"));
        }

        List<String> missing = manifest.missingDiscovered(discoveredFamilies.keySet());
        if (!missing.isEmpty()) {
            System.out.printf("  %sManifest mismatch:%s expected families not discovered: %s%n",
                    YELLOW, RESET, String.join(", ", missing));
        }
        List<String> countProblems = manifest.countConsistencyProblems();
        if (!countProblems.isEmpty()) {
            System.out.printf("  %sManifest count mismatch:%s %s%n",
                    YELLOW, RESET, String.join("; ", countProblems));
        }

        if (showAll) {
            System.out.printf("  Requested selector families: %s%n", manifest.joinedRequestedFamilies());
            System.out.printf("  Requested selector profiles: %s%n", manifest.joinedRequestedProfiles());
            System.out.printf("  Requested selector aliases: %s%n", manifest.joinedRequestedAliases());
            System.out.printf("  Available build selectors: %s%n", manifest.joinedAvailableSelectors());
            System.out.printf("  Available bundle presets: %s%n", manifest.joinedAvailableBundlePresets());
            System.out.printf("  Build bundle presets: %s%n", manifest.joinedBundlePresets());
            System.out.printf("  Build selector aliases: %s%n", manifest.joinedBundleAliases());
            System.out.printf("  Complete selector aliases: %s%n", manifest.joinedCompleteBundleAliases());
            System.out.printf("  Partial selector aliases: %s%n", manifest.joinedPartialBundleAliases());
            System.out.printf("  Unbundled families: %s%n", manifest.joinedOmittedFamiliesWithProfiles());
        }
        System.out.println();
    }

    private void printModelFamilyBundleRuntimeCompatibility(
            ModelFamilyBundleManifest manifest,
            ModelFamilyPluginRegistry registry) {
        ModelFamilyRuntimeCompatibilitySummary summary =
                registry.directSafetensorCompatibilitySummaryForFamilies(manifest.families());
        String requirement = manifest.requiresDirectSafetensorRuntime() ? "required" : "optional";
        List<String> blockedFamilyIds = manifest.families().stream()
                .filter(id -> !summary.compatibleFamilyIds().contains(id))
                .toList();
        String status = summary.empty()
                ? DIM + "none" + RESET
                : blockedFamilyIds.isEmpty() ? GREEN + "ready" + RESET : YELLOW + "blocked" + RESET;
        System.out.printf(
                "  Build direct runtime: %s (%s, selected=%d, compatible=%d, blocked=%d)%n",
                status,
                requirement,
                summary.familyCount(),
                summary.compatibleFamilyIds().size(),
                blockedFamilyIds.size());
        if (!blockedFamilyIds.isEmpty()) {
            System.out.printf("    Blocked families: %s%n", String.join(", ", blockedFamilyIds));
        }
        if (!summary.problemCounts().isEmpty()) {
            System.out.printf("    Runtime problems: %s%n", shortProblemCounts(summary.problemCounts()));
        }
    }

    private ModelFamilyPluginRegistry collectModelFamilyPluginRegistry(ClassLoader pluginClassLoader) {
        ModelFamilyPluginRegistry packagedRegistry = ModelFamilyPluginRegistry.global();
        if (modelFamilyPluginInstances != null && !modelFamilyPluginInstances.isUnsatisfied()) {
            for (ModelFamilyPlugin plugin : modelFamilyPluginInstances) {
                packagedRegistry.register(plugin);
            }
        }
        return packagedRegistry;
    }

    private UnifiedRuntimeRegistry collectUnifiedRuntimeRegistry(ClassLoader pluginClassLoader) {
        return UnifiedRuntimeRegistry.discover(pluginClassLoader);
    }

    private Map<String, ModelFamilyPlugin> collectModelFamilyPlugins(ModelFamilyPluginRegistry registry) {
        Map<String, ModelFamilyPlugin> families = new LinkedHashMap<>();
        for (ModelFamilyPlugin plugin : registry.all()) {
            families.putIfAbsent(plugin.descriptor().id(), plugin);
        }
        return families;
    }

    private ExtensionAvailabilityRegistry extensionAvailabilityRegistry(ClassLoader pluginClassLoader) {
        return PluginAvailabilityChecker.getExtensionAvailabilityRegistry(pluginClassLoader);
    }

    private static String normalizeModelFamilyId(String familyId) {
        if (familyId == null) {
            return "";
        }
        return familyId.startsWith("model-family/")
                ? familyId.substring("model-family/".length())
                : familyId;
    }

    private static ModelFamilyPlugin modelFamilyPluginById(
            Map<String, ModelFamilyPlugin> families,
            String normalizedFamilyId) {
        ModelFamilyPlugin plugin = families.get(normalizedFamilyId);
        return plugin == null ? families.get("model-family/" + normalizedFamilyId) : plugin;
    }

    private static String shortText(String value, int maxLength) {
        if (value == null || value.isBlank() || value.length() <= maxLength) {
            return value == null || value.isBlank() ? "-" : value;
        }
        return value.substring(0, Math.max(0, maxLength - 3)) + "...";
    }

    private static String shortProblemCounts(Map<String, Integer> problemCounts) {
        if (problemCounts == null || problemCounts.isEmpty()) {
            return "none";
        }
        return problemCounts.entrySet().stream()
                .map(entry -> entry.getKey() + "=" + entry.getValue())
                .reduce((left, right) -> left + ", " + right)
                .orElse("none");
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }

    private void printPluginClasspathScope(ExternalPluginClasspathScope pluginScope) {
        if (pluginScope.discoveryClassLoader() == null) {
            printPluginDirectoryScope();
            return;
        }
        System.out.printf("  Registry scope: %s%n", pluginScope.registryScope());
        System.out.printf("  External plugin classpath: %s%n",
                String.join(", ", pluginScope.displayClasspath()));
        printPluginDirectoryScope();
    }

    private void printPluginDirectoryScope() {
        printPluginDirectoryScope(false);
    }

    private void printPluginDirectoryScope(boolean alwaysPrint) {
        ExternalPluginClasspath.PluginDirectoryReport directories =
                ExternalPluginClasspath.pluginDirectoryReport(externalPluginDirectories);
        if (directories.activeDirectories().isEmpty() && !directories.defaultDirectoryExists() && !alwaysPrint) {
            return;
        }

        if (alwaysPrint) {
            System.out.printf("  Plugin dirs active: %s%n", displayPathsOrNone(directories.activeDirectories()));
            System.out.printf("  Plugin dir sources: command=%d configured=%d defaultAutoload=%s%n",
                    directories.commandDirectories().size(),
                    directories.configuredDirectories().size(),
                    directories.defaultDirectoryAutoloadEnabled() ? "on" : "off");
        }
        if (!directories.commandDirectories().isEmpty()) {
            System.out.printf("  Plugin dirs: %s%n",
                    String.join(", ", ExternalPluginClasspath.display(directories.commandDirectories())));
        }
        if (!directories.configuredDirectories().isEmpty()) {
            System.out.printf("  Configured plugin dirs: %s%n",
                    String.join(", ", ExternalPluginClasspath.display(directories.configuredDirectories())));
        }
        if (directories.defaultDirectoryExists() && !directories.defaultDirectoryActive()) {
            System.out.printf("  Default plugin dir: %s %s%n",
                    directories.defaultDirectory(),
                    DIM + "(exists; pass --plugin-dir or set GOLLEK_PLUGIN_AUTOLOAD_DEFAULT_DIR=true to load)" + RESET);
        } else if (directories.defaultDirectoryActive()) {
            System.out.printf("  Default plugin dir: %s %s%n",
                    directories.defaultDirectory(),
                    GREEN + "(active)" + RESET);
        } else if (alwaysPrint) {
            System.out.printf("  Default plugin dir: %s %s%n",
                    directories.defaultDirectory(),
                    DIM + "(missing; autoload "
                            + (directories.defaultDirectoryAutoloadEnabled() ? "on" : "off") + ")" + RESET);
        }
    }

    private void printPluginDoctor(ExternalPluginClasspathScope pluginScope) {
        PluginDoctorSnapshot snapshot = pluginDoctorSnapshot(pluginScope);
        Map<String, Object> extensionTotals = extensionAvailabilityTotals(snapshot.extensions());
        ModelFamilyRuntimeCompatibilitySummary directSummary = snapshot.directSafetensorSummary();

        System.out.println(BOLD + "=== Plugin Doctor ===" + RESET);
        System.out.printf("  Scope: %s%n", pluginScope.registryScope());
        if (pluginScope.active()) {
            System.out.printf("  External plugin classpath: %s%n",
                    String.join(", ", pluginScope.displayClasspath()));
        } else {
            System.out.println("  External plugin classpath: none (using packaged/global registry)");
        }
        printPluginDirectoryScope(true);
        printPluginDirectoryReadiness(snapshot.pluginDirectoryInspection());
        printRouteBenchmarkCacheReadiness(snapshot.routeBenchmarkCache());
        System.out.printf("  Overall: %s (%d violation(s))%n",
                coloredGateStatus(snapshot.pluginGates().passed(), snapshot.pluginGates().status()),
                snapshot.pluginGates().violationCount());
        System.out.printf("  Gate categories: %s%n", pluginGateCategorySummary(snapshot.pluginGates()));
        System.out.printf("  Extension gate: %s (%d violation(s))%n",
                coloredGateStatus(snapshot.extensionGate().passed(), snapshot.extensionGate().status()),
                snapshot.extensionGate().violationCount());
        System.out.printf("  Model-family gate: %s (%d violation(s))%n",
                coloredGateStatus(snapshot.modelFamilyGate().passed(), snapshot.modelFamilyGate().status()),
                snapshot.modelFamilyGate().violationCount());

        System.out.println();
        System.out.printf("  Extensions: discovered=%s attached=%s healthy=%s productionReady=%s%n",
                extensionTotals.get("extensions"),
                extensionTotals.get("attached"),
                extensionTotals.get("healthy"),
                extensionTotals.get("productionReady"));
        System.out.printf("  Extension ids: %s%n", joinedLimited(extensionIds(snapshot.extensions()), 12));
        System.out.printf("  Unified runtimes: discovered=%d productionReady=%s conflicts=%d%n",
                snapshot.unifiedRuntimeRegistry().reports().size(),
                snapshot.unifiedRuntimeRegistry().reports().stream()
                        .filter(report -> report.manifestAvailable()
                                && report.manifest().productionReady()
                                && report.valid())
                        .count(),
                snapshot.unifiedRuntimeRegistry().modelTypeConflicts().size());
        printUnifiedRuntimeRequirements(snapshot.unifiedRuntimeRequirementCompatibility());

        System.out.printf("  Model families: discovered=%d selected=%d selectedDiscovered=%d%n",
                snapshot.modelFamilies().size(),
                snapshot.manifest().families().size(),
                selectedDiscoveredFamilyCount(snapshot));
        System.out.printf("  Model family ids: %s%n",
                joinedLimited(snapshot.modelFamilies().keySet().stream().sorted().toList(), 12));
        System.out.printf("  Bundle preset: %s%n", snapshot.manifest().displayBundlePreset());
        System.out.printf("  Bundle availability: %s%n", snapshot.bundleAvailability().compactSummary());
        System.out.printf("  Bundle production safety: %s%n", snapshot.manifest().displayProductionSafetyStatus());

        List<String> blockedFamilyIds = snapshot.manifest().families().stream()
                .filter(id -> !directSummary.compatibleFamilyIds().contains(id))
                .toList();
        String directStatus = directSummary.empty()
                ? DIM + "none" + RESET
                : blockedFamilyIds.isEmpty() ? GREEN + "ready" + RESET : YELLOW + "blocked" + RESET;
        System.out.printf(
                "  Direct SafeTensor: %s (selected=%d compatible=%d blocked=%d)%n",
                directStatus,
                directSummary.familyCount(),
                directSummary.compatibleFamilyIds().size(),
                blockedFamilyIds.size());

        List<String> missingFamilies = snapshot.manifest().missingDiscovered(snapshot.modelFamilies().keySet());
        if (!missingFamilies.isEmpty()) {
            System.out.printf("  %sMissing selected families:%s %s%n",
                    YELLOW, RESET, String.join(", ", missingFamilies));
        }
        if (!blockedFamilyIds.isEmpty()) {
            System.out.printf("  %sDirect runtime blocked families:%s %s%n",
                    YELLOW, RESET, String.join(", ", blockedFamilyIds));
        }
        if (snapshot.pluginGates().failed()) {
            System.out.println();
            System.out.println("  " + YELLOW + "Gate violations:" + RESET);
            snapshot.pluginGates().violations().forEach(violation -> System.out.println("    - " + violation));
        }

        System.out.println();
        System.out.println(BOLD + "Recommendations" + RESET);
        pluginDoctorRecommendations(pluginScope, snapshot)
                .forEach(recommendation -> System.out.println("  - " + recommendation));
    }

    private void printPluginDirectoryReadiness(
            ExternalPluginClasspath.PluginDirectoryInspection inspection) {
        if (inspection.activeDirectories().isEmpty() && inspection.jars().isEmpty() && inspection.errors().isEmpty()) {
            return;
        }
        System.out.printf(
                "  Plugin jar readiness: jars=%d model-family=%d ready=%d not-ready=%d "
                        + "unified-runtime=%d ready=%d not-ready=%d "
                        + "tokenizer-metadata=%d ready/%d pending/%d missing/%d invalid%n",
                inspection.jarCount(),
                inspection.modelFamilyPluginCandidates(),
                inspection.pluginInstallReady(),
                inspection.pluginInstallNotReady(),
                inspection.unifiedRuntimePluginCandidates(),
                inspection.unifiedRuntimeReady(),
                inspection.unifiedRuntimeNotReady(),
                inspection.pluginTokenizerMetadataReady(),
                inspection.pluginTokenizerMetadataPending(),
                inspection.pluginTokenizerMetadataMissing(),
                inspection.pluginTokenizerMetadataInvalid());
        for (String error : inspection.errors()) {
            System.out.println("    - " + error);
        }
        inspection.jars().stream()
                .filter(ExternalPluginClasspath.PluginDirectoryJarReport::pluginInstallCandidate)
                .forEach(jar -> System.out.printf("    - %s (%s): %s%n",
                        jar.path(),
                        jar.pluginInstallReady() ? "ready" : "not-ready",
                        pluginDirectoryJarReadinessDetail(jar)));
        inspection.jars().stream()
                .filter(ExternalPluginClasspath.PluginDirectoryJarReport::hasUnifiedMultimodalRuntimeServiceEntry)
                .forEach(jar -> System.out.printf("    - %s (unified-runtime): providers=%s%n",
                        jar.path(),
                        jar.unifiedRuntimeReady()
                                ? String.join(",", jar.unifiedMultimodalRuntimeProviders())
                                : "not-ready: " + String.join("; ", jar.unifiedRuntimeErrors())));
    }

    private static String pluginDirectoryJarReadinessDetail(
            ExternalPluginClasspath.PluginDirectoryJarReport jar) {
        if (jar.pluginInstallReady()) {
            return "mainClass=" + (jar.pluginMainClass().isBlank() ? "unknown" : jar.pluginMainClass())
                    + "; pluginId=" + (jar.pluginDescriptorId().isBlank() ? "unknown" : jar.pluginDescriptorId())
                    + "; families=" + (jar.pluginFamilies().isEmpty()
                            ? "none"
                            : String.join(",", jar.pluginFamilies()))
                    + "; tokenizerKind=" + (jar.pluginTokenizerKind().isBlank()
                            ? "unspecified"
                            : jar.pluginTokenizerKind())
                    + "; tokenizerKinds=" + (jar.pluginTokenizerKinds().isEmpty()
                            ? "none"
                            : String.join(",", jar.pluginTokenizerKinds()))
                    + "; tokenizerMetadataStatus=" + jar.pluginTokenizerMetadataStatus()
                    + (jar.pluginTokenizerMetadataPendingReason().isBlank()
                            ? ""
                            : "; tokenizerMetadataPendingReason=" + jar.pluginTokenizerMetadataPendingReason());
        }
        return String.join("; ", jar.pluginInstallErrors());
    }

    private PluginDoctorSnapshot pluginDoctorSnapshot(ExternalPluginClasspathScope pluginScope) {
        ModelFamilyPluginRegistry modelFamilyRegistry =
                collectModelFamilyPluginRegistry(pluginScope.discoveryClassLoader());
        Map<String, ModelFamilyPlugin> modelFamilies = collectModelFamilyPlugins(modelFamilyRegistry);
        ExtensionAvailabilityRegistry extensionRegistry =
                extensionAvailabilityRegistry(pluginScope.discoveryClassLoader());
        UnifiedRuntimeRegistry unifiedRuntimeRegistry =
                collectUnifiedRuntimeRegistry(pluginScope.discoveryClassLoader());
        return pluginDoctorSnapshot(
                ModelFamilyBundleManifest.load(),
                modelFamilies,
                modelFamilyRegistry,
                extensionRegistry,
                unifiedRuntimeRegistry,
                ExternalPluginClasspath.inspectPluginDirectories(externalPluginDirectories));
    }

    private PluginDoctorSnapshot pluginDoctorSnapshot(
            ModelFamilyBundleManifest manifest,
            Map<String, ModelFamilyPlugin> modelFamilies,
            ModelFamilyPluginRegistry modelFamilyRegistry,
            ExtensionAvailabilityRegistry extensionRegistry,
            UnifiedRuntimeRegistry unifiedRuntimeRegistry,
            ExternalPluginClasspath.PluginDirectoryInspection pluginDirectoryInspection) {
        List<ExtensionAvailability> extensions = extensionRegistry.availabilityReports();
        ExtensionAvailabilityGate extensionGate = extensionAvailabilityGate(extensionRegistry);
        ModelFamilyBundleGate modelFamilyGate = PluginAvailabilityChecker.modelFamilyBundleGate(
                manifest,
                modelFamilies.keySet(),
                modelFamilyRegistry);
        List<UnifiedRuntimeRequirementCompatibility> unifiedRuntimeRequirements =
                UnifiedRuntimeRequirementResolver.evaluate(
                        modelFamilyRegistry,
                        unifiedRuntimeRegistry,
                        manifest.families());
        PluginGates gates = applyRunnerRouteContractGate(PluginGates.withUnifiedRuntimeRequirements(
                PluginGates.withUnifiedRuntimeReadiness(
                        PluginGates.withPluginDirectoryReadiness(
                                PluginGates.evaluate(extensionGate, modelFamilyGate),
                                pluginDirectoryInspection),
                        unifiedRuntimeRegistry),
                unifiedRuntimeRequirements));
        return new PluginDoctorSnapshot(
                manifest,
                modelFamilyRegistry,
                modelFamilies,
                extensionRegistry,
                extensions,
                extensionGate,
                modelFamilyGate,
                gates,
                PluginAvailabilityChecker.modelFamilyBundleAvailability(manifest, modelFamilies.keySet()),
                modelFamilyRegistry.directSafetensorCompatibilitySummaryForFamilies(manifest.families()),
                routeBenchmarkCacheReadinessReport(),
                unifiedRuntimeRegistry,
                unifiedRuntimeRequirements,
                pluginDirectoryInspection);
    }

    private String pluginDoctorSummary(ExternalPluginClasspathScope pluginScope, PluginDoctorSnapshot snapshot) {
        String scope = pluginScope.active() ? "External plugin classpath" : "Packaged plugin registry";
        if (snapshot.pluginGates().passed()) {
            return scope + " passed configured plugin gates.";
        }
        return scope + " failed configured plugin gates: " + snapshot.pluginGates().status() + ".";
    }

    private List<String> pluginDoctorRecommendations(
            ExternalPluginClasspathScope pluginScope,
            PluginDoctorSnapshot snapshot) {
        List<String> recommendations = new ArrayList<>();
        ExternalPluginClasspath.PluginDirectoryReport directories =
                ExternalPluginClasspath.pluginDirectoryReport(externalPluginDirectories);
        if (!pluginScope.active()) {
            recommendations.add("No external plugin classpath is attached; this doctor run only validates packaged/global plugins.");
        }
        if (directories.defaultDirectoryExists() && !directories.defaultDirectoryActive()) {
            recommendations.add("Load the default plugin directory with --plugin-dir "
                    + directories.defaultDirectory()
                    + " or set GOLLEK_PLUGIN_AUTOLOAD_DEFAULT_DIR=true.");
        }
        ExternalPluginClasspath.PluginDirectoryInspection pluginDirectoryInspection =
                snapshot.pluginDirectoryInspection();
        boolean tokenizerMetadataNeedsAttention =
                pluginDirectoryInspection.pluginTokenizerMetadataPending() > 0
                        || pluginDirectoryInspection.pluginTokenizerMetadataMissing() > 0
                        || pluginDirectoryInspection.pluginTokenizerMetadataInvalid() > 0;
        boolean unifiedRuntimeNeedsAttention =
                !snapshot.unifiedRuntimeRegistry().modelTypeConflicts().isEmpty()
                        || snapshot.unifiedRuntimeRegistry().reports().stream()
                                .anyMatch(report -> !report.valid());
        if (snapshot.pluginGates().passed() && !tokenizerMetadataNeedsAttention && !unifiedRuntimeNeedsAttention) {
            recommendations.add("Configured plugin gates pass for this scope.");
        }
        if (snapshot.extensionGate().failed()) {
            recommendations.add("Fix extension policy/provider contract violations before enabling fail-on-extension-gate in CI.");
        }
        if (!snapshot.bundleAvailability().productionSafetyPassed()) {
            String pendingFamilies = snapshot.bundleAvailability().productionPendingTokenizerFamilies().isEmpty()
                    ? "the pending tokenizer families"
                    : String.join(", ", snapshot.bundleAvailability().productionPendingTokenizerFamilies());
            recommendations.add("Keep " + pendingFamilies + " out of prod_* model-family bundle presets, "
                    + "or finish their tokenizer adapter metadata before production inference.");
        }
        if ("failed".equals(snapshot.bundleAvailability().catalogReadinessStatus())) {
            Set<String> readinessPending = new LinkedHashSet<>();
            readinessPending.addAll(snapshot.bundleAvailability().productionReadinessPendingFamilies());
            readinessPending.addAll(snapshot.bundleAvailability().directSafetensorPendingFamilies());
            String pendingFamilies = readinessPending.isEmpty()
                    ? "the production/direct readiness blockers"
                    : String.join(", ", readinessPending);
            recommendations.add("Fix " + pendingFamilies + " with :ui:gollek-cli:validateModelFamilyModuleCatalog "
                    + "before packaging this model-family bundle for production inference.");
        }
        if (snapshot.modelFamilyGate().failed()) {
            recommendations.add("Fix model-family availability/contracts or adjust the selected production bundle preset.");
        }
        if (pluginDirectoryInspection.pluginInstallNotReady() > 0) {
            recommendations.add("Repair or detach not-ready model-family plugin jars in plugin directories before "
                    + "enabling --fail-on-plugin-gates in CI.");
        }
        if (pluginDirectoryInspection.pluginTokenizerMetadataMissing() > 0) {
            recommendations.add("Add plugin.json properties.tokenizerKind or tokenizerKinds to model-family plugin jars "
                    + "so tokenizer routing is explicit for production bundles.");
        }
        if (pluginDirectoryInspection.pluginTokenizerMetadataPending() > 0) {
            recommendations.add("Keep pending-tokenizer model-family plugin jars out of production presets until their "
                    + "tokenizer adapter is implemented, or clear tokenizerMetadataStatus=pending once ready.");
        }
        if (pluginDirectoryInspection.pluginTokenizerMetadataInvalid() > 0) {
            recommendations.add("Use supported plugin.json tokenizer kinds for model-family plugin jars: "
                    + ExternalPluginClasspath.SUPPORTED_TOKENIZER_KINDS_DESCRIPTION + ".");
        }
        if (!pluginDirectoryInspection.errors().isEmpty()) {
            recommendations.add("Fix plugin directory inspection errors before relying on external plugin autoload.");
        }
        if (!snapshot.unifiedRuntimeRegistry().modelTypeConflicts().isEmpty()) {
            recommendations.add("Detach duplicate unified runtime plugins, or split their model_type claims so each "
                    + "unified model type has a single runtime owner.");
        }
        if (snapshot.unifiedRuntimeRegistry().reports().stream().anyMatch(report -> !report.valid())) {
            recommendations.add("Fix unified runtime manifest violations before relying on the runtime for model-family "
                    + "availability or production inference.");
        }
        List<String> routeBenchmarkProblems = stringList(snapshot.routeBenchmarkCache().get("problems"));
        if (routeBenchmarkProblems.contains("cache_disabled")) {
            recommendations.add("Enable the route benchmark cache before using benchmark-cache route profiles in production readiness checks.");
        }
        if (routeBenchmarkProblems.contains("cache_unreadable")
                || routeBenchmarkProblems.contains("cache_directory_unwritable")) {
            recommendations.add("Fix route benchmark cache filesystem permissions or set "
                    + "-Dgollek.cli.route_benchmark_cache_dir to a writable deployment cache directory.");
        }
        if (routeBenchmarkProblems.contains("invalid_cache_lines")
                || routeBenchmarkProblems.contains("stale_entries_ignored_for_route_profiles")) {
            recommendations.add("Refresh or prune route benchmark measurements with gollek route-benchmarks doctor --strict "
                    + "and gollek route-benchmarks prune --dry-run before comparing runner performance.");
        }
        if (routeBenchmarkProblems.contains("no_trusted_route_profiles")) {
            recommendations.add("Run a fresh local generation for each production provider/format route before relying on "
                    + "benchmark-cache route profile advice.");
        }
        recommendations.addAll(UnifiedRuntimeRequirementRecommendations.fromTotals(
                UnifiedRuntimeRequirementReports.totals(snapshot.unifiedRuntimeRequirementCompatibility())));

        List<String> missingFamilies = snapshot.manifest().missingDiscovered(snapshot.modelFamilies().keySet());
        if (!missingFamilies.isEmpty()) {
            recommendations.add("Attach model-family plugins with --plugin-dir, --plugin-classpath, or "
                    + "GOLLEK_PLUGIN_DIRS for: " + String.join(", ", missingFamilies) + ".");
        }

        ModelFamilyRuntimeCompatibilitySummary directSummary = snapshot.directSafetensorSummary();
        List<String> blockedFamilyIds = snapshot.manifest().families().stream()
                .filter(id -> !directSummary.compatibleFamilyIds().contains(id))
                .toList();
        if (snapshot.manifest().requiresDirectSafetensorRuntime() && !blockedFamilyIds.isEmpty()) {
            recommendations.add("Add direct SafeTensor adapters/tokenizers for: "
                    + String.join(", ", blockedFamilyIds) + ".");
        }
        if (recommendations.isEmpty()) {
            recommendations.add("No immediate plugin action is required for this scope.");
        }
        return recommendations;
    }

    private void printRouteBenchmarkCacheReadiness(Map<String, Object> report) {
        if (report == null || report.isEmpty()) {
            return;
        }
        System.out.printf(
                "  Route benchmark cache: %s (strictHealthy=%s trusted=%s fresh=%s stale=%s)%n",
                report.getOrDefault("profileTrustStatus", "unknown"),
                report.getOrDefault("strictHealthy", false),
                report.getOrDefault("trustedEntryCount", 0),
                report.getOrDefault("freshEntryCount", 0),
                report.getOrDefault("staleEntryCount", 0));
        List<String> problems = stringList(report.get("problems"));
        if (!problems.isEmpty()) {
            System.out.printf("  %sRoute benchmark cache problems:%s %s%n",
                    YELLOW, RESET, String.join(", ", problems));
        }
    }

    private static List<String> stringList(Object value) {
        if (value instanceof List<?> values) {
            return values.stream()
                    .map(String::valueOf)
                    .toList();
        }
        return List.of();
    }

    private long selectedDiscoveredFamilyCount(PluginDoctorSnapshot snapshot) {
        return snapshot.manifest().families().stream()
                .filter(snapshot.modelFamilies()::containsKey)
                .count();
    }

    private static List<String> extensionIds(List<ExtensionAvailability> extensions) {
        return extensions.stream()
                .map(ExtensionAvailability::id)
                .filter(ExtensionsCommand::hasText)
                .sorted()
                .toList();
    }

    private static String joinedLimited(List<String> values, int limit) {
        if (values == null || values.isEmpty()) {
            return "none";
        }
        int safeLimit = Math.max(1, limit);
        if (values.size() <= safeLimit) {
            return String.join(", ", values);
        }
        return String.join(", ", values.subList(0, safeLimit))
                + ", +" + (values.size() - safeLimit) + " more";
    }

    private static String displayPathsOrNone(List<Path> paths) {
        List<String> display = ExternalPluginClasspath.display(paths);
        return display.isEmpty() ? "none" : String.join(", ", display);
    }

    private void printExtensionAvailability(ExternalPluginClasspathScope pluginScope) {
        ExtensionAvailabilityRegistry registry =
                extensionAvailabilityRegistry(pluginScope.discoveryClassLoader());
        List<ExtensionAvailability> extensions = registry.availabilityReports();

        System.out.println(BOLD + "=== Extension Availability ===" + RESET);
        printPluginClasspathScope(pluginScope);
        if (extensions.isEmpty()) {
            System.out.println("  No extension availability providers discovered.");
            return;
        }

        Map<String, Object> totals = extensionAvailabilityTotals(extensions);
        System.out.printf("  Summary: extensions=%s attached=%s detached=%s healthy=%s productionReady=%s%n",
                totals.get("extensions"),
                totals.get("attached"),
                totals.get("detached"),
                totals.get("healthy"),
                totals.get("productionReady"));
        ExtensionAvailabilityPolicy.Result policy =
                ExtensionAvailabilityPolicy.fromRuntimeConfiguration().evaluate(extensions);
        if (policy.configured()) {
            System.out.printf("  Policy: %s%s%s (%d violation(s))%n",
                    policy.passed() ? GREEN : YELLOW,
                    policy.status(),
                    RESET,
                    policy.violations().size());
            policy.violations().forEach(violation -> System.out.println("    - " + violation));
        }
        ExtensionAvailabilityContractReport contract = registry.contractReport();
        System.out.printf("  Contract: %s%s%s (%d violation(s))%n",
                contract.passed() ? GREEN : YELLOW,
                contract.status(),
                RESET,
                contract.violationCount());
        if (contract.failed()) {
            System.out.println("  " + YELLOW + "Contract violations:" + RESET);
            contract.summaries().forEach(summary -> System.out.println("    - " + summary));
        }
        ExtensionAvailabilityGate gate = ExtensionAvailabilityGate.evaluate(policy, contract);
        System.out.printf("  Release gate: %s%s%s (%d violation(s))%n",
                gate.passed() ? GREEN : YELLOW,
                gate.status(),
                RESET,
                gate.violationCount());

        System.out.printf("  %-14s %-10s %-18s %-14s %-12s %-20s%n",
                "ID", "KIND", "NAME", "STATUS", "PRODUCTION", "FORMATS");
        System.out.println("  " + "-".repeat(94));
        for (ExtensionAvailability availability : extensions) {
            System.out.printf("  %-14s %-10s %-18s %-23s %-21s %-20s%n",
                    availability.id(),
                    availability.kind(),
                    availability.name(),
                    colorStatus(availability.status()),
                    availability.productionReady() ? GREEN + "ready" + RESET : YELLOW + "not-ready" + RESET,
                    availability.formats().isEmpty() ? DIM + "none" + RESET : String.join(", ", availability.formats()));
            if (!availability.remediationHints().isEmpty()) {
                System.out.println("    Hint: " + availability.remediationHints().getFirst());
            }
        }
    }

    private void printPluginGates(ExternalPluginClasspathScope pluginScope) {
        PluginGates gates = currentPluginGates(pluginScope);

        System.out.println(BOLD + "=== Plugin Gates ===" + RESET);
        System.out.printf("  Combined gate: %s%s%s (%d violation(s))%n",
                gates.passed() ? GREEN : YELLOW,
                gates.status(),
                RESET,
                gates.violationCount());
        System.out.printf("  Extension gate: %s (%d violation(s))%n",
                gates.extensionStatus(),
                gates.extensionViolationCount());
        System.out.printf("  Model-family gate: %s (%d violation(s))%n",
                gates.modelFamilyStatus(),
                gates.modelFamilyViolationCount());
        System.out.printf("  Gate categories: %s%n", pluginGateCategorySummary(gates));
        if (gates.failed()) {
            gates.violations().forEach(violation -> System.out.println("    - " + violation));
        }
    }

    private PluginGates currentPluginGates(ExternalPluginClasspathScope pluginScope) {
        ModelFamilyPluginRegistry modelFamilyRegistry =
                collectModelFamilyPluginRegistry(pluginScope.discoveryClassLoader());
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        Map<String, ModelFamilyPlugin> families = collectModelFamilyPlugins(modelFamilyRegistry);
        return currentPluginGates(
                extensionAvailabilityRegistry(pluginScope.discoveryClassLoader()),
                manifest,
                families.keySet(),
                modelFamilyRegistry,
                collectUnifiedRuntimeRegistry(pluginScope.discoveryClassLoader()),
                ExternalPluginClasspath.inspectPluginDirectories(externalPluginDirectories));
    }

    private PluginGates currentPluginGates(
            ExtensionAvailabilityRegistry extensionRegistry,
            ModelFamilyBundleManifest manifest,
            Set<String> discoveredFamilyIds,
            ModelFamilyPluginRegistry modelFamilyRegistry,
            UnifiedRuntimeRegistry unifiedRuntimeRegistry,
            ExternalPluginClasspath.PluginDirectoryInspection pluginDirectoryInspection) {
        List<UnifiedRuntimeRequirementCompatibility> unifiedRuntimeRequirements =
                UnifiedRuntimeRequirementResolver.evaluate(
                        modelFamilyRegistry,
                        unifiedRuntimeRegistry,
                        manifest.families());
        return applyRunnerRouteContractGate(PluginGates.withUnifiedRuntimeRequirements(
                PluginGates.withUnifiedRuntimeReadiness(
                        PluginGates.withPluginDirectoryReadiness(PluginGates.evaluate(
                                extensionAvailabilityGate(extensionRegistry),
                                PluginAvailabilityChecker.modelFamilyBundleGate(
                                        manifest,
                                        discoveredFamilyIds,
                                        modelFamilyRegistry)),
                                pluginDirectoryInspection),
                        unifiedRuntimeRegistry),
                unifiedRuntimeRequirements));
    }

    private PluginGates applyRunnerRouteContractGate(PluginGates gates) {
        return RouteBenchmarkCacheReportContract.applyGate(
                RunnerRouteContractBundle.applyGate(gates, RunnerRouteContractBundle.report()),
                routeBenchmarkCacheReadinessReport());
    }

    private Map<String, Object> routeBenchmarkCacheReadinessReport() {
        Map<String, Object> report = new LinkedHashMap<>(RouteBenchmarkCacheReports.summaryReport(5));
        report.put("failOnRouteBenchmarkCache", failOnRouteBenchmarkCache);
        return report;
    }

    private ExtensionAvailabilityGate extensionAvailabilityGate(ExtensionAvailabilityRegistry registry) {
        List<ExtensionAvailability> extensions = registry.availabilityReports();
        ExtensionAvailabilityPolicy.Result policy =
                ExtensionAvailabilityPolicy.fromRuntimeConfiguration().evaluate(extensions);
        return ExtensionAvailabilityGate.evaluate(policy, registry.contractReport());
    }

    private void enforcePluginGatesIfRequested(ExternalPluginClasspathScope pluginScope) {
        if (!failOnPluginGates) {
            return;
        }
        PluginGates gates = currentPluginGates(pluginScope);
        if (gates.failed()) {
            throw new IllegalStateException(gates.failureMessage());
        }
    }

    private void enforceExtensionGateIfRequested(ExternalPluginClasspathScope pluginScope) {
        if (!failOnExtensionGate) {
            return;
        }
        ExtensionAvailabilityGate gate = extensionAvailabilityGate(
                extensionAvailabilityRegistry(pluginScope.discoveryClassLoader()));
        if (gate.failed()) {
            throw new IllegalStateException(gate.failureMessage());
        }
    }

    private void enforceModelFamilyGateIfRequested(ExternalPluginClasspathScope pluginScope) {
        if (!failOnModelFamilyGate) {
            return;
        }
        ModelFamilyBundleManifest manifest = ModelFamilyBundleManifest.load();
        ModelFamilyPluginRegistry registry =
                collectModelFamilyPluginRegistry(pluginScope.discoveryClassLoader());
        Map<String, ModelFamilyPlugin> families = collectModelFamilyPlugins(registry);
        ModelFamilyBundleGate gate = PluginAvailabilityChecker.modelFamilyBundleGate(
                manifest,
                families.keySet(),
                registry);
        if (gate.failed()) {
            throw new IllegalStateException(gate.failureMessage());
        }
    }

    private void enforceRouteBenchmarkCacheIfRequested() {
        if (!failOnRouteBenchmarkCache) {
            return;
        }
        Map<String, Object> report = routeBenchmarkCacheReadinessReport();
        List<String> problems = stringList(report.get("problems"));
        if (!problems.isEmpty()) {
            throw new IllegalStateException("Route benchmark cache failed readiness gate: "
                    + String.join(", ", problems));
        }
    }

    private void printDynamicPlugins() {
        System.out.println(BOLD + "=== Dynamic Plugins ===" + RESET);
        try {
            List<tech.kayys.gollek.spi.plugin.GollekPlugin.PluginMetadata> plugins = sdk.listPlugins();
            if (plugins != null && !plugins.isEmpty()) {
                for (var plugin : plugins) {
                    System.out.printf("  - %s (%s) v%s%n",
                            plugin.implementationClass(), plugin.id(), plugin.version());
                }
            } else {
                System.out.println("  No dynamic plugins discovered.");
            }
        } catch (Exception e) {
            System.out.println("  " + YELLOW + "Failed to fetch plugins: " + e.getMessage() + RESET);
        }

        System.out.println();
        System.out.println(DIM + "Tip: enable cloud modules at build time with "
                + "-Pext-cloud-gemini,ext-cloud-cerebras" + RESET);
        System.out.println(DIM + "Tip: select model-family bundles with "
                + "-Pgollek.modelFamilies=direct,vlm,embedding,moe,research, none or all" + RESET);
    }

    private Set<String> getRuntimeProviderIds() {
        return new LinkedHashSet<>();
    }

    private boolean isClassPresent(String className) {
        try {
            Class.forName(className, false, Thread.currentThread().getContextClassLoader());
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    private String coloredGateStatus(boolean passed, String status) {
        return (passed ? GREEN : YELLOW) + status + RESET;
    }

    private String pluginGateCategorySummary(PluginGates gates) {
        List<String> activeCategories = PluginGateViolationReports.categories(gates).entrySet().stream()
                .filter(entry -> entry.getValue() > 0)
                .map(entry -> entry.getKey() + "=" + entry.getValue())
                .toList();
        return activeCategories.isEmpty() ? "none" : String.join(", ", activeCategories);
    }

    private String modelFamilyContractCategorySummary(List<ModelFamilyContractViolation> violations) {
        List<String> activeCategories = ModelFamilyContractViolationReports.categories(violations).entrySet().stream()
                .filter(entry -> entry.getValue() > 0)
                .map(entry -> entry.getKey() + "=" + entry.getValue())
                .toList();
        return activeCategories.isEmpty() ? "none" : String.join(", ", activeCategories);
    }

    private String colorUnifiedRequirementStatus(UnifiedRuntimeRequirementCompatibility requirement) {
        if (requirement != null && requirement.compatible()) {
            return GREEN + requirement.status() + RESET;
        }
        String status = requirement == null ? "unknown" : requirement.status();
        return YELLOW + status + RESET;
    }

    private String colorStatus(String status) {
        if ("ready".equalsIgnoreCase(status)) {
            return GREEN + status + RESET;
        }
        if ("fallback".equalsIgnoreCase(status) || "detached".equalsIgnoreCase(status)) {
            return YELLOW + status + RESET;
        }
        return DIM + status + RESET;
    }
}
