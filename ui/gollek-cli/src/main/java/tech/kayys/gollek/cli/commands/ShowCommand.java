package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.arc.Unremovable;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Any;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.cli.util.ExternalPluginClasspath;
import tech.kayys.gollek.cli.util.ExternalPluginClasspathScope;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportContract;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReportFields.DirectArchitecture;
import tech.kayys.gollek.cli.util.ModelFamilyResolutionReports;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.ModelResolver;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelLoader;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPluginRegistry;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyResolution;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeCompatibility;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyRuntimeManifest;
import tech.kayys.alkhawarizm.spi.model.ModelFamilySupportReport;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.alkhawarizm.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.context.RequestContext;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Show model details using GollekSdk.
 * Usage: gollek show <model-id>
 */
@Dependent
@Unremovable
@Command(name = "show", description = "Show details for a specific model")
public class ShowCommand implements Runnable {
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    @Inject
    GollekSdk sdk;

    @Inject
    @Any
    Instance<ModelFamilyPlugin> modelFamilyPluginInstances;

    @Parameters(index = "0", description = "Model ID or path")
    public String modelId;

    @Option(names = { "--json" }, description = "Print model details as JSON")
    boolean json;

    @Option(names = {
            ExternalPluginClasspath.OPTION_PLUGIN_CLASSPATH,
            ExternalPluginClasspath.OPTION_EXTERNAL_PLUGIN_CLASSPATH },
            split = ",",
            description = ExternalPluginClasspath.MODEL_FAMILY_OPTION_DESCRIPTION)
    List<String> externalPluginClasspath = new ArrayList<>();

    @Option(names = {
            ExternalPluginClasspath.OPTION_PLUGIN_DIR,
            ExternalPluginClasspath.OPTION_EXTERNAL_PLUGIN_DIR },
            split = ",",
            description = ExternalPluginClasspath.PLUGIN_DIRECTORY_OPTION_DESCRIPTION)
    List<String> externalPluginDirectories = new ArrayList<>();

    @Override
    public void run() {
        try (ExternalPluginClasspathScope pluginScope = pluginClasspathScope()) {
            run(pluginScope);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            System.err.println("Failed to show model: " + e.getMessage());
        }
    }

    private void run(ExternalPluginClasspathScope pluginScope) {
        try {
            Optional<ModelResolver.ResolvedModel> resolvedOpt = ModelResolver.resolve(sdk, modelId);
            if (resolvedOpt.isEmpty()) {
                LocalModelIndex.refreshFromDisk();
                Optional<LocalModelIndex.Entry> idx = LocalModelIndex.find(modelId);
                if (idx.isEmpty()) {
                    System.err.println("Model not found: " + modelId);
                    return;
                }
                LocalModelIndex.Entry entry = idx.get();
                 ModelInfo model = ModelInfo.builder()
                         .modelId(entry.id != null ? entry.id : modelId)
                         .name(entry.name)
                         .format(entry.format)
                         .sizeBytes(entry.sizeBytes)
                         .updatedAt(LocalModelIndex.parseInstant(entry.updatedAt))
                         .metadata(java.util.Map.of(
                                 "path", entry.path != null ? entry.path : "",
                                 "source", entry.source != null ? entry.source : "local"))
                         .build();
                resolvedOpt = Optional.of(new ModelResolver.ResolvedModel(
                        model.getModelId(), model, entry.path != null ? Path.of(entry.path) : null, false));
            }

            ModelResolver.ResolvedModel resolved = resolvedOpt.get();
            ModelFamilyPluginRegistry registry =
                    collectModelFamilyPluginRegistry(pluginScope.discoveryClassLoader());
            Optional<ModelFamilyResolution> modelFamily = resolveModelFamily(resolved, registry);

            if (json) {
                printModelJson(resolved, modelFamily, registry, pluginScope);
            } else {
                printModelDetails(resolved, modelFamily, registry, pluginScope);
            }

        } catch (Exception e) {
            System.err.println("Failed to show model: " + e.getMessage());
        }
    }

    private ExternalPluginClasspathScope pluginClasspathScope() {
        return ExternalPluginClasspathScope.open(
                externalPluginClasspath,
                externalPluginDirectories,
                ShowCommand.class);
    }

    private void printModelDetails(
            ModelResolver.ResolvedModel resolved,
            Optional<ModelFamilyResolution> modelFamily,
            ModelFamilyPluginRegistry registry,
            ExternalPluginClasspathScope pluginScope) {
        ModelInfo model = resolved.info();
        System.out.println("Model Details");
        System.out.println("=".repeat(50));
        System.out.printf("ID:       %s%n", model.getModelId());
        System.out.printf("Name:     %s%n", model.getName() != null ? model.getName() : "N/A");
        System.out.printf("Version:  %s%n", model.getVersion() != null ? model.getVersion() : "N/A");
        System.out.printf("Format:   %s%n", model.getFormat() != null ? model.getFormat() : "N/A");
        System.out.printf("Runtime:  %s%n", isRunnableLocally(resolved) ? "runnable" : "checkpoint-only");
        System.out.printf("Size:     %s%n", model.getSizeFormatted());
        if (model.getQuantization() != null) {
            System.out.printf("Quant:    %s%n", model.getQuantization());
        }
        System.out.printf("Created:  %s%n", model.getCreatedAt() != null ? model.getCreatedAt() : "N/A");
        System.out.printf("Modified: %s%n", model.getUpdatedAt() != null ? model.getUpdatedAt() : "N/A");

        if (model.getMetadata() != null && !model.getMetadata().isEmpty()) {
            System.out.println("\nMetadata:");
            model.getMetadata().forEach((key, value) -> System.out.printf("  %s: %s%n", key, value));
        }
        if (pluginScope.discoveryClassLoader() != null) {
            System.out.println("\nPlugin Scope:");
            System.out.printf("  Model Family Registry: %s%n", pluginScope.registryScope());
            System.out.printf("  External Plugin Classpath: %s%n",
                    String.join(", ", pluginScope.displayClasspath()));
        }
        modelFamily.ifPresent(resolution -> printModelFamilyDetails(
                resolution,
                modelDirectory(resolved),
                registry));
        if (!isRunnableLocally(resolved)) {
            System.out.println("\nNote:");
            System.out.println(
                    "  Stored as origin checkpoint artifacts; convert to GGUF/TorchScript for local inference.");
        }
    }

    private void printModelFamilyDetails(
            ModelFamilyResolution resolution,
            Optional<Path> modelDir,
            ModelFamilyPluginRegistry registry) {
        System.out.println("\nModel Family:");
        System.out.printf("  Status:   %s%n", resolution.status());
        System.out.printf("  Summary:  %s%n", resolution.summary());
        System.out.printf("  Families: %s%n", resolution.familyIds().isEmpty()
                ? "N/A"
                : String.join(", ", resolution.familyIds()));
        List<String> problemCodes = ModelFamilyResolutionReports.problemCodes(resolution, modelDir, registry);
        if (!problemCodes.isEmpty()) {
            System.out.printf("  Problems: %s%n", String.join(", ", problemCodes));
            System.out.println("  Remediation:");
            for (String hint : ModelFamilyResolutionReports.remediationHints(resolution, modelDir, registry)) {
                System.out.printf("    %s%n", hint);
            }
        }
        if (!resolution.supportReports().isEmpty()) {
            System.out.println("  Support:");
            for (ModelFamilySupportReport report : resolution.supportReports()) {
                System.out.printf("    %s: bundle=%s direct=%s tokenizers=%s%n",
                        report.id(),
                        report.bundleProfile(),
                        report.shortDirectSafetensorSummary(),
                        report.tokenizerProfileIds().isEmpty()
                                ? "none"
                                : String.join(", ", report.tokenizerProfileIds()));
            }
        }
        Map<String, Object> directArchitecture =
                ModelFamilyResolutionReports.directArchitectureReport(resolution, registry);
        @SuppressWarnings("unchecked")
        List<String> adapterIds = (List<String>) directArchitecture.get(DirectArchitecture.ADAPTER_IDS);
        if (!adapterIds.isEmpty()) {
            System.out.printf("  Direct Architecture: selected=%s adapters=%s%n",
                    directArchitecture.get(DirectArchitecture.SELECTED_ADAPTER_ID),
                    String.join(", ", adapterIds));
        }
        ModelFamilyRuntimeCompatibility directCompatibility =
                ModelFamilyResolutionReports.directSafetensorCompatibility(resolution, modelDir, registry);
        if (ModelFamilyResolutionReports.directAdapterExpected(resolution)
                || !directCompatibility.selectedArchitectureAdapterId().isBlank()) {
            System.out.printf("  Direct SafeTensor Runtime: %s selected=%s%n",
                    directCompatibility.compatible() ? "compatible" : "blocked",
                    directCompatibility.selectedArchitectureAdapterId().isBlank()
                            ? "none"
                            : directCompatibility.selectedArchitectureAdapterId());
        }
        if (!resolution.tokenizerDescriptors().isEmpty()) {
            System.out.println("  Tokenizers:");
            for (ModelTokenizerDescriptor descriptor : resolution.tokenizerDescriptors()) {
                String usable = modelDir
                        .map(dir -> descriptor.firstExistingFileGroup(dir).isPresent() ? "usable" : "missing files")
                        .orElse("not inspected");
                System.out.printf("    %s (%s): %s%n", descriptor.id(), descriptor.kind(), usable);
            }
        }
        if (!resolution.runtimeManifests().isEmpty()) {
            System.out.println("  Runtime Manifests:");
            for (ModelFamilyRuntimeManifest manifest : resolution.runtimeManifests()) {
                System.out.printf("    %s: tokenizer=%s chat=%s direct=%s adapters=%s%n",
                        manifest.familyId(),
                        manifest.tokenizerReady() ? "ready" : "none",
                        manifest.chatTemplateReady()
                                ? String.join(", ", manifest.chatTemplateIds())
                                : "none",
                         manifest.directSafetensorStatus().name().toLowerCase(Locale.ROOT),
                         manifest.architectureAdapterIds().isEmpty()
                                ? "none"
                                : String.join(", ", manifest.architectureAdapterIds()));
            }
        }
    }

    private void printModelJson(
            ModelResolver.ResolvedModel resolved,
            Optional<ModelFamilyResolution> modelFamily,
            ModelFamilyPluginRegistry registry,
            ExternalPluginClasspathScope pluginScope) throws Exception {
        ModelInfo model = resolved.info();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", model.getModelId());
        out.put("name", model.getName());
        out.put("version", model.getVersion());
        out.put("format", model.getFormat());
        out.put("runtime", isRunnableLocally(resolved) ? "runnable" : "checkpoint-only");
        out.put("sizeBytes", model.getSizeBytes());
        out.put("size", model.getSizeFormatted());
        out.put("createdAt", model.getCreatedAt() != null ? model.getCreatedAt().toString() : null);
        out.put("updatedAt", model.getUpdatedAt() != null ? model.getUpdatedAt().toString() : null);
        out.put("metadata", model.getMetadata());
        out.put("externalPluginClasspath", pluginScope.displayClasspath());
        out.put("modelFamilyRegistryScope", pluginScope.registryScope());
        Optional<Path> modelDir = modelDirectory(resolved);
        Map<String, Object> modelFamilyReport = modelFamily
                .map(resolution -> ModelFamilyResolutionReports.report(resolution, modelDir, registry))
                .orElse(null);
        out.put("modelFamily", modelFamilyReport);
        out.put("modelFamilyValidation", modelFamilyReport == null
                ? null
                : ModelFamilyResolutionReportContract.validationReport(modelFamilyReport));
        System.out.println(JSON.writeValueAsString(out));
    }

    private Optional<ModelFamilyResolution> resolveModelFamily(
            ModelResolver.ResolvedModel resolved,
            ModelFamilyPluginRegistry registry) {
        try {
            Optional<Path> modelDir = modelDirectory(resolved);
            if (modelDir.isEmpty() || !Files.isRegularFile(modelDir.get().resolve("config.json"))) {
                return Optional.empty();
            }
            ModelConfig config = ModelConfig.load(modelDir.get().resolve("config.json"), JSON);
            return Optional.of(registry.resolve(config));
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private Optional<Path> modelDirectory(ModelResolver.ResolvedModel resolved) {
        Path path = resolved.localPath();
        if (path == null) {
            path = ModelResolver.extractPath(resolved.info()).orElse(null);
        }
        if (path == null) {
            return Optional.empty();
        }
        if (Files.isRegularFile(path)) {
            return Optional.ofNullable(path.getParent());
        }
        return Optional.of(path);
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

    private boolean isRunnableLocally(ModelResolver.ResolvedModel resolved) {
        ModelInfo model = resolved.info();
        String format = model.getFormat() != null ? model.getFormat().trim().toUpperCase(Locale.ROOT) : "";
        if (format.equals("GGUF") || format.equals("TORCHSCRIPT") || format.equals("ONNX")) {
            return true;
        }
        if (format.equals("SAFETENSORS") || format.equals("PYTORCH") || format.equals("BIN")) {
            return true;
        }
        Path path = resolved.localPath();
        if (path == null) {
            path = ModelResolver.extractPath(model).orElse(null);
        }
        if (path != null) {
            String normalized = path.toString().replace('\\', '/').toLowerCase(Locale.ROOT);
            if (normalized.endsWith(".gguf")) {
                return true;
            }
            if (normalized.endsWith(".safetensors")
                    || normalized.endsWith(".safetensor")
                    || normalized.endsWith(".bin")
                    || normalized.endsWith(".pt")
                    || normalized.endsWith(".pth")) {
                return true;
            }
        }
        return true;
    }

}
