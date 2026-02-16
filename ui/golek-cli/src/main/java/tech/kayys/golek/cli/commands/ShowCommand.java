package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.model.ModelInfo;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;

/**
 * Show model details using GolekSdk.
 * Usage: golek show <model-id>
 */
@Dependent
@Unremovable
@Command(name = "show", description = "Show details for a specific model")
public class ShowCommand implements Runnable {
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    @Inject
    GolekSdk sdk;

    @Parameters(index = "0", description = "Model ID to show")
    String modelId;

    @Option(names = { "--json" }, description = "Print model details as JSON")
    boolean json;

    @Override
    public void run() {
        try {
            Optional<LocalModelResolver.ResolvedModel> resolvedOpt = LocalModelResolver.resolve(sdk, modelId);
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
                resolvedOpt = Optional.of(new LocalModelResolver.ResolvedModel(
                        model.getModelId(), model, entry.path != null ? Path.of(entry.path) : null, false));
            }

            if (json) {
                printModelJson(resolvedOpt.get());
            } else {
                printModelDetails(resolvedOpt.get());
            }

        } catch (Exception e) {
            System.err.println("Failed to show model: " + e.getMessage());
        }
    }

    private void printModelDetails(LocalModelResolver.ResolvedModel resolved) {
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
        if (!isRunnableLocally(resolved)) {
            System.out.println("\nNote:");
            System.out.println(
                    "  Stored as origin checkpoint artifacts; convert to GGUF/TorchScript for local inference.");
        }
    }

    private void printModelJson(LocalModelResolver.ResolvedModel resolved) throws Exception {
        ModelInfo model = resolved.info();
        java.util.Map<String, Object> out = new java.util.LinkedHashMap<>();
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
        System.out.println(JSON.writeValueAsString(out));
    }

    private boolean isRunnableLocally(LocalModelResolver.ResolvedModel resolved) {
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
            path = LocalModelResolver.extractPath(model).orElse(null);
        }
        if (path != null) {
            String normalized = path.toString().replace('\\', '/').toLowerCase(Locale.ROOT);
            if (normalized.contains("/.golek/models/djl/")) {
                return true;
            }
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
