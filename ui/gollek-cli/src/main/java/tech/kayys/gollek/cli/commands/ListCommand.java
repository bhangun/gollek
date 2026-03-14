package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.ModelInfo;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * List local models using GollekSdk.
 * Usage: gollek list [--format table|json] [--limit N]
 */
@Dependent
@Unremovable
@Command(name = "list", description = "List available models")
public class ListCommand implements Runnable {

    @Inject
    GollekSdk sdk;

    @Option(names = { "-f", "--format" }, description = "Output format: table, json", defaultValue = "table")
    public String format;

    @Option(names = { "-l", "--limit" }, description = "Maximum models to list", defaultValue = "50")
    public int limit;

    @Option(names = {
            "--runnable-only" }, description = "Show only models runnable in local Java runtime", defaultValue = "false")
    boolean runnableOnly;

    @Override
    public void run() {
        try {
            List<ModelInfo> models = new ArrayList<>();
            try {
                // sdk.listModels(0, limit) now uses LocalModelRegistry internally
                models.addAll(sdk.listModels(0, limit));
            } catch (Exception ignored) {
            }

            models = dedupeAndSort(models, limit);
            if (runnableOnly) {
                models = models.stream()
                        .filter(this::isRunnableModel)
                        .toList();
            }

            if (models.isEmpty()) {
                System.out.println("No models found.");
                return;
            }

            if ("json".equalsIgnoreCase(format)) {
                printJson(models);
            } else {
                printTable(models);
            }
        } catch (Exception e) {
            System.err.println("Failed to list models: " + e.getMessage());
        }
    }

    private List<ModelInfo> dedupeAndSort(List<ModelInfo> models, int max) {
        Map<String, ModelInfo> unique = new LinkedHashMap<>();
        for (ModelInfo model : models) {
            if (model == null || model.getModelId() == null || model.getModelId().isBlank()) {
                continue;
            }
            unique.putIfAbsent(model.getModelId(), model);
        }
        List<ModelInfo> filtered = filterNamespaceShadowEntries(new ArrayList<>(unique.values()));
        List<ModelInfo> sorted = new ArrayList<>(filtered);
        sorted.sort(Comparator.comparing(
                (ModelInfo m) -> m.getUpdatedAt() != null ? m.getUpdatedAt() : Instant.EPOCH).reversed());
        if (sorted.size() > max) {
            return sorted.subList(0, max);
        }
        return sorted;
    }

    private List<ModelInfo> filterNamespaceShadowEntries(List<ModelInfo> models) {
        if (models.isEmpty()) {
            return models;
        }
        List<String> ids = models.stream()
                .map(ModelInfo::getModelId)
                .filter(id -> id != null && !id.isBlank())
                .toList();

        return models.stream()
                .filter(model -> {
                    String id = model.getModelId();
                    if (id == null || id.isBlank() || id.contains("/")) {
                        return true;
                    }
                    String prefix = id + "/";
                    return ids.stream().noneMatch(other -> other != null && other.startsWith(prefix));
                })
                .toList();
    }

    private void printTable(List<ModelInfo> models) {
        System.out.printf("%-30s %-12s %-10s %-20s%n", "NAME", "SIZE", "FORMAT", "MODIFIED");
        System.out.println("-".repeat(75));

        for (ModelInfo model : models) {
            String modified = model.getUpdatedAt() != null
                    ? model.getUpdatedAt().toString().substring(0, 10)
                    : "N/A";
            System.out.printf("%-30s %-12s %-10s %-20s%n",
                    truncate(model.getModelId(), 30),
                    model.getSizeFormatted(),
                    model.getFormat() != null ? model.getFormat() : "N/A",
                    modified);
        }
        System.out.printf("%n%d model(s) found%n", models.size());
    }

    private void printJson(List<ModelInfo> models) {
        System.out.println("[");
        for (int i = 0; i < models.size(); i++) {
            ModelInfo model = models.get(i);
            System.out.printf("  {\"modelId\": \"%s\", \"name\": \"%s\", \"size\": %d, \"format\": \"%s\"}%s%n",
                    model.getModelId(),
                    model.getName() != null ? model.getName() : model.getModelId(),
                    model.getSizeBytes() != null ? model.getSizeBytes() : 0,
                    model.getFormat() != null ? model.getFormat() : "",
                    i < models.size() - 1 ? "," : "");
        }
        System.out.println("]");
    }

    private String truncate(String str, int maxLen) {
        if (str == null)
            return "";
        return str.length() > maxLen ? str.substring(0, maxLen - 3) + "..." : str;
    }

    private boolean isRunnableModel(ModelInfo model) {
        if (model == null || model.getFormat() == null) {
            return false;
        }
        String format = model.getFormat().trim().toUpperCase(Locale.ROOT);
        return format.equals("GGUF")
                || format.equals("TORCHSCRIPT")
                || format.equals("ONNX")
                || format.equals("SAFETENSORS")
                || format.equals("PYTORCH")
                || format.equals("BIN");
    }
}
