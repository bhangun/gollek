package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelInfo;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.cli.util.CLIUtils;

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
        // ANSI escape codes for coloring
        final String ANSI_RESET = "\u001B[0m";
        final String ANSI_CYAN = "\u001B[36m";
        final String ANSI_YELLOW = "\u001B[33m";
        final String ANSI_GREEN = "\u001B[32m";
        final String ANSI_WHITE_BOLD = "\u001B[1;37m";
        final String ANSI_GRAY = "\u001B[90m";

        System.out.printf(ANSI_WHITE_BOLD + "%-7s %-32s %-12s %-10s %-12s %-10s" + ANSI_RESET + "%n", 
                "ID", "NAME", "ARCH", "FORMAT", "SIZE", "MODIFIED");
        System.out.println(ANSI_GRAY + "─".repeat(85) + ANSI_RESET);

        for (ModelInfo model : models) {
            String id = model.getShortId() != null ? model.getShortId() : "n/a";
            String arch = model.getArchitecture() != null ? model.getArchitecture() : "unknown";
            String modified = model.getUpdatedAt() != null
                    ? model.getUpdatedAt().toString().substring(0, 10)
                    : "N/A";
            
            String formatColor = switch (model.getFormat() != null ? model.getFormat().toUpperCase() : "") {
                case "GGUF" -> ANSI_CYAN;
                case "SAFETENSORS" -> ANSI_YELLOW;
                case "LITERT" -> ANSI_GREEN;
                default -> "";
            };

            System.out.printf("%-7s %-32s %-12s %s%-10s%s %-12s %-10s%n",
                    ANSI_YELLOW + id + ANSI_RESET,
                    truncate(model.getName() != null ? model.getName() : model.getModelId(), 32),
                    truncate(arch, 12),
                    formatColor,
                    truncate(model.getFormat() != null ? model.getFormat() : "N/A", 10),
                    ANSI_RESET,
                    CLIUtils.formatSize(model.getSizeBytes() != null ? model.getSizeBytes() : 0),
                    modified);
        }
        System.out.printf(ANSI_WHITE_BOLD + "%n%d model(s) found" + ANSI_RESET + "%n", models.size());
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
