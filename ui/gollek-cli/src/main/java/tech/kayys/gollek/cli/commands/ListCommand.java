package tech.kayys.gollek.cli.commands;

import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;

import tech.kayys.gollek.sdk.model.ModelListRequest;

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

    @Option(names = { "-t", "--task-type",
            "--category" }, description = "Filter by task category: text, vision, tts, stt, ocr, multimodal, embedding, timeseries, 3d")
    String filterTaskType;

    @Override
    public void run() {
        try {
            ModelListRequest request = ModelListRequest.builder()
                    .limit(limit)
                    .runnableOnly(runnableOnly)
                    .dedupe(true)
                    .sort(true)
                    .build();

            List<ModelInfo> models = mergeModels(sdk.listModels(request), fromLocalIndex());

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

    private List<ModelInfo> mergeModels(List<ModelInfo> sdkModels, List<ModelInfo> indexedModels) {
        Map<String, ModelInfo> merged = new LinkedHashMap<>();
        if (sdkModels != null) {
            for (ModelInfo model : sdkModels) {
                putBest(merged, model);
            }
        }
        if (indexedModels != null) {
            for (ModelInfo model : indexedModels) {
                putBest(merged, model);
            }
        }

        List<ModelInfo> models = new ArrayList<>(merged.values());
        models.sort(Comparator.comparing((ModelInfo m) -> m.getUpdatedAt() != null ? m.getUpdatedAt() : Instant.EPOCH)
                .reversed());
        if (models.size() > limit) {
            return new ArrayList<>(models.subList(0, limit));
        }
        return models;
    }

    private void putBest(Map<String, ModelInfo> merged, ModelInfo candidate) {
        if (candidate == null) {
            return;
        }
        String key = dedupeKey(candidate);
        ModelInfo existing = merged.get(key);
        if (existing == null || isRicher(candidate, existing)) {
            merged.put(key, candidate);
        }
    }

    private String dedupeKey(ModelInfo model) {
        String modelId = model.getModelId();
        String format = model.getFormat() != null ? model.getFormat().trim().toLowerCase(Locale.ROOT) : "";
        String base;
        if (modelId != null && !modelId.isBlank()) {
            String normalized = modelId.trim();
            if (normalized.startsWith("hf:")) {
                normalized = normalized.substring("hf:".length()).trim();
            } else if (normalized.startsWith("huggingface:")) {
                normalized = normalized.substring("huggingface:".length()).trim();
            }
            base = normalized.replace("\\", "/").toLowerCase(Locale.ROOT);
            return base + "|" + format;
        }
        String name = model.getName();
        if (name != null && !name.isBlank()) {
            base = name.toLowerCase(Locale.ROOT);
            return base + "|" + format;
        }
        String shortId = model.getShortId();
        base = shortId != null ? shortId.toLowerCase(Locale.ROOT) : "";
        return base + "|" + format;
    }

    private boolean isRicher(ModelInfo candidate, ModelInfo existing) {
        int candidateScore = richnessScore(candidate);
        int existingScore = richnessScore(existing);
        if (candidateScore != existingScore) {
            return candidateScore > existingScore;
        }
        Instant candidateUpdated = candidate.getUpdatedAt() != null ? candidate.getUpdatedAt() : Instant.EPOCH;
        Instant existingUpdated = existing.getUpdatedAt() != null ? existing.getUpdatedAt() : Instant.EPOCH;
        return candidateUpdated.isAfter(existingUpdated);
    }

    private int richnessScore(ModelInfo model) {
        int score = 0;
        if (model.getFormat() != null && !model.getFormat().isBlank())
            score++;
        if (model.getArchitecture() != null && !model.getArchitecture().isBlank())
            score++;
        if (model.getSizeBytes() != null && model.getSizeBytes() > 0)
            score++;
        if (model.getUpdatedAt() != null)
            score++;
        if (model.getShortId() != null && !model.getShortId().isBlank())
            score++;
        return score;
    }

    private List<ModelInfo> fromLocalIndex() {
        List<LocalModelIndex.Entry> entries = LocalModelIndex.refreshFromDisk();
        List<ModelInfo> models = new ArrayList<>();
        for (LocalModelIndex.Entry e : entries) {
            if (e == null) {
                continue;
            }
            if (runnableOnly && !e.runnable) {
                continue;
            }
            if (filterTaskType != null && !filterTaskType.isBlank()) {
                String wanted = filterTaskType.trim().toLowerCase(Locale.ROOT);
                String actual = e.taskType != null ? e.taskType.trim().toLowerCase(Locale.ROOT) : "";
                if (!actual.equals(wanted)) {
                    continue;
                }
            }
            String modelId = (e.id != null && !e.id.isBlank()) ? e.id : e.path;
            if (modelId == null || modelId.isBlank()) {
                continue;
            }
            models.add(ModelInfo.builder()
                    .modelId(modelId)
                    .shortId(e.shortId)
                    .name(e.name != null ? e.name : modelId)
                    .architecture(e.architecture)
                    .parameterCount(e.parameterCount)
                    .format(e.format)
                    .sizeBytes(e.sizeBytes)
                    .updatedAt(parseInstant(e.updatedAt))
                    .metadata(e.taskType != null
                            ? Map.of("taskType", e.taskType)
                            : Map.of())
                    .build());
        }
        models.sort(Comparator.comparing((ModelInfo m) -> m.getUpdatedAt() != null ? m.getUpdatedAt() : Instant.EPOCH)
                .reversed());
        if (models.size() > limit) {
            return new ArrayList<>(models.subList(0, limit));
        }
        return models;
    }

    private Instant parseInstant(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        try {
            return Instant.parse(value);
        } catch (Exception ignored) {
            return null;
        }
    }

    private void printTable(List<ModelInfo> models) {
        final String ANSI_RESET = "\u001B[0m";
        final String ANSI_CYAN = "\u001B[36m";
        final String ANSI_YELLOW = "\u001B[33m";
        final String ANSI_GREEN = "\u001B[32m";
        final String ANSI_MAGENTA = "\u001B[35m";
        final String ANSI_BOLD = "\u001B[1;37m";
        final String ANSI_GRAY = "\u001B[90m";
        final String ANSI_BLUE = "\u001B[34m";

        // Header
        System.out.printf(ANSI_BOLD + "%-7s %-14s %-26s %-12s %-10s %-10s %-12s %-10s" + ANSI_RESET + "%n",
                "ID", "GROUP", "NAME", "ARCH", "FORMAT", "TASK", "SIZE", "MODIFIED");
        System.out.println(ANSI_GRAY + "─".repeat(107) + ANSI_RESET);

        for (ModelInfo model : models) {
            // Short ID
            String id = model.getShortId();
            if (id == null || id.isBlank() || id.equalsIgnoreCase("n/a")) {
                id = tech.kayys.alkhawarizm.spi.model.ModelUtils.generateShortId(model.getModelId());
            }

            // Group from modelId (e.g., "Qwen/Qwen2.5-..." → "Qwen")
            String group = "";
            String displayName = model.getName() != null ? model.getName() : model.getModelId();
            String modelId = model.getModelId();
            if (modelId != null && modelId.contains("/")) {
                int slash = modelId.indexOf('/');
                group = modelId.substring(0, slash);
                // Strip group prefix from displayName if still present
                String withUnder = group + "_";
                String withDouble = group + "__";
                if (displayName.startsWith(withDouble)) {
                    displayName = displayName.substring(withDouble.length());
                } else if (displayName.startsWith(withUnder)) {
                    displayName = displayName.substring(withUnder.length());
                }
            }

            String arch = model.getArchitecture() != null ? model.getArchitecture() : "unknown";
            String modified = model.getUpdatedAt() != null
                    ? model.getUpdatedAt().toString().substring(0, 10)
                    : "N/A";

            String fmtStr = model.getFormat() != null ? model.getFormat() : "N/A";
            String fmtColor = switch (fmtStr.toUpperCase()) {
                case "GGUF" -> ANSI_CYAN;
                case "SAFETENSORS" -> ANSI_YELLOW;
                case "LITERT" -> ANSI_GREEN;
                case "ONNX" -> ANSI_MAGENTA;
                default -> "";
            };

            // Task-type column with colour coding
            String taskStr = taskTypeOf(model);
            String taskColor = switch (taskStr.toLowerCase(Locale.ROOT)) {
                case "vision", "ocr" -> ANSI_MAGENTA;
                case "tts" -> ANSI_GREEN;
                case "stt" -> ANSI_BLUE;
                case "multimodal" -> ANSI_CYAN;
                case "embedding" -> ANSI_GRAY;
                case "timeseries" -> ANSI_YELLOW;
                default -> ""; // text → no colour
            };

            System.out.printf("%-7s %-14s %-26s %-12s %s%-10s%s %s%-10s%s %-12s %-10s%n",
                    ANSI_YELLOW + id + ANSI_RESET,
                    truncate(group, 14),
                    truncate(displayName, 26),
                    truncate(arch, 12),
                    fmtColor,
                    truncate(fmtStr, 10),
                    ANSI_RESET,
                    taskColor,
                    truncate(taskStr, 10),
                    ANSI_RESET,
                    CLIUtils.formatSize(model.getSizeBytes() != null ? model.getSizeBytes() : 0),
                    modified);
        }
        System.out.printf(ANSI_BOLD + "%n%d model(s) found" + ANSI_RESET + "%n", models.size());
    }

    private void printJson(List<ModelInfo> models) {
        System.out.println("[");
        for (int i = 0; i < models.size(); i++) {
            ModelInfo model = models.get(i);
            String modelId = model.getModelId() != null ? model.getModelId() : "";
            String group = "";
            String name = model.getName() != null ? model.getName() : modelId;
            if (modelId.contains("/")) {
                group = modelId.substring(0, modelId.indexOf('/'));
            }
            String shortId = model.getShortId();
            if (shortId == null || shortId.isBlank()) {
                shortId = tech.kayys.alkhawarizm.spi.model.ModelUtils.generateShortId(modelId);
            }
            String task = taskTypeOf(model);
            System.out.printf(
                    "  {\"id\": \"%s\", \"shortId\": \"%s\", \"group\": \"%s\", \"name\": \"%s\", " +
                            "\"modelId\": \"%s\", \"format\": \"%s\", \"taskType\": \"%s\", \"size\": %d}%s%n",
                    model.getModelId(),
                    shortId,
                    group,
                    name,
                    modelId,
                    model.getFormat() != null ? model.getFormat() : "",
                    task,
                    model.getSizeBytes() != null ? model.getSizeBytes() : 0,
                    i < models.size() - 1 ? "," : "");
        }
        System.out.println("]");
    }

    /**
     * Extracts task type from the model's metadata map. Defaults to "text" if
     * absent.
     */
    private static String taskTypeOf(ModelInfo model) {
        if (model == null)
            return "text";
        Map<String, Object> meta = model.getMetadata();
        if (meta != null) {
            Object v = meta.get("taskType");
            if (v instanceof String s && !s.isBlank()) {
                return s;
            }
        }
        return "text";
    }

    private String truncate(String str, int maxLen) {
        if (str == null)
            return "";
        return str.length() > maxLen ? str.substring(0, maxLen - 3) + "..." : str;
    }
}
