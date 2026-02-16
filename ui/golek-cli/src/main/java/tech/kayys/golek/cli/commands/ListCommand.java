package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.model.ModelInfo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * List local models using GolekSdk.
 * Usage: golek list [--format table|json] [--limit N]
 */
@Dependent
@Unremovable
@Command(name = "list", description = "List available models")
public class ListCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Option(names = { "-f", "--format" }, description = "Output format: table, json", defaultValue = "table")
    String format;

    @Option(names = { "-l", "--limit" }, description = "Maximum models to list", defaultValue = "50")
    int limit;

    @Option(names = { "--runnable-only" }, description = "Show only models runnable in local Java runtime", defaultValue = "false")
    boolean runnableOnly;

    @Override
    public void run() {
        try {
            LocalModelIndex.refreshFromDisk();
            List<ModelInfo> models = new ArrayList<>();
            try {
                models.addAll(sdk.listModels(0, limit));
            } catch (Exception ignored) {
                // Fallback scan below will still list local model files.
            }
            models.addAll(discoverLocalModels(limit));
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

    private List<ModelInfo> discoverLocalModels(int max) {
        List<ModelInfo> out = new ArrayList<>();
        Path home = Path.of(System.getProperty("user.home"));
        Path root = home.resolve(".golek").resolve("models");
        if (!Files.isDirectory(root)) {
            return out;
        }

        Path gguf = root.resolve("gguf");
        Path torchscript = root.resolve("torchscript");
        Path djl = root.resolve("djl");

        scanModelDir(gguf, "gguf", out, max);
        scanModelDir(torchscript, "torchscript", out, max);
        scanDjlRepos(djl, out, max);
        return out;
    }

    private void scanModelDir(Path baseDir, String fallbackFormat, List<ModelInfo> out, int max) {
        if (!Files.isDirectory(baseDir) || out.size() >= max) {
            return;
        }
        try (var paths = Files.walk(baseDir, 4)) {
            paths
                    .filter(Files::isRegularFile)
                    .filter(p -> !p.getFileName().toString().startsWith("."))
                    .filter(p -> isLikelyModelFile(p, fallbackFormat))
                    .limit(Math.max(1, max - out.size()))
                    .forEach(p -> out.add(toModelInfo(baseDir, p, fallbackFormat)));
        } catch (Exception ignored) {
            // Best-effort listing
        }
    }

    private boolean isLikelyModelFile(Path path, String fallbackFormat) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".json") || name.endsWith(".txt") || name.endsWith(".md")) {
            return false;
        }
        if ("gguf".equalsIgnoreCase(fallbackFormat)) {
            // GGUF files are often extensionless in this project layout.
            return true;
        }
        return name.endsWith(".gguf")
                || name.endsWith(".safetensors")
                || name.endsWith(".safetensor")
                || name.endsWith(".pt")
                || name.endsWith(".pth")
                || name.endsWith(".bin")
                || !name.contains(".");
    }

    private ModelInfo toModelInfo(Path baseDir, Path file, String fallbackFormat) {
        String relative = baseDir.relativize(file).toString().replace("\\", "/");
        String modelId = relative;
        String format = detectFormat(file, fallbackFormat);
        Long size = 0L;
        Instant updatedAt = null;
        try {
            size = Files.size(file);
            updatedAt = Files.getLastModifiedTime(file).toInstant();
        } catch (Exception ignored) {
            // Best-effort fields
        }
        return ModelInfo.builder()
                .modelId(modelId)
                .name(file.getFileName().toString())
                .format(format)
                .sizeBytes(size)
                .updatedAt(updatedAt)
                .build();
    }

    private void scanDjlRepos(Path djlDir, List<ModelInfo> out, int max) {
        if (!Files.isDirectory(djlDir) || out.size() >= max) {
            return;
        }
        try (var dirs = Files.walk(djlDir, 3)) {
            dirs.filter(Files::isDirectory)
                    .filter(dir -> !dir.equals(djlDir))
                    .forEach(dir -> {
                        if (out.size() >= max) {
                            return;
                        }
                        ModelInfo info = toDjlRepoModelInfo(djlDir, dir);
                        if (info != null) {
                            out.add(info);
                        }
                    });
        } catch (Exception ignored) {
            // Best-effort listing
        }
    }

    private ModelInfo toDjlRepoModelInfo(Path baseDir, Path repoDir) {
        long totalSize = 0L;
        Instant latest = Instant.EPOCH;
        boolean hasWeights = false;
        boolean hasDirectRelevantFile = false;
        String detectedFormat = "djl";

        try (var files = Files.walk(repoDir, 4)) {
            for (Path file : (Iterable<Path>) files.filter(Files::isRegularFile)::iterator) {
                String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
                if (!isDjlRelevantFile(name)) {
                    continue;
                }
                if (repoDir.equals(file.getParent())) {
                    hasDirectRelevantFile = true;
                }
                try {
                    totalSize += Files.size(file);
                    Instant modified = Files.getLastModifiedTime(file).toInstant();
                    if (modified.isAfter(latest)) {
                        latest = modified;
                    }
                } catch (Exception ignored) {
                    // best effort
                }
                if (isWeightFileName(name)) {
                    hasWeights = true;
                    if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
                        detectedFormat = "safetensors";
                    } else if (name.endsWith(".pt") || name.endsWith(".pth")) {
                        detectedFormat = "torchscript";
                    } else if (name.endsWith(".bin") && !"safetensors".equals(detectedFormat)) {
                        detectedFormat = "bin";
                    }
                }
            }
        } catch (Exception ignored) {
            return null;
        }

        // Ignore namespace/container directories that only have nested model dirs.
        if (!hasWeights || !hasDirectRelevantFile) {
            return null;
        }

        String modelId = baseDir.relativize(repoDir).toString().replace("\\", "/");
        return ModelInfo.builder()
                .modelId(modelId)
                .name(repoDir.getFileName().toString())
                .format(detectedFormat)
                .sizeBytes(totalSize)
                .updatedAt(latest.equals(Instant.EPOCH) ? null : latest)
                .build();
    }

    private boolean isDjlRelevantFile(String lowerName) {
        return isWeightFileName(lowerName)
                || lowerName.endsWith(".json")
                || lowerName.endsWith(".txt")
                || lowerName.endsWith(".model")
                || lowerName.endsWith(".tiktoken")
                || lowerName.endsWith(".spm");
    }

    private boolean isWeightFileName(String lowerName) {
        return lowerName.endsWith(".safetensors")
                || lowerName.endsWith(".safetensor")
                || lowerName.endsWith(".bin")
                || lowerName.endsWith(".pt")
                || lowerName.endsWith(".pth");
    }

    private String detectFormat(Path file, String fallbackFormat) {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf")) {
            return "gguf";
        }
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return "safetensors";
        }
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            return "torchscript";
        }
        if (name.endsWith(".bin")) {
            return "bin";
        }
        return fallbackFormat;
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
