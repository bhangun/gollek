package tech.kayys.gollek.cli.commands;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.model.ModelInfo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

final class LocalModelResolver {

    private LocalModelResolver() {
    }

    record ResolvedModel(String modelId, ModelInfo info, Path localPath, boolean fromSdk) {
    }

    static Optional<ResolvedModel> resolve(GollekSdk sdk, String requestedId) {
        if (requestedId == null || requestedId.isBlank()) {
            return Optional.empty();
        }

        for (String candidate : sdkCandidates(requestedId)) {
            try {
                Optional<ModelInfo> info = sdk.getModelInfo(candidate);
                if (info.isPresent()) {
                    return Optional.of(new ResolvedModel(candidate, info.get(), null, true));
                }
            } catch (Exception ignored) {
                // keep trying candidates
            }
        }

        Optional<ModelInfo> local = findLocalModel(requestedId);
        if (local.isEmpty()) {
            return Optional.empty();
        }
        Path localPath = extractPath(local.get()).orElse(null);
        String resolvedModelId = localPath != null ? localPath.toString() : requestedId;
        return Optional.of(new ResolvedModel(resolvedModelId, local.get(), localPath, false));
    }

    static Optional<ModelInfo> findLocalModel(String id) {
        if (id == null || id.isBlank()) {
            return Optional.empty();
        }

        Path input = Path.of(id);
        if (Files.isRegularFile(input)) {
            return Optional.of(toModelInfo(id, input.toAbsolutePath(), detectFormat(input, "file")));
        }

        Path root = Path.of(System.getProperty("user.home"), ".gollek", "models");
        if (!Files.isDirectory(root)) {
            return Optional.empty();
        }

        Optional<ModelInfo> found = findInBase(root.resolve("gguf"), id, "gguf");
        if (found.isPresent()) {
            return found;
        }
        found = findInBase(root.resolve("torchscript"), id, "torchscript");
        if (found.isPresent()) {
            return found;
        }
        return findInBase(root.resolve("djl"), id, "djl");
    }

    static Optional<Path> extractPath(ModelInfo info) {
        if (info == null || info.getMetadata() == null) {
            return Optional.empty();
        }
        Object raw = info.getMetadata().get("path");
        if (raw == null) {
            return Optional.empty();
        }
        String pathString = String.valueOf(raw).trim();
        if (pathString.isBlank()) {
            return Optional.empty();
        }
        try {
            if (pathString.startsWith("file:")) {
                return Optional.of(Paths.get(java.net.URI.create(pathString)));
            }
            return Optional.of(Path.of(pathString));
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static java.util.List<String> sdkCandidates(String id) {
        String normalized = id.startsWith("hf:") ? id.substring(3) : id;
        java.util.LinkedHashSet<String> candidates = new java.util.LinkedHashSet<>();
        candidates.add(id);
        candidates.add(normalized);
        candidates.add(id + "-GGUF");
        candidates.add(normalized + "-GGUF");
        if (!id.startsWith("hf:") && id.contains("/")) {
            candidates.add("hf:" + id);
            candidates.add("hf:" + id + "-GGUF");
        }
        return java.util.List.copyOf(candidates);
    }

    private static Optional<ModelInfo> findInBase(Path base, String id, String fallbackFormat) {
        if (!Files.isDirectory(base)) {
            return Optional.empty();
        }

        Path direct = base.resolve(id);
        if (Files.isRegularFile(direct)) {
            return Optional.of(toModelInfo(id, direct, detectFormat(direct, fallbackFormat)));
        }
        if (Files.isDirectory(direct)) {
            Optional<Path> primary = pickPrimaryModelFile(direct);
            if (primary.isPresent()) {
                return Optional.of(toModelInfo(id, primary.get(), detectFormat(primary.get(), fallbackFormat)));
            }
        }

        String normalized = id.replace("/", "_");
        Path normalizedPath = base.resolve(normalized);
        if (Files.isRegularFile(normalizedPath)) {
            return Optional.of(toModelInfo(id, normalizedPath, detectFormat(normalizedPath, fallbackFormat)));
        }
        if (Files.isDirectory(normalizedPath)) {
            Optional<Path> primary = pickPrimaryModelFile(normalizedPath);
            if (primary.isPresent()) {
                return Optional.of(toModelInfo(id, primary.get(), detectFormat(primary.get(), fallbackFormat)));
            }
        }

        String[] exts = { ".gguf", ".safetensors", ".safetensor", ".pt", ".pth", ".bin" };
        for (String ext : exts) {
            Path candidate = base.resolve(id + ext);
            if (Files.isRegularFile(candidate)) {
                return Optional.of(toModelInfo(id, candidate, detectFormat(candidate, fallbackFormat)));
            }
            Path normalizedCandidate = base.resolve(normalized + ext);
            if (Files.isRegularFile(normalizedCandidate)) {
                return Optional.of(toModelInfo(id, normalizedCandidate, detectFormat(normalizedCandidate, fallbackFormat)));
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> pickPrimaryModelFile(Path dir) {
        try (var files = Files.walk(dir, 2)) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        return name.endsWith(".gguf")
                                || name.endsWith(".safetensors")
                                || name.endsWith(".safetensor")
                                || name.endsWith(".pt")
                                || name.endsWith(".pth")
                                || name.endsWith(".bin");
                    })
                    .sorted((a, b) -> Long.compare(filePriority(b), filePriority(a)))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static long filePriority(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".gguf")) {
            return 50;
        }
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return 40;
        }
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            return 30;
        }
        if (name.endsWith(".bin")) {
            return 20;
        }
        return 0;
    }

    private static ModelInfo toModelInfo(String id, Path file, String format) {
        Long size = null;
        Instant updated = null;
        try {
            size = Files.size(file);
            updated = Files.getLastModifiedTime(file).toInstant();
        } catch (Exception ignored) {
            // best effort
        }
        return ModelInfo.builder()
                .modelId(id)
                .name(file.getFileName().toString())
                .format(format)
                .sizeBytes(size)
                .updatedAt(updated)
                .metadata(Map.of("path", file.toAbsolutePath().toString()))
                .build();
    }

    private static String detectFormat(Path file, String fallbackFormat) {
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
}
