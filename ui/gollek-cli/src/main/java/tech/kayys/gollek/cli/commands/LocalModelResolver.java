package tech.kayys.gollek.cli.commands;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.ModelInfo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
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

        // 1. Check SDK/Registry for first-class resolution
        for (String candidate : sdkCandidates(requestedId)) {
            try {
                Optional<ModelInfo> info = sdk.getModelInfo(candidate);
                if (info.isPresent()) {
                    return Optional
                            .of(new ResolvedModel(candidate, info.get(), extractPath(info.get()).orElse(null), true));
                }
            } catch (Exception ignored) {
            }
        }

        // 2. Direct file check as fallback
        Path input = Path.of(requestedId);
        if (Files.isRegularFile(input)) {
            ModelInfo info = toModelInfo(requestedId, input.toAbsolutePath());
            return Optional.of(new ResolvedModel(requestedId, info, input.toAbsolutePath(), false));
        }

        return Optional.empty();
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

    private static ModelInfo toModelInfo(String id, Path file) {
        Long size = null;
        Instant updated = null;
        try {
            size = Files.size(file);
            updated = Files.getLastModifiedTime(file).toInstant();
        } catch (Exception ignored) {
        }

        return ModelInfo.builder()
                .modelId(id)
                .name(file.getFileName().toString())
                .sizeBytes(size)
                .updatedAt(updated)
                .metadata(Map.of("path", file.toAbsolutePath().toString()))
                .build();
    }
}
