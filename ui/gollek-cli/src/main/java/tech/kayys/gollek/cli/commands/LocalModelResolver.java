package tech.kayys.gollek.cli.commands;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.context.RequestContext;

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

    static Optional<ResolvedModel> resolve(GollekSdk sdk, String requestedId, String branch, String format) {
        if (requestedId == null || requestedId.isBlank()) {
            return Optional.empty();
        }

        // 1. Check SDK/Registry for first-class resolution
        java.util.List<ModelInfo> allModels = null;
        try {
            allModels = sdk.listModels();
        } catch (Exception ignored) {
        }

        // Short ID check (6-character hex)
        if (requestedId.length() == 6 && allModels != null) {
            final java.util.List<ModelInfo> models = allModels;
            Optional<ModelInfo> byShortId = models.stream()
                    .filter(m -> requestedId.equalsIgnoreCase(tech.kayys.gollek.spi.model.ModelUtils.generateShortId(m.getModelId())))
                    .findFirst();
            if (byShortId.isPresent()) {
                ModelInfo mi = byShortId.get();
                return Optional.of(new ResolvedModel(mi.getModelId(), mi, extractPath(mi).orElse(null), true));
            }
        }

        for (String candidate : sdkCandidates(requestedId, branch)) {
            try {
                if (format != null && !format.isBlank()) {
                    // When format matters, we must look for the exact matching variant
                    // since getModelInfo() only returns the first one it finds.
                    Optional<ModelInfo> matchingModel = sdk.listModels().stream()
                            .filter(m -> (candidate.equals(m.getModelId()) || candidate.equals(m.getName())) && formatsMatch(format, m.getFormat()))
                            .findFirst();
                    if (matchingModel.isPresent()) {
                         return Optional
                            .of(new ResolvedModel(candidate, matchingModel.get(), extractPath(matchingModel.get()).orElse(null), true));
                    }
                    continue; // Variant not found with this candidate name, try next
                }

                Optional<ModelInfo> info = sdk.getModelInfo(candidate);
                if (info.isPresent()) {
                    return Optional
                            .of(new ResolvedModel(candidate, info.get(), extractPath(info.get()).orElse(null), true));
                }
            } catch (Exception ignored) {
            }
        }

        // 2. Direct file/directory check as fallback
        Path input = Path.of(requestedId);
        if (Files.isRegularFile(input)) {
            ModelInfo info = toModelInfo(requestedId, input.toAbsolutePath());
            return Optional.of(new ResolvedModel(requestedId, info, input.toAbsolutePath(), false));
        } else if (Files.isDirectory(input) && Files.exists(input.resolve("model_index.json"))) {
            // It's a pipeline directory (e.g. Stable Diffusion)
            ModelInfo info = toModelInfo(requestedId, input.toAbsolutePath());
            return Optional.of(new ResolvedModel(requestedId, info, input.toAbsolutePath(), false));
        }

        return Optional.empty();
    }

    static Optional<ResolvedModel> resolve(GollekSdk sdk, String requestedId) {
        return resolve(sdk, requestedId, null, null);
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

    private static java.util.List<String> sdkCandidates(String id, String branch) {
        String normalized = id.startsWith("hf:") ? id.substring(3) : id;
        java.util.LinkedHashSet<String> candidates = new java.util.LinkedHashSet<>();
        
        // 1. Manifest-style ID (new system): org__model--branch
        String manifestBase = normalized.replace("/", "__");
        if (branch != null && !branch.trim().isEmpty()
                && !branch.trim().equalsIgnoreCase("main") && !branch.trim().equalsIgnoreCase("master")) {
            String manifestBranch = manifestBase + "--" + branch.trim();
            candidates.add(manifestBranch);
        }
        candidates.add(manifestBase);

        // 2. Legacy-style IDs (backward compat): org/model-branch
        if (branch != null && !branch.trim().isEmpty() && !branch.trim().equalsIgnoreCase("main")) {
            String branchSuffix = "-" + branch.trim();
            candidates.add(id + branchSuffix);
            candidates.add(normalized + branchSuffix);
            if (!id.startsWith("hf:") && id.contains("/")) {
                candidates.add("hf:" + id + branchSuffix);
            }
        }

        // 3. Original ID (main branch)
        candidates.add(id);
        candidates.add(normalized);
        if (!id.startsWith("hf:") && id.contains("/")) {
            candidates.add("hf:" + id);
        }
        
        // 4. GGUF fallback
        candidates.add(id + "-GGUF");
        candidates.add(normalized + "-GGUF");
        if (!id.startsWith("hf:") && id.contains("/")) {
            candidates.add("hf:" + id + "-GGUF");
        }
        return java.util.List.copyOf(candidates);
    }

    private static boolean formatsMatch(String requested, String actual) {
        if (requested == null || actual == null) return false;
        String r = normalizeFormat(requested);
        String a = normalizeFormat(actual);
        return r.equals(a);
    }

    private static String normalizeFormat(String format) {
        if (format == null) return "";
        String n = format.trim().toUpperCase();
        if (n.equals("SAFETENSORS")) return "SAFETENSOR";
        if (n.equals("PYTORCH") || n.equals("TORCH")) return "TORCHSCRIPT";
        if (n.equals("TFLITE") || n.equals("TASK")) return "LITERT";
        return n;
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
                .requestContext(RequestContext.of("community", "community"))
                .metadata(Map.of("path", file.toAbsolutePath().toString()))
                .build();
    }
}
