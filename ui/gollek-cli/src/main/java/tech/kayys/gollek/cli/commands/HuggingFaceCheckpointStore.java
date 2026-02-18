package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.inject.Instance;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.sdk.core.model.PullProgress;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Consumer;

final class HuggingFaceCheckpointStore {

    private HuggingFaceCheckpointStore() {
    }

    record StoreResult(Path rootDir, int fileCount, boolean hasWeights) {
    }

    static boolean shouldStoreOnPullFailure(String reason) {
        if (reason == null || reason.isBlank()) {
            return false;
        }
        String detail = reason.toLowerCase(Locale.ROOT);
        return detail.contains("checkpoint files only and auto-conversion is disabled")
                || detail.contains("no gguf artifact found and auto-conversion is disabled")
                || detail.contains("repository has pytorch checkpoints only");
    }

    static Optional<StoreResult> storeCheckpointArtifacts(
            Instance<HuggingFaceClient> hfClientInstance,
            String modelSpec,
            Consumer<PullProgress> progressCallback) {
        try {
            if (hfClientInstance == null || !hfClientInstance.isResolvable()) {
                return Optional.empty();
            }
            if (modelSpec == null || modelSpec.isBlank()) {
                return Optional.empty();
            }
            String repo = modelSpec.startsWith("hf:") ? modelSpec.substring(3) : modelSpec;
            if (!repo.contains("/")) {
                return Optional.empty();
            }

            HuggingFaceClient client = hfClientInstance.get();
            List<String> files = client.listFiles(repo);
            List<String> selected = files.stream()
                    .filter(HuggingFaceCheckpointStore::isCheckpointRelevantFile)
                    .toList();
            if (selected.isEmpty()) {
                return Optional.empty();
            }

            Path root = Path.of(System.getProperty("user.home"), ".gollek", "models", "djl", repo);
            Files.createDirectories(root);

            boolean hasWeights = false;
            int downloaded = 0;
            for (String file : selected) {
                if (isWeightFile(file)) {
                    hasWeights = true;
                }
                Path target = root.resolve(file);
                if (target.getParent() != null) {
                    Files.createDirectories(target.getParent());
                }
                client.downloadFile(repo, file, target, (completed, total, progress) -> {
                    if (progressCallback != null) {
                        progressCallback.accept(PullProgress.of("Downloading checkpoint artifacts", null, total, completed));
                    }
                });
                downloaded++;
            }

            if (progressCallback != null) {
                progressCallback.accept(PullProgress.of("Checkpoint artifacts stored locally"));
            }
            return Optional.of(new StoreResult(root, downloaded, hasWeights));
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean isCheckpointRelevantFile(String file) {
        if (file == null || file.isBlank() || file.startsWith(".")) {
            return false;
        }
        String lower = file.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".md") || lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return false;
        }
        return lower.endsWith(".safetensors")
                || lower.endsWith(".safetensors.index.json")
                || lower.endsWith(".bin")
                || lower.endsWith(".pt")
                || lower.endsWith(".pth")
                || lower.endsWith(".json")
                || lower.endsWith(".txt")
                || lower.endsWith(".model")
                || lower.endsWith(".tiktoken")
                || lower.endsWith(".spm");
    }

    private static boolean isWeightFile(String file) {
        String lower = file.toLowerCase(Locale.ROOT);
        return lower.endsWith(".safetensors")
                || lower.endsWith(".bin")
                || lower.endsWith(".pt")
                || lower.endsWith(".pth");
    }
}
