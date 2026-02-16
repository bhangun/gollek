package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.golek.sdk.core.GolekSdk;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

@Dependent
@Unremovable
@Command(name = "delete", aliases = { "remove", "rm" }, description = "Delete/remove a local model by id, name, or path")
public class DeleteCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Parameters(index = "0", description = "Model id/name/path to delete")
    String modelRef;

    @Option(names = { "-y", "--yes" }, description = "Skip confirmation")
    boolean assumeYes;

    @Option(names = { "--all-matches" }, description = "Delete all matching local files/directories")
    boolean allMatches;

    @Override
    public void run() {
        if (modelRef == null || modelRef.isBlank()) {
            System.err.println("Model reference is required.");
            return;
        }
        LocalModelIndex.refreshFromDisk();
        String ref = modelRef.trim();

        // Fast path: let SDK remove registered models by id.
        boolean sdkAttempted = false;
        try {
            sdkAttempted = true;
            sdk.deleteModel(ref);
            List<Path> postDeleteTargets = resolveTargets(ref);
            if (postDeleteTargets.isEmpty()) {
                System.out.println("Deleted via SDK: " + ref);
                return;
            }
            // Continue with filesystem cleanup when SDK operation did not remove local artifacts.
        } catch (Exception ignored) {
            // fallback to filesystem matching
        }

        List<Path> targets = resolveTargets(ref);
        if (targets.isEmpty()) {
            if (sdkAttempted) {
                System.out.println("Delete request accepted by SDK: " + ref);
                return;
            }
            System.err.println("Model not found: " + modelRef);
            return;
        }

        if (targets.size() > 1 && !allMatches) {
            System.err.println("Multiple matches found:");
            for (Path p : targets) {
                System.err.println("  - " + p);
            }
            System.err.println("Use --all-matches or specify a more exact id/path.");
            return;
        }

        List<Path> toDelete = allMatches ? targets : List.of(targets.get(0));
        if (!assumeYes && !confirmDeletion(toDelete)) {
            System.out.println("Delete cancelled.");
            return;
        }

        int deleted = 0;
        for (Path target : toDelete) {
            if (deletePath(target)) {
                deleted++;
                System.out.println("Deleted: " + target);
            }
        }
        if (deleted == 0) {
            System.err.println("No model deleted.");
        } else {
            LocalModelIndex.refreshFromDisk();
        }
    }

    private List<Path> resolveTargets(String ref) {
        List<Path> targets = new ArrayList<>();
        Path direct = Path.of(ref);
        if (Files.exists(direct)) {
            targets.add(direct.toAbsolutePath().normalize());
            return dedupe(targets);
        }

        // First try SDK deletion path resolution by id.
        try {
            var info = sdk.getModelInfo(ref);
            if (info.isPresent()) {
                LocalModelResolver.extractPath(info.get()).ifPresent(p -> targets.add(p.toAbsolutePath().normalize()));
            }
        } catch (Exception ignored) {
            // fallback local scan below
        }

        LocalModelIndex.find(ref)
                .flatMap(e -> {
                    try {
                        if (e.path != null && !e.path.isBlank()) {
                            return Optional.of(Path.of(e.path).toAbsolutePath().normalize());
                        }
                    } catch (Exception ignored) {
                        // fall through
                    }
                    return Optional.empty();
                })
                .ifPresent(targets::add);

        Path modelsRoot = Path.of(System.getProperty("user.home"), ".golek", "models");
        if (!Files.isDirectory(modelsRoot)) {
            return dedupe(targets);
        }
        Path gguf = modelsRoot.resolve("gguf");
        Path torchscript = modelsRoot.resolve("torchscript");
        Path djl = modelsRoot.resolve("djl");

        targets.addAll(findInBase(gguf, ref));
        targets.addAll(findInBase(torchscript, ref));
        targets.addAll(findInBase(djl, ref));

        // Filename-only fallback search.
        String lowered = ref.toLowerCase(Locale.ROOT);
        for (Path base : List.of(gguf, torchscript, djl)) {
            if (!Files.isDirectory(base)) {
                continue;
            }
            try (var stream = Files.walk(base, 5)) {
                stream.filter(Files::isRegularFile)
                        .filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).equals(lowered))
                        .forEach(p -> targets.add(p.toAbsolutePath().normalize()));
            } catch (Exception ignored) {
                // best effort
            }
        }
        return dedupe(targets);
    }

    private List<Path> findInBase(Path base, String ref) {
        if (!Files.isDirectory(base)) {
            return List.of();
        }
        List<Path> out = new ArrayList<>();
        Path direct = base.resolve(ref);
        if (Files.exists(direct)) {
            out.add(direct.toAbsolutePath().normalize());
        }

        String normalized = ref.replace("/", "_");
        Path normalizedPath = base.resolve(normalized);
        if (Files.exists(normalizedPath)) {
            out.add(normalizedPath.toAbsolutePath().normalize());
        }

        String[] exts = { ".gguf", ".safetensors", ".safetensor", ".pt", ".pth", ".bin" };
        for (String ext : exts) {
            Path candidate = base.resolve(ref + ext);
            if (Files.exists(candidate)) {
                out.add(candidate.toAbsolutePath().normalize());
            }
            Path normalizedCandidate = base.resolve(normalized + ext);
            if (Files.exists(normalizedCandidate)) {
                out.add(normalizedCandidate.toAbsolutePath().normalize());
            }
        }
        return out;
    }

    private boolean confirmDeletion(List<Path> targets) {
        java.io.Console console = System.console();
        if (console == null) {
            return false;
        }
        if (targets.size() == 1) {
            String answer = console.readLine("Delete '%s'? [y/N]: ", targets.get(0));
            return answer != null && answer.trim().equalsIgnoreCase("y");
        }
        System.out.println("Delete " + targets.size() + " targets:");
        for (Path p : targets) {
            System.out.println("  - " + p);
        }
        String answer = console.readLine("Continue? [y/N]: ");
        return answer != null && answer.trim().equalsIgnoreCase("y");
    }

    private boolean deletePath(Path target) {
        try {
            if (Files.isDirectory(target)) {
                try (var paths = Files.walk(target)) {
                    paths.sorted(Comparator.reverseOrder())
                            .forEach(p -> {
                                try {
                                    Files.deleteIfExists(p);
                                } catch (Exception ignored) {
                                    // handled by post-check
                                }
                            });
                }
                return !Files.exists(target);
            }
            Files.deleteIfExists(target);
            return !Files.exists(target);
        } catch (Exception e) {
            System.err.println("Failed to delete " + target + ": " + e.getMessage());
            return false;
        }
    }

    private List<Path> dedupe(List<Path> paths) {
        return paths.stream().distinct().toList();
    }
}
