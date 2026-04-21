package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.cli.GollekHome;
import tech.kayys.gollek.sdk.core.GollekSdk;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import tech.kayys.gollek.cli.util.CLIUtils;

@Dependent
@Unremovable
@Command(name = "delete", aliases = { "remove",
        "rm" }, description = "Delete/remove a local model by id, name, or path")
public class DeleteCommand implements Runnable {

    @Inject
    GollekSdk sdk;

    @Parameters(index = "0", description = "Model id/name/path to delete")
    String modelRef;

    @Option(names = { "-y", "--yes" }, description = "Skip confirmation")
    boolean assumeYes;

    @Option(names = { "--all-matches" }, description = "Delete all matching local files/directories")
    boolean allMatches;

    @Override
    public void run() {
        final String ANSI_RESET = "\u001B[0m";
        final String ANSI_RED = "\u001B[31m";
        final String ANSI_GREEN = "\u001B[32m";
        final String ANSI_YELLOW = "\u001B[33m";

        if (modelRef == null || modelRef.isBlank()) {
            System.err.println(ANSI_RED + "Model reference is required." + ANSI_RESET);
            return;
        }
        LocalModelIndex.refreshFromDisk();
        String ref = modelRef.trim();

        // 1. Resolve targets (supporting short ID, name, or path)
        List<Path> targets = resolveTargets(ref);
        if (targets.isEmpty()) {
            System.err.println(ANSI_RED + "Error: Model not found: " + modelRef + ANSI_RESET);
            return;
        }

        if (targets.size() > 1 && !allMatches) {
            System.err.println(ANSI_YELLOW + "Found multiple matching files for '" + ref + "':" + ANSI_RESET);
            for (Path p : targets) {
                System.err.println("  - " + p);
            }
            System.err.println("Use " + ANSI_YELLOW + "--all-matches" + ANSI_RESET + " to delete all of them, or specify an exact path/ID.");
            return;
        }

        List<Path> toDelete = allMatches ? targets : List.of(targets.get(0));
        
        // 2. Confirmation
        if (!assumeYes && !confirmDeletion(toDelete)) {
            System.out.println(ANSI_YELLOW + "Delete cancelled." + ANSI_RESET);
            return;
        }

        // 3. Execution
        int deletedCount = 0;
        for (Path target : toDelete) {
            System.out.print("Deleting " + target.getFileName() + "... ");
            if (deletePath(target)) {
                deletedCount++;
                System.out.println(ANSI_GREEN + "DONE" + ANSI_RESET);
            } else {
                System.out.println(ANSI_RED + "FAILED" + ANSI_RESET);
            }
        }

        if (deletedCount > 0) {
            System.out.println(ANSI_GREEN + "\nSuccessfully deleted " + deletedCount + " model(s)." + ANSI_RESET);
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

        Path modelsRoot = GollekHome.path("models");
        if (!Files.isDirectory(modelsRoot)) {
            return dedupe(targets);
        }
        Path gguf = modelsRoot.resolve("gguf");
        Path libtorchscript = modelsRoot.resolve("libtorchscript");
        Path litert = modelsRoot.resolve("litert");

        targets.addAll(findInBase(gguf, ref));
        targets.addAll(findInBase(libtorchscript, ref));
        targets.addAll(findInBase(litert, ref));

        // Filename-only fallback search.
        String lowered = ref.toLowerCase(Locale.ROOT);
        for (Path base : List.of(gguf, libtorchscript, litert)) {
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

        String[] exts = { ".gguf", ".safetensors", ".safetensor", ".pt", ".pth", ".bin", ".litertlm", ".task" };
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
