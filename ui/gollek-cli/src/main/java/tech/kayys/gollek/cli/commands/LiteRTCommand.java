package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.sdk.util.GollekHome;
import tech.kayys.gollek.model.repo.kaggle.KaggleClient;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.alkhawarizm.spi.download.DownloadProgressListener;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;

/**
 * LiteRT / TFLite model management and inference commands.
 *
 * Usage:
 *   gollek litert list                                - List loaded models
 *   gollek litert load <model-id> <model-path>        - Load a model
 *   gollek litert unload <model-id>                   - Unload a model
 *   gollek litert metrics                             - Show performance metrics
 *   gollek litert pull <model-spec> [--dir <path>]    - Pull .tflite model
 */
@Dependent
@Unremovable
@Command(name = "litert",
         description = "LiteRT / TFLite model management and inference",
         subcommands = {
             LiteRTCommand.ListCommand.class,
             LiteRTCommand.LoadCommand.class,
             LiteRTCommand.UnloadCommand.class,
             LiteRTCommand.MetricsCommand.class,
             LiteRTCommand.PullCommand.class
         })
public class LiteRTCommand implements Runnable {

    @Option(names = {"-h", "--help"}, usageHelp = true, description = "Show help")
    boolean helpRequested;

    @Override
    public void run() {
        System.out.println("Usage: gollek litert <command> [options]");
        System.out.println();
        System.out.println("LiteRT / TFLite model management commands:");
        System.out.println("  list                                List loaded models");
        System.out.println("  load <model-id> <model-path>        Load a LiteRT model");
        System.out.println("  unload <model-id>                   Unload a model");
        System.out.println("  metrics                             Show performance metrics");
        System.out.println("  pull <model-spec> [--dir <path>]    Pull .tflite model");
    }

    @Command(name = "list", description = "List loaded LiteRT models")
    public static class ListCommand implements Callable<Integer> {
        @Override
        public Integer call() throws Exception {
            System.out.println("LiteRT models: (SDK integration active)");
            System.out.println("Use SDK directly for full functionality: LiteRTSdk sdk = new LiteRTSdk();");
            return 0;
        }
    }

    @Command(name = "load", description = "Load a LiteRT model")
    public static class LoadCommand implements Callable<Integer> {
        @Parameters(index = "0", description = "Model identifier")
        String modelId;

        @Parameters(index = "1", description = "Path to .litertlm or .litert model file")
        String modelPath;

        @Option(names = {"--threads"}, description = "Number of CPU threads (default: 4)")
        int numThreads = 4;

        @Override
        public Integer call() throws Exception {
            System.out.println("Loading model: " + modelId);
            System.out.println("  Path: " + modelPath);
            System.out.println("  Threads: " + numThreads);
            System.out.println();
            System.out.println("✓ Model load command received (use SDK for actual loading)");
            return 0;
        }
    }

    @Command(name = "unload", description = "Unload a LiteRT model")
    public static class UnloadCommand implements Callable<Integer> {
        @Parameters(index = "0", description = "Model identifier")
        String modelId;

        @Override
        public Integer call() throws Exception {
            System.out.println("✓ Model unload command received: " + modelId);
            return 0;
        }
    }

    @Command(name = "metrics", description = "Show performance metrics")
    public static class MetricsCommand implements Callable<Integer> {
        @Option(names = {"--reset"}, description = "Reset metrics after displaying")
        boolean reset = false;

        @Override
        public Integer call() throws Exception {
            System.out.println("Performance Metrics:");
            System.out.println("  (Use SDK directly: LiteRTMetrics metrics = sdk.getMetrics(null);)");
            return 0;
        }
    }

    /**
     * Pull a .tflite model from HF or Kaggle.
     */
    @Command(name = "pull", description = "Pull a .tflite model from HuggingFace or Kaggle")
    public static class PullCommand implements Callable<Integer> {

        @Inject
        Instance<HuggingFaceClient> hfClientInstance;
        @Inject
        Instance<KaggleClient> kaggleClientInstance;

        @Parameters(index = "0", description = "Model spec (e.g. hf:user/repo, kaggle:user/repo)")
        String modelSpec;

        @Option(names = {"--dir"}, description = "Output directory (default: ~/.gollek/models/litert)")
        String outputDir;

        @Override
        public Integer call() throws Exception {
            Path dir = outputDir != null ? Path.of(outputDir)
                    : GollekHome.path("models", "litert");
            Files.createDirectories(dir);

            if (modelSpec.startsWith("hf:")) {
                return pullFromHf(modelSpec.substring(3), dir);
            } else if (modelSpec.startsWith("kaggle:")) {
                return pullFromKaggle(modelSpec.substring(7), dir);
            } else {
                // Default to HF
                return pullFromHf(modelSpec, dir);
            }
        }

        private int pullFromHf(String repoId, Path targetDir) throws Exception {
            if (hfClientInstance.isUnsatisfied()) {
                System.err.println("HuggingFace client not available");
                return 1;
            }
            var client = hfClientInstance.get();
            List<String> files = client.listFiles(repoId);

            // Find .tflite file
            Optional<String> tfliteFile = files.stream()
                    .filter(f -> f.toLowerCase().endsWith(".tflite") || f.toLowerCase().endsWith(".litertlm"))
                    .findFirst();

            if (tfliteFile.isEmpty()) {
                System.err.println("No .tflite or .litertlm file found in " + repoId);
                return 1;
            }

            Path target = targetDir.resolve(fileNameOnly(tfliteFile.get()));
            System.out.println("Downloading " + tfliteFile.get() + " from HF " + repoId);
            client.downloadFile(repoId, tfliteFile.get(), target, progressPrinter(tfliteFile.get()));

            System.out.println("\n✓ Model saved to " + target.toAbsolutePath());
            return 0;
        }

        private int pullFromKaggle(String slug, Path targetDir) throws Exception {
            if (kaggleClientInstance.isUnsatisfied()) {
                System.err.println("Kaggle client not available");
                return 1;
            }
            var client = kaggleClientInstance.get();
            List<String> files = client.listFiles(slug);

            Optional<String> tfliteFile = files.stream()
                    .filter(f -> f.toLowerCase().endsWith(".tflite") || f.toLowerCase().endsWith(".litertlm"))
                    .findFirst();

            if (tfliteFile.isEmpty()) {
                System.err.println("No .tflite or .litertlm file found in " + slug);
                return 1;
            }

            Path target = targetDir.resolve(fileNameOnly(tfliteFile.get()));
            System.out.println("Downloading " + tfliteFile.get() + " from Kaggle " + slug);
            client.downloadFile(slug, tfliteFile.get(), target, progressPrinter(tfliteFile.get()));

            System.out.println("\n✓ Model saved to " + target.toAbsolutePath());
            return 0;
        }

        private String fileNameOnly(String path) {
            int idx = path.lastIndexOf('/');
            return idx >= 0 ? path.substring(idx + 1) : path;
        }

        private DownloadProgressListener progressPrinter(String filename) {
            return new DownloadProgressListener() {
                private long lastUpdate = 0;

                @Override
                public void onStart(long totalBytes) {
                    System.out.println("Starting download: " + filename);
                }

                @Override
                public void onProgress(long downloaded, long total, double progress) {
                    long now = System.currentTimeMillis();
                    if (now - lastUpdate < 500) return;
                    lastUpdate = now;
                    if (total > 0) {
                        int pct = (int) Math.min(100, Math.round(progress * 100));
                        System.out.printf("\r  %d%% (%d/%d MB)   ", pct,
                                downloaded / 1024 / 1024, total / 1024 / 1024);
                    }
                }

                @Override
                public void onComplete(long totalBytes) {
                    System.out.println();
                }

                @Override
                public void onError(Throwable error) {
                    System.out.println();
                    System.err.println("Download failed: " + error.getMessage());
                }
            };
        }
    }
}
