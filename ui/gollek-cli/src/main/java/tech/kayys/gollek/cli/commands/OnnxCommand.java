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
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.kaggle.KaggleClient;
import tech.kayys.alkhawarizm.spi.download.DownloadProgressListener;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;

/**
 * ONNX model management and inference commands.
 *
 * Usage:
 *   gollek onnx list                                - List loaded models
 *   gollek onnx load <model-id> <model-path>        - Load an ONNX model
 *   gollek onnx unload <model-id>                   - Unload a model
 *   gollek onnx pull <model-spec> [--dir <path>]    - Pull .onnx model
 */
@Dependent
@Unremovable
@Command(name = "onnx",
         description = "ONNX model management and inference",
         subcommands = {
             OnnxCommand.ListCommand.class,
             OnnxCommand.LoadCommand.class,
             OnnxCommand.UnloadCommand.class,
             OnnxCommand.PullCommand.class
         })
public class OnnxCommand implements Runnable {

    @Option(names = {"-h", "--help"}, usageHelp = true, description = "Show help")
    boolean helpRequested;

    @Override
    public void run() {
        System.out.println("Usage: gollek onnx <command> [options]");
        System.out.println();
        System.out.println("ONNX model management commands:");
        System.out.println("  list                                List loaded models");
        System.out.println("  load <model-id> <model-path>        Load an ONNX model");
        System.out.println("  unload <model-id>                   Unload a model");
        System.out.println("  pull <model-spec> [--dir <path>]    Pull .onnx model");
    }

    @Command(name = "list", description = "List loaded ONNX models")
    public static class ListCommand implements Callable<Integer> {
        @Override
        public Integer call() {
            System.out.println("ONNX models: (use SDK for full functionality)");
            System.out.println("  onnx:models -> list all cached ONNX models");
            return 0;
        }
    }

    @Command(name = "load", description = "Load an ONNX model")
    public static class LoadCommand implements Callable<Integer> {
        @Parameters(index = "0", description = "Model identifier")
        String modelId;

        @Parameters(index = "1", description = "Path to .onnx model file")
        String modelPath;

        @Option(names = {"--ep"}, description = "Execution provider: CPU, CUDA, TensorRT (default: CPU)")
        String ep = "CPU";

        @Option(names = {"--threads"}, description = "Number of intra-op threads (default: 4)")
        int numThreads = 4;

        @Override
        public Integer call() {
            System.out.println("Loading ONNX model: " + modelId);
            System.out.println("  Path: " + modelPath);
            System.out.println("  EP: " + ep);
            System.out.println("  Threads: " + numThreads);
            System.out.println();
            System.out.println("✓ Model load command received (use SDK for actual loading)");
            return 0;
        }
    }

    @Command(name = "unload", description = "Unload an ONNX model")
    public static class UnloadCommand implements Callable<Integer> {
        @Parameters(index = "0", description = "Model identifier")
        String modelId;

        @Override
        public Integer call() {
            System.out.println("✓ Model unload command received: " + modelId);
            return 0;
        }
    }

    /**
     * Pull an .onnx model from HF or Kaggle.
     */
    @Command(name = "pull", description = "Pull an .onnx model from HuggingFace or Kaggle")
    public static class PullCommand implements Callable<Integer> {

        @Inject
        Instance<HuggingFaceClient> hfClientInstance;
        @Inject
        Instance<KaggleClient> kaggleClientInstance;

        @Parameters(index = "0", description = "Model spec (e.g. hf:user/repo, kaggle:user/repo)")
        String modelSpec;

        @Option(names = {"--dir"}, description = "Output directory (default: ~/.gollek/models/onnx)")
        String outputDir;

        @Override
        public Integer call() throws Exception {
            Path dir = outputDir != null ? Path.of(outputDir)
                    : GollekHome.path("models", "onnx");
            Files.createDirectories(dir);

            if (modelSpec.startsWith("hf:")) {
                return pullFromHf(modelSpec.substring(3), dir);
            } else if (modelSpec.startsWith("kaggle:")) {
                return pullFromKaggle(modelSpec.substring(7), dir);
            } else {
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

            Optional<String> onnxFile = files.stream()
                    .filter(f -> f.toLowerCase().endsWith(".onnx"))
                    .findFirst();

            if (onnxFile.isEmpty()) {
                System.err.println("No .onnx file found in " + repoId);
                return 1;
            }

            Path target = targetDir.resolve(fileNameOnly(onnxFile.get()));
            System.out.println("Downloading " + onnxFile.get() + " from HF " + repoId);
            client.downloadFile(repoId, onnxFile.get(), target, progressPrinter(onnxFile.get()));

            System.out.println("\n✓ ONNX model saved to " + target.toAbsolutePath());
            return 0;
        }

        private int pullFromKaggle(String slug, Path targetDir) throws Exception {
            if (kaggleClientInstance.isUnsatisfied()) {
                System.err.println("Kaggle client not available");
                return 1;
            }
            var client = kaggleClientInstance.get();
            List<String> files = client.listFiles(slug);

            Optional<String> onnxFile = files.stream()
                    .filter(f -> f.toLowerCase().endsWith(".onnx"))
                    .findFirst();

            if (onnxFile.isEmpty()) {
                System.err.println("No .onnx file found in " + slug);
                return 1;
            }

            Path target = targetDir.resolve(fileNameOnly(onnxFile.get()));
            System.out.println("Downloading " + onnxFile.get() + " from Kaggle " + slug);
            client.downloadFile(slug, onnxFile.get(), target, progressPrinter(onnxFile.get()));

            System.out.println("\n✓ ONNX model saved to " + target.toAbsolutePath());
            return 0;
        }

        private String fileNameOnly(String path) {
            int idx = path.lastIndexOf('/');
            return idx >= 0 ? path.substring(idx + 1) : path;
        }

        private DownloadProgressListener progressPrinter(String filename) {
            return new DownloadProgressListener() {
                private long lastUpdate = 0;
                @Override public void onStart(long totalBytes) {
                    System.out.println("Starting download: " + filename);
                }
                @Override public void onProgress(long downloaded, long total, double progress) {
                    long now = System.currentTimeMillis();
                    if (now - lastUpdate < 500) return;
                    lastUpdate = now;
                    if (total > 0) {
                        int pct = (int) Math.min(100, Math.round(progress * 100));
                        System.out.printf("\r  %d%% (%d/%d MB)   ", pct,
                                downloaded / 1024 / 1024, total / 1024 / 1024);
                    }
                }
                @Override public void onComplete(long totalBytes) { System.out.println(); }
                @Override public void onError(Throwable error) {
                    System.out.println();
                    System.err.println("Download failed: " + error.getMessage());
                }
            };
        }
    }
}
