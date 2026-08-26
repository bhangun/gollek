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
import tech.kayys.alkhawarizm.spi.download.DownloadProgressListener;
import tech.kayys.gollek.model.repo.local.GollekManifest;
import tech.kayys.gollek.model.repo.local.ManifestStore;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.hf.HuggingFaceModelInfo;
import tech.kayys.gollek.model.repo.hf.HuggingFaceRepository;
import tech.kayys.gollek.model.repo.kaggle.KaggleClient;
import tech.kayys.gollek.model.repo.kaggle.KaggleRepository;
import tech.kayys.gollek.sdk.model.ModelPullRequest;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.cli.util.CLIUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * Pull model using GollekSdk or directly from HuggingFace / Kaggle.
 *
 * <p>For {@code hf:} model specs the CLI always goes through the
 * {@link HuggingFaceClient} directly so that HTTP {@code Range}-based resumption
 * of partial ({@code .part}) downloads works reliably. For {@code kaggle:} / {@code kg:}
 * specs it uses {@link KaggleClient}. The SDK path is kept for
 * other providers (Ollama-style names, etc.).
 *
 * <p>Usage: {@code gollek pull [--force] [--include GLOB] [--task-type TYPE] <model>}
 */
@Dependent
@Unremovable
@Command(name = "pull", description = "Pull a model from a provider")
public class PullCommand implements Runnable {

    // ── ANSI helpers ────────────────────────────────────────────────────────
    private static final String RESET   = "\u001B[0m";
    private static final String BOLD    = "\u001B[1m";
    private static final String GREEN   = "\u001B[32m";
    private static final String CYAN    = "\u001B[36m";
    private static final String YELLOW  = "\u001B[33m";
    private static final String DIM     = "\u001B[2m";
    private static final String RED     = "\u001B[31m";

    @Inject GollekSdk sdk;
    @Inject Instance<HuggingFaceRepository> huggingFaceRepository;
    @Inject Instance<HuggingFaceClient>     huggingFaceClient;
    @Inject Instance<KaggleRepository>      kaggleRepository;
    @Inject Instance<KaggleClient>          kaggleClient;
    @Inject Instance<ManifestStore>         manifestStore;

    @Parameters(index = "0", arity = "0..1",
            description = "Model to pull (e.g. Qwen/Qwen2.5-7B-Instruct-GGUF or hf:owner/repo or kg:owner/model)")
    public String modelSpec;

    @Parameters(index = "1..*", arity = "0..*", hidden = true)
    public List<String> extraArgs;

    @Option(names = { "--insecure" }, description = "Allow insecure connections", defaultValue = "false")
    public boolean insecure;

    @Option(names = { "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    @Option(names = { "-f", "--file" }, description = "Copy model from local file path to internal model repo")
    String file;

    @Option(names = { "-m", "--move" }, description = "Move model from local file path instead of copying")
    boolean move;

    @Option(names = { "-i", "--include" },
            description = "Glob pattern to filter files when pulling from HuggingFace (e.g. '*Q4_K_M*'). "
                    + "Supports multiple values. Config/tokenizer files are always included.")
    List<String> include;

    @Option(names = { "--task-type", "--category" },
            description = "Override task category (text, vision, tts, stt, ocr, multimodal, embedding). "
                    + "Auto-detected from HuggingFace pipeline_tag when not set.")
    String taskType;

    @Option(names = { "--force", "--re-download" },
            description = "Force re-download even if files are already present and complete.",
            defaultValue = "false")
    boolean force;

    // ────────────────────────────────────────────────────────────────────────

    @Override
    public void run() {
        try {
            String rawSpec = modelSpec;
            if (extraArgs != null && !extraArgs.isEmpty() && rawSpec != null) {
                if ("kg:pull".equalsIgnoreCase(rawSpec) || "kg".equalsIgnoreCase(rawSpec)) {
                    rawSpec = "kg:" + String.join(" ", extraArgs).trim();
                } else if ("kaggle:pull".equalsIgnoreCase(rawSpec) || "kaggle".equalsIgnoreCase(rawSpec)) {
                    rawSpec = "kaggle:" + String.join(" ", extraArgs).trim();
                } else if ("hf:pull".equalsIgnoreCase(rawSpec) || "hf".equalsIgnoreCase(rawSpec)) {
                    rawSpec = "hf:" + String.join(" ", extraArgs).trim();
                } else {
                    rawSpec = rawSpec + " " + String.join(" ", extraArgs).trim();
                }
            }

            String effectiveModelSpec = normalizeModelSpec(rawSpec);

            if (file != null && !file.isBlank()) {
                handleLocalFileCopy(effectiveModelSpec);
                return;
            }

            if (effectiveModelSpec != null && (effectiveModelSpec.startsWith("hf:") || effectiveModelSpec.startsWith("huggingface:"))) {
                // ── Fast path: always use HF client for reliable resume ──────
                pullFromHuggingFace(effectiveModelSpec);
            } else if (effectiveModelSpec != null && (effectiveModelSpec.startsWith("kaggle:") || effectiveModelSpec.startsWith("kg:"))) {
                // ── Direct Kaggle pull path ──────────────────────────────────
                pullFromKaggle(effectiveModelSpec);
            } else {
                // ── SDK path for other providers ─────────────────────────────
                pullViaSdk(effectiveModelSpec);
            }
        } catch (Exception e) {
            System.err.println(RED + "\nFailed to pull model: " + e.getMessage() + RESET);
            if (e.getCause() != null) {
                System.err.println(DIM + "Cause: " + e.getCause().getMessage() + RESET);
            }
        }
    }

    // ── HuggingFace direct pull ──────────────────────────────────────────────

    private void pullFromHuggingFace(String effectiveModelSpec) throws Exception {
        if (huggingFaceClient == null || huggingFaceClient.isUnsatisfied()) {
            throw new RuntimeException(
                    "HuggingFace client is not available. Check your build profile or HF token configuration.");
        }

        String repoId = effectiveModelSpec.substring("hf:".length()).trim();
        if (repoId.isBlank()) {
            throw new RuntimeException("Empty HuggingFace repo ID in spec: " + effectiveModelSpec);
        }

        System.out.printf("%sPulling%s hf:%s%n%n", BOLD, RESET, repoId);

        HuggingFaceClient client = huggingFaceClient.get();

        // ── Fetch file list and model info in one API call ───────────────────
        HuggingFaceModelInfo modelInfo;
        try {
            modelInfo = client.getModelInfo(repoId);
        } catch (IOException e) {
            throw new RuntimeException("Failed to fetch model info for " + repoId
                    + ": " + e.getMessage(), e);
        }

        List<HuggingFaceModelInfo.ModelFile> allModelFiles =
                modelInfo.getFiles() != null ? modelInfo.getFiles() : List.of();

        List<String> allFilenames = allModelFiles.stream()
                .map(HuggingFaceModelInfo.ModelFile::getFilename)
                .toList();

        List<String> filesToDownload = filterFiles(allFilenames, include);

        if (filesToDownload.isEmpty()) {
            System.err.println(RED + "No files matched the --include filter. Available files:" + RESET);
            allFilenames.forEach(f -> System.err.println("  " + f));
            throw new RuntimeException("No matching files for include filter: " + include);
        }

        String manifestName = GollekManifest.computeName(repoId, "main");
        Path targetDir = ManifestStore.resolveBlobDir(repoId, manifestName);
        Files.createDirectories(targetDir);

        // ── Build a size map from the API response (LFS preferred) ───────────
        java.util.Map<String, Long> knownSizes = new java.util.HashMap<>();
        for (HuggingFaceModelInfo.ModelFile mf : allModelFiles) {
            if (mf.getFilename() == null) continue;
            Long size = mf.getLfs() != null && mf.getLfs().getSize() != null
                    ? mf.getLfs().getSize()
                    : mf.getSize();
            if (size != null && size > 0) {
                knownSizes.put(mf.getFilename(), size);
            }
        }

        // ── Pre-flight: classify each file ───────────────────────────────────
        record FileStatus(String name, String status, long localBytes, long totalBytes) {}
        List<FileStatus> plan = new ArrayList<>();
        long totalNew = 0, totalResume = 0, totalSkip = 0;
        int filesToTransferCount = 0;

        for (String filename : filesToDownload) {
            if (filename.startsWith(".")) continue;
            Path dest = targetDir.resolve(filename);
            Path part = dest.resolveSibling(dest.getFileName() + ".part");
            long known = knownSizes.getOrDefault(filename, 0L);

            String status;
            long local = 0;
            if (force) {
                // Delete both target and .part only if explicitly forced
                Files.deleteIfExists(dest);
                Files.deleteIfExists(part);
                status = "DOWNLOAD";
                totalNew += known;
                filesToTransferCount++;
            } else if (Files.exists(dest)) {
                local = Files.size(dest);
                if (known > 0 && local == known) {
                    status = "SKIP";
                    totalSkip += known;
                } else if (known <= 0 && local > 0) {
                    // Size was not returned in API metadata, but full local file exists
                    status = "SKIP";
                    totalSkip += local;
                } else {
                    // Size known and local size mismatch
                    status = "DOWNLOAD";
                    totalNew += known;
                    filesToTransferCount++;
                }
            } else if (Files.exists(part)) {
                local = Files.size(part);
                if (known > 0 && local >= known) {
                    try {
                        Files.move(part, dest, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                        status = "SKIP";
                        totalSkip += local;
                    } catch (Exception e) {
                        status = "RESUME";
                        totalResume += (known > local ? known - local : 0);
                        filesToTransferCount++;
                    }
                } else {
                    status = "RESUME";
                    totalResume += (known > local ? known - local : local);
                    filesToTransferCount++;
                }
            } else {
                status = "DOWNLOAD";
                totalNew += known;
                filesToTransferCount++;
            }
            plan.add(new FileStatus(filename, status, local, known > 0 ? known : local));
        }

        // ── Print the plan ────────────────────────────────────────────────────
        int maxName = plan.stream().mapToInt(s -> s.name().length()).max().orElse(20);
        maxName = Math.max(maxName, 20);
        for (FileStatus fs : plan) {
            String colour = switch (fs.status()) {
                case "SKIP"     -> DIM + GREEN;
                case "RESUME"   -> CYAN;
                case "DOWNLOAD" -> YELLOW;
                default         -> "";
            };
            String sizeInfo = fs.totalBytes() > 0
                    ? CLIUtils.formatSize(fs.totalBytes())
                    : "?";
            String resumeInfo = "RESUME".equals(fs.status()) && fs.localBytes() > 0
                    ? DIM + " (+" + CLIUtils.formatSize(Math.max(0, fs.totalBytes() - fs.localBytes())) + " to go)" + RESET
                    : "";
            System.out.printf("  %s%-8s%s %-" + maxName + "s  %s%s%n",
                    colour, fs.status(), RESET,
                    truncate(fs.name(), maxName),
                    sizeInfo, resumeInfo);
        }

        long totalTransfer = totalNew + totalResume;
        System.out.println();
        if (filesToTransferCount > 0) {
            if (totalTransfer > 0) {
                System.out.printf("  %s→ %s to transfer%s   %s%s already complete%s%n%n",
                        BOLD, CLIUtils.formatSize(totalTransfer), RESET,
                        DIM, CLIUtils.formatSize(totalSkip), RESET);
            } else {
                System.out.printf("  %s→ %d file(s) to transfer%s   %s%s already complete%s%n%n",
                        BOLD, filesToTransferCount, RESET,
                        DIM, CLIUtils.formatSize(totalSkip), RESET);
            }
        } else {
            System.out.printf("  %s✔ All files already complete.%s  "
                    + "Use %s--force%s to re-download.%n%n",
                    GREEN, RESET, BOLD, RESET);
        }

        // ── Download ──────────────────────────────────────────────────────────
        int filesDone = 0, filesSkipped = 0;
        for (FileStatus fs : plan) {
            if ("SKIP".equals(fs.status())) {
                filesSkipped++;
                continue;
            }
            Path dest = targetDir.resolve(fs.name());
            Files.createDirectories(dest.getParent());

            String verb = "RESUME".equals(fs.status()) ? "Resuming" : "Downloading";
            System.out.printf("  %s%s%s %s%s%s ...%n",
                    CYAN, verb, RESET, DIM, fs.name(), RESET);

            try (HfProgressRenderer progress = new HfProgressRenderer(
                    plan.size() - filesSkipped, filesDone + 1, fs.name())) {
                client.downloadFile(repoId, fs.name(), dest, progress);
            }
            filesDone++;
        }

        // ── Manifest + index ──────────────────────────────────────────────────
        saveDirectDownloadManifest(repoId, targetDir, filesToDownload, manifestName, modelInfo);
        LocalModelIndex.refreshFromDisk();

        // ── Summary ───────────────────────────────────────────────────────────
        System.out.println();
        if (filesDone == 0 && filesSkipped > 0) {
            System.out.printf("%s✔ Already up to date%s — %s%d file(s) complete%s%n",
                    GREEN, RESET, DIM, filesSkipped, RESET);
        } else {
            System.out.printf("%s✔ Pull complete%s — %s%d downloaded, %d skipped%s%n",
                    GREEN, RESET, DIM, filesDone, filesSkipped, RESET);
        }
        System.out.printf("  %sLocation:%s %s%s%n",
                DIM, RESET, targetDir.toAbsolutePath(), RESET);
    }

    // ── SDK path (non-HF providers) ─────────────────────────────────────────

    private void pullViaSdk(String effectiveModelSpec) throws Exception {
        boolean convert = !"off".equalsIgnoreCase(convertMode);
        ModelPullRequest request = ModelPullRequest.builder()
                .modelSpec(effectiveModelSpec)
                .convertIfNecessary(convert)
                .quantization(ggufOutType)
                .outType(ggufOutType)
                .build();

        System.out.println("Pulling model: " + effectiveModelSpec);
        System.out.println();

        sdk.pullModel(request, progress -> {
            if (progress.getTotal() > 0) {
                String bar = progress.getProgressBar(30);
                System.out.printf("\r%s [%s] %3d%% (%d/%d MB)",
                        progress.getStatus(), bar,
                        progress.getPercentComplete(),
                        progress.getCompleted() / 1024 / 1024,
                        progress.getTotal() / 1024 / 1024);
            } else {
                System.out.printf("\r%s...", progress.getStatus());
            }
        });

        System.out.println("\n" + GREEN + "✔ Pull complete: " + RESET + effectiveModelSpec);
    }

    // ── Manifest helper ──────────────────────────────────────────────────────

    private void saveDirectDownloadManifest(
            String repoId,
            Path targetDir,
            List<String> files,
            String manifestName,
            HuggingFaceModelInfo modelInfo) throws Exception {

        ManifestStore store = manifestStore != null && !manifestStore.isUnsatisfied()
                ? manifestStore.get()
                : new ManifestStore();

        String format = ManifestStore.detectFormat(targetDir);
        GollekManifest manifest = new GollekManifest();
        manifest.setId(manifestName);
        manifest.setModelId(repoId);
        manifest.setName(manifestName);
        manifest.setFormat(format);
        manifest.setPipeline(files != null && files.stream().anyMatch(n -> n.endsWith("model_index.json")));
        manifest.setSource("huggingface");
        manifest.setRepository(repoId);
        manifest.setBranch("main");
        manifest.setBlobPath(targetDir.toAbsolutePath().toString());
        manifest.setFiles(ManifestStore.listBlobFiles(targetDir));
        manifest.setCreatedAt(Instant.now());
        manifest.setSizeBytes(computeSize(targetDir));
        manifest.setArchitecture(ManifestStore.detectArchitecture(manifest));

        // ── Task type ────────────────────────────────────────────────────────
        String resolvedTaskType = taskType; // --task-type flag wins
        String detectedPipelineTag = modelInfo != null ? modelInfo.getPipelineTag() : null;
        if ((resolvedTaskType == null || resolvedTaskType.isBlank()) && detectedPipelineTag != null) {
            resolvedTaskType = LocalModelIndex.pipelineTagToTaskType(detectedPipelineTag);
        }
        manifest.setTaskType(resolvedTaskType != null ? resolvedTaskType : "text");

        // ── Metadata ─────────────────────────────────────────────────────────
        java.util.Map<String, String> meta = new java.util.LinkedHashMap<>();
        meta.put("source", "huggingface");
        meta.put("format", format != null ? format : "unknown");
        if (detectedPipelineTag != null && !detectedPipelineTag.isBlank()) {
            meta.put("pipeline_tag", detectedPipelineTag);
        }
        if (modelInfo != null && modelInfo.getCommitSha() != null) {
            meta.put("commit_sha", modelInfo.getCommitSha());
        }
        manifest.setMetadata(meta);

        store.save(manifest);
    }

    // ── Local file import ────────────────────────────────────────────────────

    private void handleLocalFileCopy(String effectiveModelSpec) throws Exception {
        Path sourcePath = Path.of(file);
        if (!Files.exists(sourcePath)) {
            throw new RuntimeException("Source file does not exist: " + file);
        }

        String repoId = effectiveModelSpec != null && effectiveModelSpec.startsWith("hf:")
                ? effectiveModelSpec.substring("hf:".length()).trim()
                : (effectiveModelSpec != null ? effectiveModelSpec.trim() : sourcePath.getFileName().toString());

        String manifestName = GollekManifest.computeName(repoId, "main");
        Path targetDir = ManifestStore.resolveBlobDir(repoId, manifestName);
        Files.createDirectories(targetDir);
        Path targetPath = targetDir.resolve(sourcePath.getFileName());

        if (move) {
            System.out.println("Moving " + sourcePath + " → " + targetPath + " ...");
            Files.move(sourcePath, targetPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } else {
            System.out.println("Copying " + sourcePath + " → " + targetPath + " ...");
            Files.copy(sourcePath, targetPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }

        ManifestStore store = manifestStore != null && !manifestStore.isUnsatisfied()
                ? manifestStore.get()
                : new ManifestStore();
        store.registerLocal(repoId, targetPath);
        LocalModelIndex.refreshFromDisk();

        System.out.println(GREEN + "\n✔ Pull complete: " + RESET + repoId + " (imported from local file)");
    }

    // ── File filter ──────────────────────────────────────────────────────────

    /**
     * Filter a list of filenames against glob patterns.
     * Config/tokenizer files are always included so the model remains usable.
     * Returns all files when no patterns are given.
     */
    private List<String> filterFiles(List<String> files, List<String> patterns) {
        if (patterns == null || patterns.isEmpty()) {
            return files;
        }
        return files.stream()
                .filter(f -> isEssentialFile(f) || patterns.stream().anyMatch(p -> globMatches(p, f)))
                .collect(java.util.stream.Collectors.toList());
    }

    private boolean isEssentialFile(String filename) {
        String base = Path.of(filename).getFileName().toString().toLowerCase(Locale.ROOT);
        return base.equals("config.json")
                || base.equals("tokenizer.json")
                || base.equals("tokenizer_config.json")
                || base.equals("tokenizer.model")
                || base.equals("special_tokens_map.json")
                || base.equals("generation_config.json")
                || base.equals("vocab.json")
                || base.equals("merges.txt")
                || base.equals("readme.md")
                || base.equals("license");
    }

    private boolean globMatches(String pattern, String filename) {
        String base = Path.of(filename).getFileName().toString().toLowerCase(Locale.ROOT);
        String pat  = pattern.toLowerCase(Locale.ROOT);
        String regex = pat.replace(".", "\\.").replace("*", ".*").replace("?", ".");
        return base.matches(regex);
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    private String normalizeModelSpec(String raw) {
        if (raw == null) return null;
        String trimmed = raw.trim();
        if (trimmed.startsWith("kg:pull ") || trimmed.startsWith("kg:pull\t")) {
            trimmed = "kg:" + trimmed.substring(8).trim();
        } else if (trimmed.startsWith("kaggle:pull ") || trimmed.startsWith("kaggle:pull\t")) {
            trimmed = "kaggle:" + trimmed.substring(12).trim();
        } else if (trimmed.startsWith("hf:pull ") || trimmed.startsWith("hf:pull\t")) {
            trimmed = "hf:" + trimmed.substring(8).trim();
        }

        if (trimmed.startsWith("hf:") || trimmed.startsWith("huggingface:")) return trimmed;
        if (trimmed.startsWith("kg:") || trimmed.startsWith("kaggle:")) return trimmed;
        if (trimmed.startsWith("ms:") || trimmed.startsWith("modelscope:")) return trimmed;
        if (trimmed.startsWith("ollama:")) return trimmed;
        if (trimmed.contains("://")) return trimmed;
        if (trimmed.contains("/")) return "hf:" + trimmed;
        return trimmed;
    }

    // ── Kaggle direct pull ───────────────────────────────────────────────────

    private void pullFromKaggle(String effectiveModelSpec) throws Exception {
        if (kaggleClient == null || kaggleClient.isUnsatisfied()) {
            throw new RuntimeException(
                    "Kaggle client is not available. Check your build profile or Kaggle configuration.");
        }

        String slug = effectiveModelSpec.startsWith("kaggle:")
                ? effectiveModelSpec.substring("kaggle:".length()).trim()
                : effectiveModelSpec.substring("kg:".length()).trim();

        if (slug.isBlank()) {
            throw new RuntimeException("Empty Kaggle model slug in spec: " + effectiveModelSpec);
        }

        System.out.printf("%sPulling%s kaggle:%s%n%n", BOLD, RESET, slug);

        KaggleClient client = kaggleClient.get();
        List<KaggleClient.KaggleFileEntry> allFiles;
        try {
            allFiles = client.listModelFiles(slug);
        } catch (IOException e) {
            throw new RuntimeException("Failed to fetch model info for " + slug + ": " + e.getMessage(), e);
        }

        if (allFiles == null || allFiles.isEmpty()) {
            throw new RuntimeException("No files found in Kaggle model: " + slug);
        }

        List<String> allFilenames = allFiles.stream()
                .map(KaggleClient.KaggleFileEntry::path)
                .toList();

        List<String> filesToDownload = filterFiles(allFilenames, include);

        if (filesToDownload.isEmpty()) {
            System.err.println(RED + "No files matched the --include filter. Available files:" + RESET);
            allFilenames.forEach(f -> System.err.println("  " + f));
            throw new RuntimeException("No matching files for include filter: " + include);
        }

        String manifestName = GollekManifest.computeName(slug.replace('/', '-'), "main");
        Path targetDir = ManifestStore.resolveBlobDir(slug.replace('/', '-'), manifestName);
        Files.createDirectories(targetDir);

        java.util.Map<String, Long> knownSizes = new java.util.HashMap<>();
        for (KaggleClient.KaggleFileEntry fe : allFiles) {
            if (fe.path() != null && fe.size() > 0) {
                knownSizes.put(fe.path(), fe.size());
            }
        }

        record FileStatus(String name, String status, long localBytes, long totalBytes) {}
        List<FileStatus> plan = new ArrayList<>();
        long totalPlanBytes = 0;
        int cachedCount = 0;

        for (String filename : filesToDownload) {
            Path targetFile = targetDir.resolve(filename);
            long expectedSize = knownSizes.getOrDefault(filename, -1L);
            if (!force && Files.exists(targetFile)) {
                long localSize = Files.size(targetFile);
                if (expectedSize > 0 && localSize == expectedSize) {
                    plan.add(new FileStatus(filename, "CACHED", localSize, expectedSize));
                    cachedCount++;
                    continue;
                } else if (localSize > 0 && (expectedSize < 0 || localSize < expectedSize)) {
                    plan.add(new FileStatus(filename, "RESUME", localSize, expectedSize));
                    totalPlanBytes += (expectedSize > 0 ? expectedSize - localSize : 0);
                    continue;
                }
            }
            plan.add(new FileStatus(filename, "NEW", 0, expectedSize));
            totalPlanBytes += Math.max(0, expectedSize);
        }

        System.out.printf("%d file(s) to process (%d cached, %s to download):%n",
                filesToDownload.size(), cachedCount, CLIUtils.formatSize(totalPlanBytes));
        for (FileStatus fs : plan) {
            String sizeStr = fs.totalBytes() > 0 ? " (" + CLIUtils.formatSize(fs.totalBytes()) + ")" : "";
            switch (fs.status()) {
                case "CACHED" -> System.out.printf("  %s✔ %s%s [cached]%s%n", GREEN, fs.name(), sizeStr, RESET);
                case "RESUME" -> System.out.printf("  %s↻ %s%s [resuming from %s]%s%n", YELLOW, fs.name(), sizeStr, CLIUtils.formatSize(fs.localBytes()), RESET);
                default -> System.out.printf("  %s↓ %s%s%s%n", CYAN, fs.name(), sizeStr, RESET);
            }
        }
        System.out.println();

        int filesDone = 0, filesSkipped = 0;
        for (FileStatus fs : plan) {
            if ("CACHED".equals(fs.status())) {
                filesSkipped++;
                continue;
            }

            Path targetFile = targetDir.resolve(fs.name());
            Files.createDirectories(targetFile.getParent());

            String verb = "RESUME".equals(fs.status()) ? "Resuming" : "Downloading";
            System.out.printf("  %s%s%s %s%s%s ...%n",
                    CYAN, verb, RESET, DIM, fs.name(), RESET);

            try (HfProgressRenderer progress = new HfProgressRenderer(
                    plan.size() - filesSkipped, filesDone + 1, fs.name())) {
                client.downloadFile(slug, fs.name(), targetFile, progress);
            }
            filesDone++;
        }

        saveDirectDownloadManifestKaggle(slug, targetDir, filesToDownload, manifestName);
        LocalModelIndex.refreshFromDisk();
        System.out.printf("%n%s✔ Pull complete:%s kaggle:%s%n", GREEN, RESET, slug);
    }

    private void saveDirectDownloadManifestKaggle(
            String slug,
            Path targetDir,
            List<String> files,
            String manifestName) throws Exception {

        ManifestStore store = manifestStore != null && !manifestStore.isUnsatisfied()
                ? manifestStore.get()
                : new ManifestStore();

        String format = ManifestStore.detectFormat(targetDir);
        GollekManifest manifest = new GollekManifest();
        manifest.setId(manifestName);
        manifest.setModelId(slug);
        manifest.setName(manifestName);
        manifest.setFormat(format);
        manifest.setPipeline(files != null && files.stream().anyMatch(n -> n.endsWith("model_index.json")));
        manifest.setSource("kaggle");
        manifest.setRepository(slug);
        manifest.setBranch("main");
        manifest.setBlobPath(targetDir.toAbsolutePath().toString());
        manifest.setFiles(ManifestStore.listBlobFiles(targetDir));
        manifest.setCreatedAt(Instant.now());
        manifest.setSizeBytes(computeSize(targetDir));
        manifest.setArchitecture(ManifestStore.detectArchitecture(manifest));

        String resolvedTaskType = taskType;
        manifest.setTaskType(resolvedTaskType != null ? resolvedTaskType : "text");

        java.util.Map<String, String> meta = new java.util.LinkedHashMap<>();
        meta.put("source", "kaggle");
        meta.put("format", format != null ? format : "unknown");
        manifest.setMetadata(meta);

        store.save(manifest);
    }

    private long computeSize(Path path) {
        if (path == null || !Files.exists(path)) return 0L;
        try {
            if (Files.isRegularFile(path)) return Files.size(path);
            try (Stream<Path> walk = Files.walk(path)) {
                return walk.filter(Files::isRegularFile)
                        .mapToLong(p -> { try { return Files.size(p); } catch (Exception e) { return 0L; } })
                        .sum();
            }
        } catch (Exception ignored) {
            return 0L;
        }
    }

    private String truncate(String s, int max) {
        if (s == null) return "";
        return s.length() > max ? "…" + s.substring(s.length() - (max - 1)) : s;
    }

    // ── Progress renderer ────────────────────────────────────────────────────

    private static final class HfProgressRenderer implements DownloadProgressListener, AutoCloseable {
        private static final String[] SPINNER  = { "|", "/", "-", "\\" };
        private static final int      BAR_WIDTH = 30;
        private static final long     MIN_NS    = 70_000_000L;

        private final int    totalFiles;
        private final int    currentFileNo;
        private final String filename;

        private long lastRedrawNanos = 0L;
        private int  spinnerTick     = 0;
        private long fileStartNanos  = System.nanoTime();
        private long initialBytes    = -1L;

        HfProgressRenderer(int totalFiles, int currentFileNo, String filename) {
            this.totalFiles    = Math.max(1, totalFiles);
            this.currentFileNo = currentFileNo;
            this.filename      = filename;
        }

        @Override
        public synchronized void onProgress(long downloaded, long total, double progress) {
            long now = System.nanoTime();
            if (now - lastRedrawNanos < MIN_NS && downloaded < total) return;
            lastRedrawNanos = now;

            if (initialBytes < 0) {
                initialBytes = downloaded;
            }

            double pct    = Math.max(0.0, Math.min(1.0, progress));
            int    filled = (int) Math.round(BAR_WIDTH * pct);
            String bar    = "=".repeat(filled) + ".".repeat(BAR_WIDTH - filled);

            double elapsedSec = Math.max(0.001, (now - fileStartNanos) / 1e9);
            double mbDone     = downloaded / 1024.0 / 1024.0;
            double mbTotal    = total > 0 ? total / 1024.0 / 1024.0 : 0.0;
            double sessionMb  = Math.max(0.0, (downloaded - initialBytes) / 1024.0 / 1024.0);
            double speed      = sessionMb / elapsedSec;
            String spin       = SPINNER[spinnerTick++ % SPINNER.length];

            System.out.printf(
                    "\r  \u001B[36m%s\u001B[0m [\u001B[32m%-30s\u001B[0m] %3d%%"
                            + "  \u001B[2m%.1f/%.1f MB  %.1f MB/s  [%d/%d]\u001B[0m",
                    spin, bar, (int) Math.round(pct * 100),
                    mbDone, mbTotal, speed,
                    currentFileNo, totalFiles);
            System.out.flush();
        }

        @Override
        public synchronized void onComplete(long totalBytes) {
            double mb = totalBytes / 1024.0 / 1024.0;
            System.out.printf(
                    "\r  \u001B[32m✔\u001B[0m [\u001B[32m%-30s\u001B[0m] 100%%"
                            + "  \u001B[2m%.1f MB  [%d/%d]\u001B[0m%n",
                    "==============================",
                    mb, currentFileNo, totalFiles);
            System.out.flush();
        }

        @Override
        public void close() {
            System.out.flush();
        }
    }
}
