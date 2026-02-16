package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderInfo;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;

/**
 * Run inference command using GolekSdk.
 * Usage: golek run --model <model> --prompt <prompt> [--provider
 * ollama|gguf|gemini] [--stream]
 */
@Dependent
@Unremovable
@Command(name = "run", description = "Run inference using a specified model")
public class RunCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Option(names = { "-m", "--model" }, description = "Model ID or path", required = true)
    String modelId;

    @Option(names = { "-p", "--prompt" }, description = "Input prompt", required = true)
    String prompt;

    @Option(names = {
            "--provider" }, description = "Provider: litert, gguf, djl, libtorch(experimental), ollama, gemini, openai, anthropic, cerebras")
    String providerId;

    @Option(names = { "-s", "--stream" }, description = "Stream output", defaultValue = "true")
    boolean stream;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.2")
    double temperature;

    @Option(names = { "--top-p" }, description = "Top-p sampling", defaultValue = "0.9")
    double topP;

    @Option(names = { "--top-k" }, description = "Top-k sampling", defaultValue = "40")
    int topK;

    @Option(names = { "--repeat-penalty" }, description = "Repeat penalty", defaultValue = "1.1")
    double repeatPenalty;

    @Option(names = { "--json" }, description = "Enable JSON mode", defaultValue = "false")
    boolean jsonMode;

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate", defaultValue = "96")
    int maxTokens;

    @Option(names = { "--mirostat" }, description = "Mirostat mode (0, 1, 2)", defaultValue = "0")
    int mirostat;

    @Option(names = { "--grammar" }, description = "GBNF grammar string")
    String grammar;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

    @Option(names = { "--no-cache" }, description = "Bypass response cache")
    boolean noCache;

    @Option(names = { "--offline",
            "--local" }, description = "Force using existing models without checking for updates/downloads")
    boolean offline;

    @Option(names = { "--model-path" }, description = "Path to a custom model file (bypasses repository lookup)")
    String modelPath;

    @Option(names = { "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String CYAN = "\u001B[36m";
    private static final String YELLOW = "\u001B[33m";
    private static final String DIM = "\u001B[2m";
    private static final String BOLD = "\u001B[1m";

    @Override
    public void run() {
        long startTime = System.currentTimeMillis();

        try {
            configureCheckpointConversionPreference();

            boolean customModelPathUsed = false;
            System.out.println(BOLD + YELLOW + "  _____       _      _    " + RESET);
            System.out.println(BOLD + YELLOW + " / ____|     | |    | |   " + RESET);
            System.out.println(BOLD + YELLOW + "| |  __  ___ | | ___| | __" + RESET);
            System.out.println(BOLD + YELLOW + "| | |_ |/ _ \\| |/ _ \\ |/ /" + RESET);
            System.out.println(BOLD + YELLOW + "| |__| | (_) | |  __/   < " + RESET);
            System.out.println(BOLD + YELLOW + " \\_____|\\___/|_|\\___|_|\\_\\" + RESET);
            System.out.println();

            // If --model-path is provided, use it directly
            if (modelPath != null && !modelPath.isEmpty()) {
                Path customModelPath = Paths.get(modelPath);
                if (!Files.exists(customModelPath)) {
                    System.err.println("Error: Model file not found: " + modelPath);
                    return;
                }
                System.out.println("Using model from: " + customModelPath.toAbsolutePath());
                modelId = customModelPath.toAbsolutePath().toString();
                customModelPathUsed = true;
            } else {
                // Check if model exists locally
                System.out.printf("Checking model: %s... ", modelId);
                var modelInfoOpt = sdk.getModelInfo(modelId);
                boolean exists = modelInfoOpt.isPresent();

                // Smart fallback: check for -GGUF variant locally if base not found
                if (!exists) {
                    String fallbackId = modelId + "-GGUF";
                    var fallbackInfoOpt = sdk.getModelInfo(fallbackId);
                    if (fallbackInfoOpt.isPresent()) {
                        System.out.println("found local variant: " + fallbackId);
                        modelId = fallbackId;
                        modelInfoOpt = fallbackInfoOpt;
                        exists = true;
                    }
                }

                if (!exists) {
                    System.out.println("not found locally.");

                    if (offline) {
                        System.err.println("Error: Model not found locally and --offline mode is active.");
                        return;
                    }

                    if (modelId.contains("/") || modelId.startsWith("hf:")) {
                        System.out.println("Attempting to download model from Hugging Face...");
                        boolean pulled = false;
                        Exception lastPullError = null;
                        for (String pullSpec : buildPullSpecs(modelId)) {
                            try {
                                sdk.pullModel(pullSpec, progress -> {
                                    if (progress.getTotal() > 0) {
                                        System.out.printf("\rDownloading: %s %d%% (%d/%d MB)",
                                                progress.getProgressBar(20),
                                                progress.getPercentComplete(),
                                                progress.getCompleted() / 1024 / 1024,
                                                progress.getTotal() / 1024 / 1024);
                                    } else {
                                        System.out.print("\rDownloading: " + progress.getStatus());
                                    }
                                });
                                pulled = true;
                                break;
                            } catch (Exception e) {
                                lastPullError = e;
                                if (isFatalPullError(e)) {
                                    break;
                                }
                            }
                        }
                        if (!pulled) {
                            String reason = lastPullError != null ? describeError(lastPullError) : "unknown error";
                            System.err.println("Error: Failed to download from Hugging Face. " + reason);
                            return;
                        }
                        System.out.println("\nDownload complete!");

                        // Refresh model info after download
                        modelInfoOpt = sdk.getModelInfo(modelId);
                        if (modelInfoOpt.isEmpty()) {
                            // Check fallback again if download resolved to fallback
                            String fallbackId = modelId + "-GGUF";
                            if (sdk.getModelInfo(fallbackId).isPresent()) {
                                modelId = fallbackId;
                                modelInfoOpt = sdk.getModelInfo(modelId);
                            }
                        }
                        if (modelInfoOpt.isEmpty() && modelId.startsWith("hf:")) {
                            String normalizedId = modelId.substring(3);
                            modelInfoOpt = sdk.getModelInfo(normalizedId);
                            if (modelInfoOpt.isPresent()) {
                                modelId = normalizedId;
                            }
                        }
                        if (modelInfoOpt.isEmpty() && modelId.startsWith("hf:")) {
                            String normalizedFallback = modelId.substring(3) + "-GGUF";
                            modelInfoOpt = sdk.getModelInfo(normalizedFallback);
                            if (modelInfoOpt.isPresent()) {
                                modelId = normalizedFallback;
                            }
                        }
                        if (modelInfoOpt.isEmpty()) {
                            System.err.println(
                                    "Error: Download completed but model is still not available locally: " + modelId);
                            return;
                        }

                    } else {
                        System.err.println(
                                "Error: Model not found locally and does not appear to be a remote repository specification.");
                        return;
                    }
                } else {
                    System.out.println("found locally.");
                    modelInfoOpt.ifPresent(this::maybeUpgradeDjlLayout);
                }

                // Print model path if available
                modelInfoOpt.flatMap(info -> java.util.Optional.ofNullable(info.getMetadata().get("path")))
                        .ifPresent(path -> System.out.println("Model path: " + path));
            }

            // Set preferred provider if specified
            if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
            } else {
                maybeAutoSelectProvider();
            }
            if (!ensureProviderReady()) {
                return;
            }

            // Build request
            InferenceRequest.Builder requestBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .model(modelId)
                    .temperature(temperature)
                    .parameter("top_p", topP)
                    .parameter("top_k", topK)
                    .parameter("repeat_penalty", repeatPenalty)
                    .parameter("json_mode", jsonMode)
                    .maxTokens(maxTokens)
                    .streaming(stream);

            if (mirostat > 0) {
                requestBuilder.parameter("mirostat", mirostat);
            }
            if (grammar != null && !grammar.isEmpty()) {
                requestBuilder.parameter("grammar", grammar);
            }
            if (customModelPathUsed) {
                requestBuilder.parameter("model_path", modelId);
            }

            if (systemPrompt != null && !systemPrompt.isEmpty()) {
                requestBuilder.message(Message.system(systemPrompt));
            }
            requestBuilder.message(Message.user(prompt));

            if (providerId != null && !providerId.isEmpty()) {
                requestBuilder.preferredProvider(providerId);
            }

            requestBuilder.cacheBypass(noCache);

            InferenceRequest request = requestBuilder.build();

            System.out.printf(BOLD + "Model: " + RESET + CYAN + "%s" + RESET + "%n", modelId);
            System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s" + RESET + "%n",
                    providerId != null ? providerId : "auto-select");
            System.out.println(DIM + "-".repeat(50) + RESET);

            if (stream) {
                CountDownLatch latch = new CountDownLatch(1);
                // Streaming mode
                java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(
                        0);
                sdk.streamCompletion(request)
                        .subscribe().with(
                                chunk -> {
                                    String delta = chunk.getDelta();
                                    if (delta != null) {
                                        System.out.print(delta);
                                        tokenCount.incrementAndGet();
                                        System.out.flush();
                                    }
                                },
                                error -> {
                                    System.err.println("\n" + YELLOW + "Error: " + RESET + error.getMessage());
                                    latch.countDown();
                                },
                                () -> {
                                    long duration = System.currentTimeMillis() - startTime;
                                    double tps = (tokenCount.get() / (duration / 1000.0));
                                    System.out.printf(
                                            "%n" + DIM + "[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]" + RESET
                                                    + "%n",
                                            tokenCount.get(), duration / 1000.0, tps);
                                    latch.countDown();
                                });
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            } else {
                // Sync mode
                InferenceResponse response = sdk.createCompletion(request);
                printResponse(response, startTime);
            }

        } catch (Exception e) {
            System.err.println("\nInference failed: " + e.getMessage());
            if (e.getCause() != null && !e.getMessage().contains(e.getCause().getMessage())) {
                System.err.println("Detail: " + e.getCause().getMessage());
            }
        }
    }

    private java.util.List<String> buildPullSpecs(String modelSpec) {
        if (modelSpec == null || modelSpec.isBlank()) {
            return java.util.List.of();
        }
        if (modelSpec.startsWith("hf:")) {
            String bare = modelSpec.substring(3);
            return java.util.List.of(modelSpec, bare);
        }
        if (modelSpec.contains("/")) {
            return java.util.List.of(modelSpec, "hf:" + modelSpec);
        }
        return java.util.List.of(modelSpec);
    }

    private void printResponse(InferenceResponse response, long startTime) {
        System.out.println();
        System.out.println(GREEN + response.getContent() + RESET);
        System.out.printf("%n" + DIM + "[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]" + RESET + "%n",
                response.getTokensUsed(),
                response.getDurationMs() / 1000.0,
                (response.getTokensUsed() / (response.getDurationMs() / 1000.0)));
    }

    private void maybeAutoSelectProvider() {
        try {
            var modelInfoOpt = sdk.getModelInfo(modelId);
            if (modelInfoOpt.isEmpty()) {
                return;
            }
            String format = modelInfoOpt.get().getFormat();
            String inferredProvider = providerForFormat(format);
            if (inferredProvider == null || inferredProvider.isBlank()) {
                return;
            }
            if (!isProviderHealthy(inferredProvider)) {
                return;
            }
            sdk.setPreferredProvider(inferredProvider);
            providerId = inferredProvider;
        } catch (Exception ignored) {
            // Keep default router behavior when format/provider probing is not available.
        }
    }

    private String providerForFormat(String format) {
        if (format == null || format.isBlank()) {
            return null;
        }
        String normalized = format.trim().toUpperCase();
        return switch (normalized) {
            case "GGUF" -> "gguf";
            case "TORCHSCRIPT" -> "djl";
            case "ONNX" -> "onnx";
            default -> null;
        };
    }

    private void maybeUpgradeDjlLayout(tech.kayys.golek.sdk.core.model.ModelInfo info) {
        if (info == null || info.getFormat() == null) {
            return;
        }
        String format = info.getFormat().trim().toUpperCase();
        if (!format.equals("TORCHSCRIPT")) {
            return;
        }
        Object rawPath = info.getMetadata() != null ? info.getMetadata().get("path") : null;
        String path = rawPath != null ? String.valueOf(rawPath).toLowerCase() : "";
        if (!path.contains("/models/torchscript/")) {
            return;
        }
        if (!(modelId.contains("/") || modelId.startsWith("hf:"))) {
            return;
        }
        try {
            System.out.println("Upgrading local model layout for DJL runtime: " + modelId);
            for (String spec : buildPullSpecs(modelId)) {
                try {
                    sdk.pullModel(spec, null);
                    break;
                } catch (Exception ignored) {
                    // try next variant
                }
            }
        } catch (Exception ignored) {
            // keep existing model when upgrade is unavailable
        }
    }

    private boolean isProviderHealthy(String id) {
        try {
            return sdk.listAvailableProviders().stream()
                    .filter(p -> id.equalsIgnoreCase(p.id()))
                    .findFirst()
                    .map(p -> p.healthStatus() == tech.kayys.golek.spi.provider.ProviderHealth.Status.HEALTHY)
                    .orElse(false);
        } catch (Exception e) {
            return false;
        }
    }

    private boolean ensureProviderReady() {
        try {
            if (providerId != null && !providerId.isBlank()) {
                return ensureProviderHealthy(providerId);
            }
            var modelInfoOpt = sdk.getModelInfo(modelId);
            if (modelInfoOpt.isEmpty()) {
                return true;
            }
            var modelInfo = modelInfoOpt.get();
            String format = modelInfo.getFormat();
            if (isCheckpointOnlyFormat(format)) {
                if (offline) {
                    System.err.printf("Model '%s' uses checkpoint format '%s' and cannot run in offline Java runtime.%n",
                            modelId, format);
                    System.err.println("Use a GGUF or TorchScript model, or rerun without --offline for fallback pull.");
                    return false;
                }
                if (tryRefreshCompatibleModel()) {
                    modelInfoOpt = sdk.getModelInfo(modelId);
                    if (modelInfoOpt.isPresent()) {
                        format = modelInfoOpt.get().getFormat();
                    }
                }
            }

            if (isCheckpointOnlyFormat(format)) {
                System.err.printf("Model '%s' uses unsupported runtime format '%s'.%n", modelId, format);
                System.err.println("Use a GGUF or TorchScript model.");
                return false;
            }

            String inferredProvider = providerForFormat(format);
            if (inferredProvider == null || inferredProvider.isBlank()) {
                return true;
            }
            return ensureProviderHealthy(inferredProvider);
        } catch (Exception ignored) {
            return true;
        }
    }

    private boolean isCheckpointOnlyFormat(String format) {
        if (format == null) {
            return false;
        }
        String normalized = format.trim().toUpperCase();
        return normalized.equals("PYTORCH") || normalized.equals("SAFETENSORS");
    }

    private void configureCheckpointConversionPreference() {
        String mode = convertMode == null ? "auto" : convertMode.trim().toLowerCase();
        if (mode.equals("off")) {
            System.setProperty("golek.gguf.converter.auto", "false");
        } else {
            System.setProperty("golek.gguf.converter.auto", "true");
        }
        if (ggufOutType != null && !ggufOutType.isBlank()) {
            System.setProperty("golek.gguf.converter.outtype", ggufOutType.trim().toLowerCase());
        }
    }

    private boolean tryRefreshCompatibleModel() {
        if (!(modelId.contains("/") || modelId.startsWith("hf:"))) {
            return false;
        }
        System.out.println("Checkpoint-only model detected. Trying GGUF/TorchScript fallback...");
        for (String spec : buildPullSpecs(modelId)) {
            try {
                sdk.pullModel(spec, null);
            } catch (Exception e) {
                if (isFatalPullError(e)) {
                    break;
                }
            }
        }
        String normalized = modelId.startsWith("hf:") ? modelId.substring(3) : modelId;
        for (String candidate : java.util.List.of(modelId, normalized, modelId + "-GGUF", normalized + "-GGUF")) {
            try {
                var info = sdk.getModelInfo(candidate);
                if (info.isPresent()) {
                    String fmt = info.get().getFormat();
                    if (!isCheckpointOnlyFormat(fmt)) {
                        modelId = candidate;
                        System.out.println("Using compatible model: " + modelId);
                        return true;
                    }
                }
            } catch (Exception ignored) {
                // continue
            }
        }
        return false;
    }

    private boolean ensureProviderHealthy(String provider) {
        Optional<ProviderInfo> info = findProviderInfo(provider);
        if (info.isEmpty()) {
            System.err.printf("Required provider is not available: %s%n", provider);
            return false;
        }
        if (info.get().healthStatus() != ProviderHealth.Status.UNHEALTHY) {
            return true;
        }
        printProviderSetupHint(info.get());
        return false;
    }

    private Optional<ProviderInfo> findProviderInfo(String id) {
        try {
            return sdk.listAvailableProviders().stream()
                    .filter(p -> id.equalsIgnoreCase(p.id()))
                    .findFirst();
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    private void printProviderSetupHint(ProviderInfo provider) {
        String id = provider.id();
        System.err.printf("Provider '%s' is installed but not healthy.%n", id);
        String healthMessage = null;
        if (provider.metadata() != null) {
            Object value = provider.metadata().get("healthMessage");
            if (value != null) {
                healthMessage = String.valueOf(value);
            }
        }
        if (healthMessage != null && !healthMessage.isBlank()) {
            System.err.println("Reason: " + healthMessage);
        }
        if (provider.metadata() != null) {
            Object details = provider.metadata().get("healthDetails");
            if (details instanceof java.util.Map<?, ?> map) {
                Object reason = map.get("reason");
                if (reason != null) {
                    System.err.println("Detail: " + reason);
                }
            }
        }
        if ("djl".equalsIgnoreCase(id)) {
            System.err.println(
                    "DJL PyTorch engine is unavailable. Ensure internet access for first-run native download or pre-install DJL native runtime.");
        } else if ("libtorch".equalsIgnoreCase(id)) {
            System.err.println("Set LIBTORCH_PATH to your libtorch native library directory.");
        } else if ("gguf".equalsIgnoreCase(id)) {
            System.err.println("Set GOLEK_LLAMA_LIB_DIR or GOLEK_LLAMA_LIB_PATH to llama.cpp native libraries.");
        }
    }

    private String describeError(Throwable throwable) {
        if (throwable == null) {
            return "unknown error";
        }
        StringBuilder sb = new StringBuilder();
        Throwable current = throwable;
        int guard = 0;
        while (current != null && guard++ < 8) {
            String msg = current.getMessage();
            if (msg != null && !msg.isBlank()) {
                if (sb.length() > 0) {
                    sb.append(" | ");
                }
                sb.append(msg);
            }
            current = current.getCause();
        }
        if (sb.length() == 0) {
            return throwable.getClass().getSimpleName();
        }
        return sb.toString();
    }

    private boolean isFatalPullError(Throwable throwable) {
        String detail = describeError(throwable).toLowerCase();
        return detail.contains("401")
                || detail.contains("403")
                || detail.contains("404")
                || detail.contains("unauthorized")
                || detail.contains("forbidden")
                || detail.contains("gated")
                || detail.contains("access denied")
                || detail.contains("not found")
                || detail.contains("model conversion service not available")
                || detail.contains("conversion process failed")
                || detail.contains("no gguf found")
                || detail.contains("unsupported runtime format")
                || detail.contains("checkpoint-only model");
    }
}
