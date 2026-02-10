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

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.UUID;

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
            "--provider" }, description = "Provider: litert, gguf, ollama, gemini, openai, anthropic, cerebras")
    String providerId;

    @Option(names = { "-s", "--stream" }, description = "Stream output", defaultValue = "true")
    boolean stream;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.8")
    double temperature;

    @Option(names = { "--top-p" }, description = "Top-p sampling", defaultValue = "0.95")
    double topP;

    @Option(names = { "--top-k" }, description = "Top-k sampling", defaultValue = "40")
    int topK;

    @Option(names = { "--repeat-penalty" }, description = "Repeat penalty", defaultValue = "1.1")
    double repeatPenalty;

    @Option(names = { "--json" }, description = "Enable JSON mode", defaultValue = "true")
    boolean jsonMode;

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate", defaultValue = "2048")
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
            // If --model-path is provided, use it directly
            if (modelPath != null && !modelPath.isEmpty()) {
                Path customModelPath = Paths.get(modelPath);
                if (!Files.exists(customModelPath)) {
                    System.err.println("Error: Model file not found: " + modelPath);
                    return;
                }
                System.out.println("Using model from: " + customModelPath.toAbsolutePath());
                // Continue with inference using the custom path
                // The SDK will handle the file directly
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
                        sdk.pullModel(modelId, progress -> {
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

                    } else {
                        System.err.println(
                                "Error: Model not found locally and does not appear to be a remote repository specification.");
                        return;
                    }
                } else {
                    System.out.println("found locally.");
                }

                // Print model path if available
                modelInfoOpt.flatMap(info -> java.util.Optional.ofNullable(info.getMetadata().get("path")))
                        .ifPresent(path -> System.out.println("Model path: " + path));

                // Set preferred provider if specified
                if (providerId != null && !providerId.isEmpty()) {
                    sdk.setPreferredProvider(providerId);
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

                if (systemPrompt != null && !systemPrompt.isEmpty()) {
                    requestBuilder.message(Message.system(systemPrompt));
                }
                requestBuilder.message(Message.user(prompt));

                if (providerId != null && !providerId.isEmpty()) {
                    requestBuilder.preferredProvider(providerId);
                }

                requestBuilder.cacheBypass(noCache);

                InferenceRequest request = requestBuilder.build();

                System.out.printf("Running inference [Model: %s]%n", modelId);
                if (providerId != null) {
                    System.out.printf("Provider: %s%n", providerId);
                }

                if (stream) {
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
                                        }
                                    },
                                    error -> System.err.println("\n" + YELLOW + "Error: " + RESET + error.getMessage()),
                                    () -> {
                                        long duration = System.currentTimeMillis() - startTime;
                                        double tps = (tokenCount.get() / (duration / 1000.0));
                                        System.out.printf(
                                                "%n" + DIM + "[Duration: %d ms, Tokens: %d, Speed: %.2f t/s]" + RESET
                                                        + "%n",
                                                duration, tokenCount.get(), tps);
                                    });
                } else {
                    // Sync mode
                    InferenceResponse response = sdk.createCompletion(request);
                    printResponse(response, startTime);
                }

            }

        } catch (Exception e) {
            System.err.println("\nInference failed: " + e.getMessage());
            if (e.getCause() != null && !e.getMessage().contains(e.getCause().getMessage())) {
                System.err.println("Detail: " + e.getCause().getMessage());
            }
        }
    }

    private void printResponse(InferenceResponse response, long startTime) {
        System.out.println();
        System.out.println(GREEN + response.getContent() + RESET);
        System.out.printf("%n" + DIM + "[Duration: %d ms, Tokens: %d, Speed: %.2f t/s]" + RESET + "%n",
                response.getDurationMs(),
                response.getTokensUsed(),
                (response.getTokensUsed() / (response.getDurationMs() / 1000.0)));
    }
}
