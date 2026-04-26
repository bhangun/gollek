package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import tech.kayys.gollek.cli.chat.ChatUIRenderer;
import tech.kayys.gollek.cli.util.PluginAvailabilityChecker;
import tech.kayys.gollek.cli.audit.QuantAuditRecord;
import tech.kayys.gollek.cli.audit.QuantAuditRenderer;
import tech.kayys.gollek.cli.audit.QuantAuditTrail;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;
import tech.kayys.gollek.plugin.kernel.KernelPlatformDetector;

/**
 * Run inference command using GollekSdk.
 * Usage: gollek run --model <model> --prompt <prompt> [--provider
 * litert|llamacpp|gemini] [--stream]
 */
@Dependent
@Unremovable
@Command(name = "run", description = "Run inference using a specified model")
public class RunCommand implements Runnable {

    @ParentCommand
    GollekCommand parentCommand;

    @Inject
    GollekSdk sdk;
    @Inject
    Instance<HuggingFaceClient> hfClientInstance;
    @Inject
    ProviderRegistry providerRegistry;
    @Inject
    PluginAvailabilityChecker pluginChecker;
    @Inject
    ChatUIRenderer uiRenderer;
    @Inject
    tech.kayys.gollek.cli.util.ModelImporter modelImporter;

    @Option(names = { "-m", "--model" }, description = "Model ID for repository resolution (e.g., huggingface ID)")
    public String modelId;

    @Option(names = { "--modelFile" }, description = "Path to a local model file (.gguf, .tflite, .task, .litertlm)")
    public String modelFile;

    @Option(names = { "--modelDir" }, description = "Path to a local model directory (Safetensors)")
    public String modelDir;

    @Option(names = { "-p", "--prompt" }, description = "Input prompt", required = true)
    public String prompt;

    @Option(names = {
            "--provider" }, description = "Provider: native, litert, llamacpp, safetensor, libtorch(experimental), gemini, openai, anthropic, cerebras. Omit for auto-detection.", arity = "0..1", fallbackValue = "")
    String providerId;

    @Option(names = {
            "--import" }, description = "Import (move) the model file/dir into the gollek model repository (~/.gollek/models/)")
    boolean importModel;

    @Option(names = {
            "--copy" }, description = "Copy the model file/dir into the gollek model repository (~/.gollek/models/)")
    boolean copyModel;

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

    @Option(names = {
            "--enable-json" }, description = "Emit OpenAI-compatible SSE JSON for streamed chunks", defaultValue = "false")
    boolean enableJsonSse;

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate", defaultValue = "256")
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

    @Option(names = {
            "--model-path" }, description = "Path to a custom model file (bypasses repository lookup, prefer --modelFile)")
    String modelPath;

    @Option(names = {
            "--convert-mode" }, description = "Checkpoint conversion mode: auto or off", defaultValue = "auto")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. f16, q8_0, f32)")
    String ggufOutType;

    @Option(names = { "-b", "--branch" }, description = "HuggingFace branch/revision (e.g. main, fp16, onnx)")
    String branch;

    @Option(names = { "--force" }, description = "Force re-download and replace existing files", defaultValue = "false")
    boolean force;

    @Option(names = { "--format" }, description = "Preferred model format (e.g., safetensors, gguf, onnx)")
    String format;

    @Option(names = { "--direct" }, description = "Force use of native Safetensor direct engine", defaultValue = "false")
    boolean direct;

    @Option(names = { "--gguf-quant" }, description = "Specify quantization for GGUF conversion (e.g. Q4_K_M)")
    String ggufQuant;

    @Option(names = { "--gguf" }, description = "Force GGUF conversion and usage", defaultValue = "false")
    boolean forceGguf;

    @Option(names = { "--quantize" }, description = "Enable JIT quantization during inference (bnb, turbo, awq, gptq, autoround)")
    String quantizeStrategy;

    @Option(names = { "--quantize-bits" }, description = "Bit width for JIT quantization (default: 4)", defaultValue = "4")
    int quantizeBits;

    // Stable Diffusion specific parameters
    @Option(names = { "--seed" }, description = "Random seed for image generation (default: random)")
    Long seed;

    @Option(names = { "--output", "-o" }, description = "Output image file path (default: output.png)")
    String outputPath;

    @Option(names = { "--steps" }, description = "Number of denoising steps (default: 20)")
    Integer steps;

    @Option(names = { "--guidance-scale", "--cfg" }, description = "Classifier-free guidance scale (default: 7.5)")
    Float guidanceScale;

    @Option(names = { "--width" }, description = "Output image width in pixels (default: 512, must be multiple of 64)")
    Integer width;

    @Option(names = {
            "--height" }, description = "Output image height in pixels (default: 512, must be multiple of 64)")
    Integer height;

    @Override
    public void run() {
        try {
            // Check plugin availability first
            if (!pluginChecker.hasProviders() && !pluginChecker.hasRunnerPlugins()) {
                System.err.println(pluginChecker.getNoPluginsError());
                System.exit(1);
                return;
            }

            // Auto-detect and display kernel platform
            if (parentCommand != null && parentCommand.verbose) System.out.println("Starting platform detection...");
            KernelPlatform detectedPlatform;
            try {
                detectedPlatform = KernelPlatformDetector.detect();
            } catch (Throwable t) {
                System.err.println("CRITICAL: Platform detection failed: " + t.getMessage());
                t.printStackTrace();
                System.exit(1);
                return;
            }

            if (parentCommand == null || !parentCommand.verbose) {
                System.out.println(ChatUIRenderer.CYAN + "Platform: " + detectedPlatform.getDisplayName() + ChatUIRenderer.RESET);
                if (detectedPlatform.isCpu()) {
                    System.out.println(ChatUIRenderer.YELLOW + "⚠️  Running on CPU (GPU acceleration not available)" + ChatUIRenderer.RESET);
                } else {
                    System.out.println(ChatUIRenderer.GREEN + "✓ GPU acceleration enabled" + ChatUIRenderer.RESET);
                }
                System.out.println();
            }


        long startTime = System.currentTimeMillis();

        // Check if specific provider is requested but not available
        if (providerId != null && !providerId.trim().isEmpty()) {
            if (!pluginChecker.hasProvider(providerId)) {
                System.err.println(pluginChecker.getProviderNotFoundError(providerId));
                System.exit(1);
                return;
            }
        }

        if (parentCommand != null) {
            parentCommand.applyRuntimeOverrides();
        }
            configureCheckpointConversionPreference();

            boolean customModelPathUsed = false;
            String finalLocalPath = null;
            uiRenderer.setJsonMode(enableJsonSse);
            uiRenderer.printBanner();

            if (isMcpProvider()) {
                System.out.println("MCP provider selected; skipping local model lookup.");
            } else if (modelFile != null && !modelFile.isBlank()) {
                // --- Resolve from --modelFile ---
                Path filePath = Paths.get(modelFile);
                if (!Files.exists(filePath)) {
                    System.err.println("Error: Model file not found: " + modelFile);
                    return;
                }

                // Handle --import or --copy
                if (importModel || copyModel) {
                    filePath = modelImporter.importModel(filePath, importModel, false);
                    System.out.println((importModel ? "Imported" : "Copied") + " model to: " + filePath.toAbsolutePath());
                }

                modelId = filePath.toAbsolutePath().toString();
                finalLocalPath = modelId;
                customModelPathUsed = true;

                // Auto-detect provider from extension
                if ("native".equals(providerId)) {
                    if (modelFile.endsWith(".gguf")) {
                        providerId = "native";
                    } else if (modelFile.endsWith(".litertlm") || modelFile.endsWith(".tflite") || modelFile.endsWith(".task")) {
                        providerId = "litert";
                    }
                }
                System.out.println("Model path: " + filePath.toAbsolutePath());

            } else if (modelDir != null && !modelDir.isBlank()) {
                // --- Resolve from --modelDir ---
                Path dirPath = Paths.get(modelDir);
                if (!Files.isDirectory(dirPath)) {
                    System.err.println("Error: Model directory not found: " + modelDir);
                    return;
                }

                // Handle --import or --copy
                if (importModel || copyModel) {
                    dirPath = modelImporter.importModel(dirPath, importModel, true);
                    System.out.println((importModel ? "Imported" : "Copied") + " model to: " + dirPath.toAbsolutePath());
                }

                modelId = dirPath.toAbsolutePath().toString();
                customModelPathUsed = true;
                providerId = "safetensor";
                System.out.println("Model dir: " + dirPath.toAbsolutePath());

            } else if (modelPath != null && !modelPath.isEmpty()) {
                Path customModelPath = Paths.get(modelPath);
                if (!Files.exists(customModelPath)) {
                    System.err.println("Error: Model file not found: " + modelPath);
                    return;
                }
                System.out.println("Using model from: " + customModelPath.toAbsolutePath());
                modelId = customModelPath.toAbsolutePath().toString();
                customModelPathUsed = true;
            } else {
                // Prepare model using SDK (this handles pulling, registration, and conversion)
                try {
                    // 1. Initial preparation (pulls and registers if missing)
                    String quant = ggufQuant != null ? ggufQuant : (ggufOutType != null ? ggufOutType : "Q8_0");
                    var resolution = sdk.prepareModel(modelId, format, false, quant, progress -> {
                        if (progress.getTotal() > 0) {
                            System.out.printf("\r%s %s %d%% (%d/%d MB)",
                                    ChatUIRenderer.CYAN + progress.getStatus() + ChatUIRenderer.RESET,
                                    progress.getProgressBar(20),
                                    progress.getPercentComplete(),
                                    progress.getCompleted() / 1024 / 1024,
                                    progress.getTotal() / 1024 / 1024);
                        } else {
                            System.out.print("\r" + ChatUIRenderer.CYAN + progress.getStatus() + ChatUIRenderer.RESET);
                        }
                    });
                    System.out.println();

                    // 2. Handle explicit GGUF conversion quest if requested
                    if (forceGguf && !"GGUF".equalsIgnoreCase(resolution.getInfo().getFormat())) {
                        System.out.println("Processing GGUF conversion request (Quantization: " + quant + ")...");
                        resolution = sdk.convertToGguf(resolution, quant, progress -> {
                            System.out.print("\r" + ChatUIRenderer.CYAN + progress.getStatus() + " " + progress.getPercentComplete() + "%" + ChatUIRenderer.RESET);
                        });
                        System.out.println();
                    }

                    modelId = resolution.getModelId();
                    if (resolution.getLocalPath() != null) {
                        finalLocalPath = resolution.getLocalPath();
                        String displayPath = finalLocalPath;
                        String userHome = System.getProperty("user.home");
                        if (userHome != null && displayPath.startsWith(userHome)) {
                            displayPath = "~" + displayPath.substring(userHome.length());
                        }
                        System.out.println("Model ready at: " + displayPath);
                        customModelPathUsed = true; // Signals that we have a local path for metadata
                    }
                    
                    // Auto-select provider based on final format
                    String resolvedFormat = resolution.getInfo().getFormat();
                    maybeAutoSelectProviderByFormat(resolvedFormat, finalLocalPath);
                    
                    // If user explicitly requested a format, let's make sure our providerId matches that intent
                    // even if the above auto-selection was confused.
                    if (this.format != null && !this.format.isBlank()) {
                        String intendedProvider = providerForFormat(this.format, finalLocalPath);
                        if (intendedProvider != null && !intendedProvider.equalsIgnoreCase(providerId)) {
                             if (ensureProviderHealthy(intendedProvider)) {
                                 providerId = intendedProvider;
                                 sdk.setPreferredProvider(intendedProvider);
                             }
                        }
                    }

                } catch (Exception e) {
                    System.err.println("\nError: Failed to prepare model: " + e.getMessage());
                    if (parentCommand != null && parentCommand.verbose) {
                        e.printStackTrace();
                    }
                    return;
                }
            }

            // Smart quantization suggestion for large models
            tech.kayys.gollek.cli.util.QuantSuggestionDetector.suggestIfNeeded(
                    modelId, finalLocalPath, quantizeStrategy, false);

            // Set preferred provider if specified
            if (direct) {
                providerId = "safetensor";
                sdk.setPreferredProvider("safetensor");
            } else if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
            }
            if (!ensureProviderReady()) {
                return;
            }
            printCompatibilityHintBeforeInference();
            // Check if provider supports streaming if requested
            // TODO: Add isStreamingSupported() method to GollekSdk
            // if (stream && !sdk.isStreamingSupported()) {
            // System.out.println(
            // "Provider '" + providerId + "' does not support streaming; switching to
            // non-streaming mode.");
            // stream = false;
            // }

            // Build request
            InferenceRequest.Builder requestBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .model(modelId)
                    .temperature(temperature)
                    .topP(topP)
                    .topK(topK)
                    .repeatPenalty(repeatPenalty)
                    .jsonMode(jsonMode)
                    .maxTokens(maxTokens)
                    .streaming(stream);

            if (customModelPathUsed && finalLocalPath != null) {
                requestBuilder.parameter("model_path", finalLocalPath);
            }

            if (mirostat > 0) {
                requestBuilder.mirostat(mirostat);
            }
            if (grammar != null && !grammar.isEmpty()) {
                requestBuilder.grammar(grammar);
            }

            // Quantization parameters for JIT
            if (quantizeStrategy != null && !quantizeStrategy.isBlank()) {
                requestBuilder.parameter("quantize_strategy", quantizeStrategy);
                requestBuilder.parameter("quantize_bits", quantizeBits);
            }

            // Stable Diffusion parameters
            if (seed != null) {
                requestBuilder.parameter("seed", seed);
            }
            if (steps != null) {
                requestBuilder.parameter("steps", steps);
            }
            if (format != null) {
                requestBuilder.parameter("format", format);
            }
            if (guidanceScale != null) {
                requestBuilder.parameter("guidance_scale", guidanceScale);
            }
            if (outputPath != null) {
                requestBuilder.parameter("output_path", outputPath);
            }
            if (width != null) {
                if (width % 64 != 0) {
                    System.err.println("Warning: Width should be a multiple of 64 for Stable Diffusion. Adjusting to " + ((width + 32) / 64 * 64));
                    width = (width + 32) / 64 * 64;
                }
                requestBuilder.parameter("width", width);
            }
            if (height != null) {
                if (height % 64 != 0) {
                    System.err.println("Warning: Height should be a multiple of 64 for Stable Diffusion. Adjusting to " + ((height + 32) / 64 * 64));
                    height = (height + 32) / 64 * 64;
                }
                requestBuilder.parameter("height", height);
            }

            if (systemPrompt != null && !systemPrompt.isEmpty()) {
                requestBuilder.message(Message.system(systemPrompt));
            } else {
                // Default system prompt to establish identity and language without triggering negative prompting hallucinations
                requestBuilder.message(Message.system("I'm gollek, and you are using model " + modelId + " to serve you."));
            }
            requestBuilder.message(Message.user(prompt));

            if (providerId != null && !providerId.isEmpty()) {
                requestBuilder.preferredProvider(providerId);
            }

            requestBuilder.cacheBypass(noCache);

            InferenceRequest request = requestBuilder.build();
            boolean directProviderBypass = "safetensor".equalsIgnoreCase(providerId);

            if (!enableJsonSse) {
                uiRenderer.printModelInfo(modelId, providerId, format, null, false);
            }

            if (directProviderBypass) {
                InferenceResponse response = inferDirectWithProvider(providerId, request);
                printResponse(response, startTime);
                return;
            }

            if (stream) {
                java.io.ByteArrayOutputStream imageBuffer = new java.io.ByteArrayOutputStream();
                CountDownLatch latch = new CountDownLatch(1);
                // Streaming mode
                java.util.concurrent.atomic.AtomicReference<String> lastProgress = new java.util.concurrent.atomic.AtomicReference<>();
                java.util.concurrent.atomic.AtomicReference<java.util.Map<String, Object>> metricsRef = new java.util.concurrent.atomic.AtomicReference<>();

                java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(0);
                sdk.streamCompletion(request)
                        .subscribe().with(
                                chunk -> {
                                    // Debug: log all chunks to see what we're receiving
                                    if (parentCommand != null && parentCommand.verbose) {
                                        System.err.println("\n[DEBUG] Chunk received: modality=" + chunk.modality() + 
                                            ", index=" + chunk.index() + 
                                            ", hasImageDelta=" + (chunk.imageDeltaBase64() != null) +
                                            ", hasDelta=" + (chunk.getDelta() != null));
                                    }

                                    if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                        metricsRef.set(chunk.metadata());
                                    }
                                    
                                    if (chunk.modality() == tech.kayys.gollek.spi.model.ModalityType.IMAGE) {
                                        if (parentCommand != null && parentCommand.verbose) {
                                            System.err.println("\n[DEBUG] IMAGE chunk detected, size: " + 
                                                (chunk.imageDeltaBase64() != null ? chunk.imageDeltaBase64().length() : 0) + " chars");
                                        }
                                        if (chunk.imageDeltaBase64() != null) {
                                            try {
                                                byte[] decoded = java.util.Base64.getDecoder().decode(chunk.imageDeltaBase64());
                                                imageBuffer.write(decoded);
                                                if (parentCommand != null && parentCommand.verbose) {
                                                    System.err.println("\n[DEBUG] Image buffer size: " + imageBuffer.size() + " bytes");
                                                }
                                            } catch (Exception e) {
                                                System.err.println("\nFailed to decode image chunk: " + e.getMessage());
                                            }
                                        }
                                        return;
                                    }

                                    String delta = chunk.getDelta();
                                    if (delta != null) {
                                        // Check if this is a progress message (SD-style)
                                        if (delta.startsWith("[") && delta.contains("]")) {
                                            lastProgress.set(delta);
                                            if (!enableJsonSse) {
                                                System.out.print("\r" + ChatUIRenderer.CYAN + delta + ChatUIRenderer.RESET + "  ");
                                                System.out.flush();
                                            }
                                        } else if (enableJsonSse) {
                                            printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                                        } else {
                                            System.out.print(delta);
                                            System.out.flush();
                                        }
                                        tokenCount.incrementAndGet();
                                    }
                                },
                                error -> {
                                    uiRenderer.printError(error.getMessage(), false);
                                    printProviderHintFromError(error);
                                    latch.countDown();
                                },
                                () -> {
                                    long duration = System.currentTimeMillis() - startTime;

                                    if (parentCommand != null && parentCommand.verbose) {
                                        System.err.println("\n[DEBUG] onComplete - imageBuffer.size()=" + imageBuffer.size() + " bytes");
                                    }
                                    
                                    if (imageBuffer.size() > 0) {
                                        try {
                                            Path outPath = (outputPath != null && !outputPath.isEmpty()) 
                                                ? Path.of(outputPath) 
                                                : Path.of("output.png");
                                            Files.write(outPath, imageBuffer.toByteArray());
                                            System.out.println("\n" + ChatUIRenderer.GREEN + ChatUIRenderer.BOLD + "✓ Image saved to: " + ChatUIRenderer.RESET + outPath.toAbsolutePath());
                                            
                                            // Auto-open image on macOS
                                            autoOpenImage(outPath);
                                        } catch (Exception e) {
                                            System.err.println("\nFailed to save image: " + e.getMessage());
                                        }
                                    }

                                    double tps = (tokenCount.get() / (duration / 1000.0));
                                    if (enableJsonSse) {
                                        printOpenAiSseFinal(request.getRequestId(), request.getModel());
                                    } else {
                                        uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, false);
                                        uiRenderer.printBenchmarks(metricsRef.get(), false);
                                    }
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

        } catch (Throwable e) {
            System.err.println("\n[FATAL] RunCommand failed with unhandled error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
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

    private boolean isMcpProvider() {
        return providerId != null && "mcp".equalsIgnoreCase(providerId.trim());
    }

    private void printResponse(InferenceResponse response, long startTime) {
        System.out.println();
        System.out.println(ChatUIRenderer.GREEN + response.getContent() + ChatUIRenderer.RESET);
        double seconds = Math.max(response.getDurationMs() / 1000.0, 0.001);
        double tps = response.getTokensUsed() / seconds;
        uiRenderer.printStats(response.getTokensUsed(), response.getDurationMs() / 1000.0, tps, false);
    }

    private void printOpenAiSseDelta(String requestId, String model, String delta) {
        long created = System.currentTimeMillis() / 1000L;
        String id = "chatcmpl-" + (requestId != null ? requestId : UUID.randomUUID().toString());
        String payload = "{\"id\":\"" + escapeJson(id)
                + "\",\"object\":\"chat.completion.chunk\""
                + ",\"created\":" + created
                + ",\"model\":\"" + escapeJson(model != null ? model : "") + "\""
                + ",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + escapeJson(delta)
                + "\"},\"finish_reason\":null}]}";
        System.out.println("data: " + payload);
        System.out.flush();
    }

    private void printOpenAiSseFinal(String requestId, String model) {
        long created = System.currentTimeMillis() / 1000L;
        String id = "chatcmpl-" + (requestId != null ? requestId : UUID.randomUUID().toString());
        String payload = "{\"id\":\"" + escapeJson(id)
                + "\",\"object\":\"chat.completion.chunk\""
                + ",\"created\":" + created
                + ",\"model\":\"" + escapeJson(model != null ? model : "") + "\""
                + ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}";
        System.out.println("data: " + payload);
        System.out.println("data: [DONE]");
        System.out.flush();
    }

    private String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder out = new StringBuilder(value.length() + 16);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '\\':
                    out.append("\\\\");
                    break;
                case '"':
                    out.append("\\\"");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
            }
        }
        return out.toString();
    }

    private InferenceResponse inferDirectWithProvider(String id, InferenceRequest request) {
        String providerModel = resolveProviderModel(request);
        ProviderRequest providerRequest = buildDirectProviderRequest(request, providerModel);
        try {
            return inferDirect(id, providerRequest);
        } catch (RuntimeException primary) {
            if (shouldFallbackToTorch(id, providerModel)) {
                try {
                    System.out.println("DJL checkpoint load failed; falling back to libtorch (experimental)...");
                    return inferDirect("libtorch", providerRequest);
                } catch (RuntimeException fallback) {
                    throw new RuntimeException(
                            "DJL checkpoint load failed and libtorch fallback also failed: " + fallback.getMessage(),
                            primary);
                }
            }
            throw primary;
        }
    }

    private InferenceResponse inferDirect(String id, ProviderRequest providerRequest) {
        Optional<LLMProvider> providerOpt = providerRegistry.getProvider(id);
        if (providerOpt.isEmpty()) {
            throw new RuntimeException("Provider not available: " + id);
        }
        LLMProvider provider = providerOpt.get();
        return provider.infer(providerRequest)
                .await()
                .atMost(Duration.ofSeconds(180));
    }

    private ProviderRequest buildDirectProviderRequest(InferenceRequest request, String providerModel) {
        return ProviderRequest.builder()
                .model(providerModel)
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .streaming(false)
                .timeout(Duration.ofSeconds(120))
                .metadata("request_id", request.getRequestId())
                .metadata("tenantId", "community")
                .build();
    }

    private String resolveProviderModel(InferenceRequest request) {
        Object modelPathParam = request.getParameters().get("model_path");
        if (modelPathParam != null && !String.valueOf(modelPathParam).isBlank()) {
            return String.valueOf(modelPathParam);
        }
        return request.getModel();
    }

    private boolean shouldFallbackToTorch(String providerId, String providerModel) {
        if (providerModel == null) {
            return false;
        }
        String normalized = providerModel.toLowerCase();
        return normalized.endsWith(".safetensors")
                || normalized.endsWith(".safetensor")
                || normalized.endsWith(".bin")
                || normalized.endsWith(".pth");
    }

    private void maybeAutoSelectProvider() {
        try {
            var resolvedOpt = LocalModelResolver.resolve(sdk, modelId, branch, format);
            if (resolvedOpt.isEmpty()) {
                return;
            }
            var resolved = resolvedOpt.get();
            this.format = resolved.info().getFormat();
            String path = resolved.localPath() != null ? resolved.localPath().toString() : modelId;
            maybeAutoSelectProviderByFormat(this.format, path);
        } catch (Exception ignored) {
            // Keep default router behavior when format/provider probing is not available.
        }
    }

    private void maybeAutoSelectProviderByFormat(String format, String path) throws SdkException {
        // Do not overwrite user-specified provider
        if (this.providerId != null && !this.providerId.trim().isEmpty()) {
            return;
        }

        String inferredProvider = providerForFormat(format, path);
        if (inferredProvider == null || inferredProvider.isBlank()) {
            return;
        }
        if (!isProviderHealthy(inferredProvider)) {
            System.err.println("Warning: Auto-selected provider '" + inferredProvider + "' is not available or healthy. Falling back to default routing.");
            return;
        }
        sdk.setPreferredProvider(inferredProvider);
        providerId = inferredProvider;
    }

    private String providerForFormat(String format, String path) {
        if (format == null || format.isBlank()) {
            return null;
        }
        String normalized = format.trim().toUpperCase();
        
        // Special case: Stable Diffusion (requires ONNX or specialized runner)
        if ("SAFETENSORS".equals(normalized) || "SAFETENSOR".equals(normalized)) {
            try {
                Path p = path != null ? Path.of(path) : null;
                boolean isSd = p != null && tech.kayys.gollek.spi.model.ModelFormatDetector.isStableDiffusion(p);
                System.err.println("SD Detection: format=" + normalized + ", path=" + path + " -> isSd=" + isSd);
                if (isSd) {
                    return "onnx";
                }
            } catch (Exception ignored) {}
        }

        return switch (normalized) {
            case "GGUF" -> "gguf";
            case "TORCHSCRIPT" -> "libtorch";
            case "PYTORCH" -> "libtorch";
            case "SAFETENSOR" -> "safetensor";
            case "SAFETENSORS" -> "safetensor";
            case "ONNX" -> "onnx";
            default -> null;
        };
    }

    private boolean isProviderHealthy(String id) {
        try {
            return sdk.listAvailableProviders().stream()
                    .filter(p -> id.equalsIgnoreCase(p.id()))
                    .findFirst()
                    .map(p -> p.healthStatus() == tech.kayys.gollek.spi.provider.ProviderHealth.Status.HEALTHY)
                    .orElse(false);
        } catch (Exception e) {
            return false;
        }
    }

    private boolean ensureProviderReady() {
        try {
            System.err.println("ensureProviderReady: providerId=" + providerId + ", modelId=" + modelId);
            if (providerId != null && !providerId.isBlank()) {
                boolean healthy = ensureProviderHealthy(providerId);
                System.err.println("ensureProviderReady: providerId=" + providerId + " healthy=" + healthy);
                if (healthy) return true;
            }
            var modelInfoOpt = LocalModelResolver.resolve(sdk, modelId, branch, format)
                    .map(LocalModelResolver.ResolvedModel::info);
            if (modelInfoOpt.isEmpty()) {
                return true;
            }
            var modelInfo = modelInfoOpt.get();
            this.format = modelInfo.getFormat();
            if (isCheckpointOnlyFormat(this.format)) {
                String checkpointProvider = providerForFormat(this.format, modelId);
                if (checkpointProvider != null && ensureProviderHealthy(checkpointProvider)) {
                    providerId = checkpointProvider;
                    sdk.setPreferredProvider(checkpointProvider);
                    return true;
                }
                if (offline) {
                    System.err.printf(
                            "Model '%s' uses checkpoint format '%s' and cannot run in offline Java runtime.%n",
                            modelId, format);
                    System.err
                            .println("Use a GGUF or TorchScript model, or rerun without --offline for fallback pull.");
                    return false;
                }
                if (tryRefreshCompatibleModel()) {
                    modelInfoOpt = LocalModelResolver.resolve(sdk, modelId, branch, format)
                            .map(LocalModelResolver.ResolvedModel::info);
                    if (modelInfoOpt.isPresent()) {
                        format = modelInfoOpt.get().getFormat();
                    }
                }
                if (isCheckpointOnlyFormat(format)) {
                    checkpointProvider = providerForFormat(format, modelId);
                    if (checkpointProvider != null && ensureProviderHealthy(checkpointProvider)) {
                        providerId = checkpointProvider;
                        sdk.setPreferredProvider(checkpointProvider);
                        return true;
                    }
                    // experimental fallback
                    if (ensureProviderHealthy("libtorch")) {
                        providerId = "libtorch";
                        sdk.setPreferredProvider("libtorch");
                        return true;
                    }
                    System.err.printf("Model '%s' uses unsupported runtime format '%s'.%n", modelId, format);
                    System.err.println(
                            "Use a GGUF/TorchScript model. Checkpoint runtime requires DJL (recommended) or --provider libtorch (experimental).");
                    return false;
                }
            }

            String inferredProvider = providerForFormat(format, modelId);
            if (inferredProvider == null || inferredProvider.isBlank()) {
                return true;
            }
            if (ensureProviderHealthy(inferredProvider)) {
                providerId = inferredProvider;
                sdk.setPreferredProvider(inferredProvider);
                return true;
            }
            return false;
        } catch (Exception ignored) {
            return true;
        }
    }

    private boolean isCheckpointOnlyFormat(String format) {
        if (format == null) {
            return false;
        }
        String normalized = format.trim().toUpperCase();
        return normalized.equals("PYTORCH") || normalized.equals("SAFETENSORS") || normalized.equals("SAFETENSOR");
    }

    private void configureCheckpointConversionPreference() {
        String mode = convertMode == null ? "auto" : convertMode.trim().toLowerCase();
        if (mode.equals("off")) {
            System.setProperty("gollek.gguf.converter.auto", "false");
        } else {
            System.setProperty("gollek.gguf.converter.auto", "true");
        }
        if (ggufOutType != null && !ggufOutType.isBlank()) {
            System.setProperty("gollek.gguf.converter.outtype", ggufOutType.trim().toLowerCase());
        }
    }

    private boolean tryRefreshCompatibleModel() {
        if (!(modelId.contains("/") || modelId.startsWith("hf:"))) {
            return false;
        }
        System.out.println("Checkpoint-only model detected. Trying GGUF/TorchScript fallback...");
        for (String spec : buildPullSpecs(modelId)) {
            try {
                sdk.pullModel(spec, branch, force, null);
            } catch (Exception e) {
                if (isFatalPullError(e)) {
                    break;
                }
            }
        }
        String normalized = modelId.startsWith("hf:") ? modelId.substring(3) : modelId;
        for (String candidate : java.util.List.of(modelId, normalized, modelId + "-GGUF", normalized + "-GGUF")) {
            try {
                var resolved = LocalModelResolver.resolve(sdk, candidate, branch, format);
                if (resolved.isPresent()) {
                    String fmt = resolved.get().info().getFormat();
                    if (!isCheckpointOnlyFormat(fmt)) {
                        modelId = resolved.get().modelId();
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
            printGenericProviderSetupHint(provider);
            return false;
        }
        if (info.get().healthStatus() != ProviderHealth.Status.UNHEALTHY) {
            return true;
        }
        printProviderSetupHint(info.get());
        return false;
    }

    private void printGenericProviderSetupHint(String providerId) {
        if ("libtorch".equalsIgnoreCase(providerId)) {
            System.err.println(
                    "LibTorch runtime is not loaded. Set GOLLEK_LIBTORCH_LIB_PATH (or LIBTORCH_PATH) and include gollek-runner-libtorch.");
        } else if ("onnx".equalsIgnoreCase(providerId)) {
            System.err.println(
                    "ONNX Runtime is not available or enabled. Run 'brew install onnxruntime' (macOS) and ensure gollek-runner-onnx is active.");
        } else if ("llamacpp".equalsIgnoreCase(providerId)) {
            System.err.println(
                    "GGUF runtime is not loaded. Set GOLLEK_LLAMA_LIB_DIR/GOLLEK_LLAMA_LIB_PATH and include gollek-ext-runner-gguf.");
        }
    }

    private void printCompatibilityHintBeforeInference() {
        if (providerId == null || providerId.isBlank() || modelId == null || modelId.isBlank()) {
            return;
        }

        String provider = providerId.trim().toLowerCase(java.util.Locale.ROOT);
        String model = modelId.trim().toLowerCase(java.util.Locale.ROOT);
        String modelName = Path.of(modelId).getFileName() != null ? Path.of(modelId).getFileName().toString() : modelId;

        if ("safetensor".equals(provider)) {
            if (looksLikeMultimodalModel(model)) {
                System.err.println(
                        "Hint: provider 'safetensor' is text-checkpoint oriented. This model looks multimodal/VLM and may fail.");
                System.err.println(
                        "Hint: try a text-only checkpoint, or use a provider/runtime with multimodal support.");
            } else if (model.endsWith(".gguf") || model.endsWith(".pt") || model.endsWith(".pth")) {
                System.err.println(
                        "Hint: provider 'safetensor' expects .safetensor/.safetensors weights, but got: " + modelName);
            }
            return;
        }

        if ("llamacpp".equals(provider) && !(model.endsWith(".gguf") || model.contains("/models/gguf/"))) {
            System.err.println("Hint: provider 'llamacpp' works best with GGUF models (.gguf).");
        }
    }

    private boolean looksLikeMultimodalModel(String normalizedModel) {
        return normalizedModel.contains("vlm")
                || normalizedModel.contains("vision")
                || normalizedModel.contains("llava")
                || normalizedModel.contains("idefics")
                || normalizedModel.contains("qwen-vl")
                || normalizedModel.contains("smolvlm");
    }

    private void printProviderHintFromError(Throwable throwable) {
        String detail = describeError(throwable).toLowerCase();
        if (detail.contains("provider not available: libtorch")) {
            printGenericProviderSetupHint("libtorch");
        } else if (detail.contains("provider not available: llamacpp")) {
            printGenericProviderSetupHint("llamacpp");
        } else if (detail.contains("429") || detail.contains("resource_exhausted") || detail.contains("quota")) {
            System.err.println(
                    "Hint: Provider quota/rate limit reached. Wait for retry window, or switch provider/model.");
        }
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
        if ("libtorch".equalsIgnoreCase(id)) {
            System.err.println(
                    "Set GOLLEK_LIBTORCH_LIB_PATH (or LIBTORCH_PATH) to your libtorch native library directory.");
        } else if ("llamacpp".equalsIgnoreCase(id)) {
            System.err.println("Set GOLLEK_LLAMA_LIB_DIR or GOLLEK_LLAMA_LIB_PATH to llama.cpp native libraries.");
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

    /**
     * Automatically opens the generated image using the system's default viewer.
     * Supports macOS (open), Linux (xdg-open), and Windows (cmd /c start).
     */
    private void autoOpenImage(Path imagePath) {
        try {
            String os = System.getProperty("os.name").toLowerCase();
            ProcessBuilder pb = null;

            if (os.contains("mac")) {
                pb = new ProcessBuilder("open", imagePath.toAbsolutePath().toString());
            } else if (os.contains("linux")) {
                pb = new ProcessBuilder("xdg-open", imagePath.toAbsolutePath().toString());
            } else if (os.contains("windows")) {
                pb = new ProcessBuilder("cmd", "/c", "start", imagePath.toAbsolutePath().toString());
            }

            if (pb != null) {
                pb.inheritIO();
                Process process = pb.start();
                // Don't wait for viewer to finish
            }
        } catch (Exception e) {
            // Silently fail - opening viewer is optional
            if (parentCommand != null && parentCommand.verbose) {
                System.err.println("Debug: Failed to auto-open image: " + e.getMessage());
            }
        }
    }
}
