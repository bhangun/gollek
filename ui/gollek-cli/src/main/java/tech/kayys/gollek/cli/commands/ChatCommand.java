package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import io.quarkus.arc.Unremovable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.cli.chat.*;
import tech.kayys.gollek.sdk.model.ModelResolver;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.sdk.model.ModelResolution;
import tech.kayys.gollek.cli.util.CliInferenceFeatures;

import org.jline.console.CmdDesc;
import org.jline.utils.AttributedString;

import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.*;
import tech.kayys.gollek.cli.runtime.CliMetalRuntime;
import tech.kayys.gollek.cli.util.PluginAvailabilityChecker;
import tech.kayys.gollek.plugin.kernel.KernelPlatform;
import tech.kayys.gollek.plugin.kernel.KernelPlatformDetector;

/**
 * Interactive chat session using GollekSdk.
 */
@Dependent
@Unremovable
@Command(name = "chat", description = "Starts an interactive chat session.")
public class ChatCommand implements Runnable {

    @ParentCommand
    GollekCommand parentCommand;

    @Inject
    ChatTerminalHandler terminalHandler;
    @Inject
    ChatUIRenderer uiRenderer;
    @Inject
    ChatSessionManager sessionManager;
    @Inject
    ChatCommandHandler commandHandler;

    @Inject
    GollekSdk sdk;
    @Inject
    PluginAvailabilityChecker pluginChecker;

    @Option(names = { "-m", "--model" }, description = "Model ID for repository resolution (e.g., huggingface ID)")
    public String modelId;

    @Option(names = { "--modelFile" }, description = "Path to a local model file (.litertlm, .tflite, .task, .gguf)")
    public String modelFile;

    @Option(names = { "--modelDir" }, description = "Path to a local model directory (Safetensors)")
    public String modelDir;

    @Option(names = { "-p", "--provider" }, description = "Provider ID (default: auto). Options: native, safetensor, gguf, cerebras, mistral, openai, gemini")
    public String providerId;

    @Option(names = { "--import" }, description = "Import (move) the model file/dir into the gollek model repository (~/.gollek/models/)")
    public boolean importModel;

    @Option(names = { "--copy" }, description = "Copy the model file/dir into the gollek model repository (~/.gollek/models/)")
    public boolean copyModel;

    @Option(names = { "-s", "--system" }, description = "System prompt")
    public String systemPrompt;

    @Option(names = { "--tool-file", "--tools-file" }, split = ",", description = "JSON tool definition file(s); accepts OpenAI-style tools arrays or Gollek ToolDefinition JSON")
    public List<String> toolFiles;

    @Option(names = { "--tool-choice" }, description = "Tool choice policy such as auto, none, required, or a JSON object")
    public String toolChoice;

    @Option(names = { "--mcp-tool", "--mcp-tools" }, split = ",", description = "MCP tool name(s) to expose, optionally as server/tool or server:tool")
    public List<String> mcpTools;

    @Option(names = { "--rag-context" }, split = "\\|", description = "Inline retrieval context block(s), separated with | when repeated in one argument")
    public List<String> ragContexts;

    @Option(names = { "--rag-file", "--context-file" }, split = ",", description = "Text file(s) to inject as retrieval context")
    public List<String> ragFiles;

    @Option(names = { "--rag-max-chars" }, description = "Maximum retrieval context characters to inject", defaultValue = "12000")
    public int ragMaxChars = 12000;

    @Option(names = { "--embedding-model", "--embed-model" }, description = "Embedding model id/path to attach to embedding or RAG-aware providers")
    public String embeddingModel;

    @Option(names = { "--temperature" }, description = "Sampling temperature (default 0.2)")
    public double temperature = 0.2;

    @Option(names = { "--max-tokens" }, description = "Max tokens to generate (default 256)")
    public int maxTokens = 256;

    @Option(names = { "--top-p" }, description = "Top-p sampling (default 0.95)")
    public double topP = 0.95;

    @Option(names = { "--top-k" }, description = "Top-k sampling (default 40)")
    public int topK = 40;

    @Option(names = { "--repeat-penalty" }, description = "Repeat penalty (default 1.1)")
    public double repeatPenalty = 1.1;

    @Option(names = { "--mirostat" }, description = "Mirostat sampling mode (0=off, 1, 2) (default 0)")
    public int mirostat = 0;

    @Option(names = { "--grammar" }, description = "GBNF grammar string for constrained sampling")
    public String grammar;

    @Option(names = { "--stream" }, description = "Stream the response token by token (default true)", negatable = true)
    public boolean stream = true;

    @Option(names = { "--json" }, description = "Enable JSON output mode")
    public boolean jsonMode = false;

    @Option(names = { "--timeout" }, description = "Inference timeout in milliseconds (default 60000)")
    public long inferenceTimeoutMs = 60000;

    @Option(names = { "--no-cache" }, description = "Bypass KV cache")
    public boolean noCache = false;

    @Option(names = { "--concise" }, description = "Use a default concise system prompt")
    public boolean concise = false;

    @Option(names = {
            "--session" }, description = "Enable persistent session (KV cache reuse across calls)", negatable = true)
    public Boolean enableSession;

    @Option(names = {
            "--auto-continue" }, description = "Automatically request continuation for truncated responses", negatable = true)
    public boolean autoContinue = true;

    @Option(names = { "-q", "--quiet" }, description = "Quiet mode: only output messages")
    public boolean quiet = false;

    @Option(names = { "--no-color" }, description = "Disable ANSI color output (also respects NO_COLOR env var)")
    public boolean noColor = false;


    @Option(names = { "-o", "--output" }, description = "Save the whole conversation to a file")
    public java.io.File outputFile;

    @Option(names = { "--sse" }, description = "Output as OpenAI-compatible SSE JSON (for streaming only)")
    public boolean enableJsonSse = false;

    @Option(names = { "--gguf" }, description = "Force use of GGUF format (converts if necessary)")
    public boolean forceGguf = false;

    @Option(names = { "--fallback" }, split = ",", description = "Comma-separated fallback model IDs or short IDs to try only if the primary model is incompatible")
    public List<String> fallbackModelIds = List.of();

    @Option(names = { "--quant" }, description = "Quantization type for GGUF conversion. " +
            "Options: Q4_0 (fastest, smallest), Q4_K_M (balanced), Q5_0, Q5_K_M, Q6_K, Q8_0 (best quality), " +
            "F16, F32 (no quantization). Default: Q4_K_M")
    public String quantization = "Q4_K_M";

    @Option(names = { "--quantize" }, description = "Enable JIT quantization during inference (bnb, turbo, awq, gptq, autoround)")
    public String quantizeStrategy;

    @Option(names = { "--quantize-bits" }, description = "Bit width for JIT quantization (default: 4)", defaultValue = "4")
    public int quantizeBits = 4;

    @Option(names = { "--plugin" }, description = "Explicit plugin/engine to use (e.g. llamacpp, java, bnb)")
    public String pluginId;

    private static final String DEFAULT_CONCISE_SYSTEM_PROMPT = "Answer briefly and directly. Keep responses relevant to the question. "
            + "Prefer 1-4 short sentences unless the user asks for detail.";

    private String modelPathOverride;
    private List<tech.kayys.gollek.spi.tool.ToolDefinition> loadedTools = List.of();
    private Object loadedToolChoice;
    private CliInferenceFeatures.RagContext loadedRagContext = new CliInferenceFeatures.RagContext("", List.of());

    @Override
    public void run() {
        try {
            if (parentCommand != null) {
                parentCommand.bootstrapInheritedEnvironment();
            }

            providerId = normalizeRequestedProvider(providerId);
            boolean providerExplicit = providerId != null && !providerId.isBlank();

            if (noColor) ChatUIRenderer.disableColor();

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
            CliMetalRuntime.initializeIfMetal(detectedPlatform);

            if (!quiet && parentCommand != null && parentCommand.verbose) {
                System.out.println("Platform: " + detectedPlatform.getDisplayName());
                if (detectedPlatform.isCpu()) {
                    System.out.println("⚠️  Running on CPU (GPU acceleration not available)");
                } else if (CliMetalRuntime.isMetal(detectedPlatform) && !CliMetalRuntime.isNativeActive()) {
                    System.out.println("⚠️  Metal selected but native runtime is not active (CPU fallback likely)");
                } else {
                    System.out.println("✓ GPU acceleration enabled");
                }
                System.out.println();
            }

            if (CliMetalRuntime.isMetal(detectedPlatform)
                    && !CliMetalRuntime.isNativeActive()
                    && !CliMetalRuntime.allowCpuFallbackWhenMetalRequested()) {
                System.err.println("Error: Metal platform selected but native Metal runtime is not active.");
                System.err.println("Refusing CPU fallback for this chat so performance behavior stays explicit.");
                System.err.println("Set GOLLEK_ALLOW_CPU_FALLBACK=true to override.");
                System.exit(1);
                return;
            }

            // Check if specific provider is requested but not available
            if (providerExplicit) {
                if (!pluginChecker.hasProvider(providerId)) {
                    System.err.println(pluginChecker.getProviderNotFoundError(providerId));
                    System.exit(1);
                    return;
                }
            }

            configureLogging();

            if (providerId != null && !providerId.isEmpty()) {
                
            }

            // --- Model Resolution Strategy ---
            boolean isLocal = false;
            if (modelFile != null && !modelFile.isBlank()) {
                java.nio.file.Path filePath = java.nio.file.Paths.get(modelFile);
                if (!java.nio.file.Files.exists(filePath)) {
                    System.err.println("Error: Model file not found: " + modelFile);
                    return;
                }
                // Handle --import or --copy
                if (importModel || copyModel) {
                    var res = sdk.importModel(filePath, importModel);
                    filePath = java.nio.file.Paths.get(res.getLocalPath());
                    System.out.println((importModel ? "Imported" : "Copied") + " model to: " + filePath.toAbsolutePath());
                }
                modelPathOverride = filePath.toAbsolutePath().toString();
                isLocal = true;
                // Auto-detect provider from extension
                if ("native".equals(providerId)) {
                    if (modelFile.endsWith(".litertlm") || modelFile.endsWith(".tflite") || modelFile.endsWith(".task")) {
                        providerId = "litert";
                    }
                }
                if (modelId == null) modelId = filePath.getFileName().toString();
            } else if (modelDir != null && !modelDir.isBlank()) {
                java.nio.file.Path dirPath = java.nio.file.Paths.get(modelDir);
                if (!java.nio.file.Files.isDirectory(dirPath)) {
                    System.err.println("Error: Model directory not found: " + modelDir);
                    return;
                }
                // Handle --import or --copy
                if (importModel || copyModel) {
                    var res = sdk.importModel(dirPath, importModel);
                    dirPath = java.nio.file.Paths.get(res.getLocalPath());
                    System.out.println((importModel ? "Imported" : "Copied") + " model to: " + dirPath.toAbsolutePath());
                }
                modelPathOverride = dirPath.toAbsolutePath().toString();
                isLocal = true;
                if (!providerExplicit) {
                    providerId = "safetensor";
                }
                if (modelId == null) modelId = dirPath.getFileName().toString();
            }

            if (!isLocal
                    
                    
                    && modelId != null
                    && !modelId.isBlank()) {
                try {
                    var indexed = LocalModelIndex.find(modelId);
                    if (indexed.isPresent()) {
                        var entry = indexed.get();
                        if (!entry.enabled) {
                            System.err.println("\n\u001B[31mError: Model '" + entry.id + "' is disabled.\u001B[0m");
                            System.err.println("\u001B[33mUse 'gollek model enable " + entry.shortId + "' to enable it for chat.\u001B[0m\n");
                            return;
                        }
                        if (entry.path != null && !entry.path.isBlank()) {
                            java.nio.file.Path indexedPath = java.nio.file.Path.of(entry.path);
                            if (java.nio.file.Files.exists(indexedPath)) {
                                modelPathOverride = indexedPath.toAbsolutePath().toString();
                                modelId = modelPathOverride;
                                isLocal = true;
                                if (!providerExplicit) {
                                    String inferred = inferProviderFromIndex(entry);
                                    if (inferred != null && !inferred.isBlank()) {
                                        providerId = inferred;
                                    }
                                }
                                if (!quiet) {
                                    System.out.println("Resolved local model index entry: " + modelPathOverride);
                                }
                            }
                        }
                    }
                } catch (Exception ignored) {
                    // Fall back to normal SDK resolution if the local index is stale or unreadable.
                }
            }

            if (!isLocal) {
                if (modelId == null || modelId.isBlank()) {
                    modelId = sdk.resolveDefaultModel().orElse(null);
                    if (modelId == null || modelId.isBlank()) {
                        System.err.println("Error: No model specified or found.");
                        printStartupCatalog();
                        return;
                    }
                }

                if (parentCommand != null && parentCommand.verbose) System.out.println("[ChatCommand] forceGguf=" + forceGguf + ", quantization=" + quantization + ", modelId=" + modelId);
                var resolution = sdk.prepareModel(modelId, null, pluginId, forceGguf, quantization, fallbackModelIds, progress -> {
                    if (!quiet)
                        System.out.print(
                                "\rPulling/Converting: " + progress.getPercentComplete() + "% " + progress.getProgressBar(20));
                });

                if (!quiet && resolution.getLocalPath() == null) {
                    System.out.println();
                }

                modelId = resolution.getModelId();
                modelPathOverride = resolution.getLocalPath();
                if (!quiet && resolution.getNotice() != null && !resolution.getNotice().isBlank()) {
                    System.out.println(resolution.getNotice());
                }
                if (parentCommand != null && parentCommand.verbose) System.out.println("[ChatCommand] resolution: localPath=" + modelPathOverride + ", format=" + resolution.getInfo().getFormat());

                // Auto-select provider for downloaded models if not forced
                if (providerId == null) {
                    providerId = resolution.getProviderId();
                }
                if (providerId == null) {
                    providerId = null;
                }
            }

            

            if (!ensureProviderReady()) {
                return;
            }

            // Smart quantization suggestion for large models
            tech.kayys.gollek.cli.util.QuantSuggestionDetector.suggestIfNeeded(
                    modelId, modelPathOverride, quantizeStrategy, quiet);

            setupSession();
            startChatLoop();

        } catch (Throwable e) {
            System.err.println("\n[FATAL] ChatCommand failed with unhandled error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private String normalizeRequestedProvider(String rawProviderId) {
        if (rawProviderId == null) {
            return null;
        }
        String trimmed = rawProviderId.trim();
        if (trimmed.isEmpty()) {
            return trimmed;
        }
        return switch (trimmed.toLowerCase(Locale.ROOT)) {
            case "torch", "torchscript", "pytorch" -> "libtorch";
            default -> trimmed.toLowerCase(Locale.ROOT);
        };
    }


    private void configureLogging() {
        if (parentCommand != null && parentCommand.verbose) {
            System.setProperty("quarkus.log.console.level", "DEBUG");
            System.setProperty("quarkus.log.level", "DEBUG");
            System.setProperty("quarkus.log.category.\"tech.kayys.gollek\".level", "DEBUG");
        }
        // Non-verbose: console stays OFF (set in application.properties),
        // all output goes to ~/.gollek/logs/cli.log
    }

    private void setupSession() throws Exception {
        uiRenderer.setJsonMode(jsonMode);
        boolean persistentSessionEnabled = enableSession == null ? true : enableSession;
        sessionManager.initialize(modelId, providerId, modelPathOverride, persistentSessionEnabled, forceGguf);
        sessionManager.setInferenceParams(autoContinue, maxTokens, temperature);
        loadedTools = CliInferenceFeatures.loadTools(toolFiles, mcpTools);
        loadedToolChoice = CliInferenceFeatures.parseToolChoice(toolChoice);
        loadedRagContext = CliInferenceFeatures.loadRagContext(ragContexts, ragFiles, ragMaxChars);

        PrintWriter writer = null;
        if (outputFile != null) {
            try {
                if (outputFile.getParentFile() != null)
                    outputFile.getParentFile().mkdirs();
                writer = new PrintWriter(new FileWriter(outputFile, true), true);
                writer.println("\n--- Chat Session Started " + java.time.Instant.now() + " ---");
            } catch (Exception e) {
                System.err.println("Failed to open output file: " + e.getMessage());
            }
        }
        sessionManager.setUIHooks(uiRenderer, writer, quiet);

        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            sessionManager.addMessage(tech.kayys.gollek.spi.Message.system(systemPrompt));
        } else if (concise) {
            sessionManager.addMessage(tech.kayys.gollek.spi.Message.system(DEFAULT_CONCISE_SYSTEM_PROMPT));
        } else {
            sessionManager.addMessage(tech.kayys.gollek.spi.Message.system("I'm gollek, and you are using model " + modelId + " to serve you."));
        }
        if (!quiet) {
            uiRenderer.printBanner();
            uiRenderer.printModelInfo(modelId, providerId, null, outputFile != null ? outputFile.getAbsolutePath() : null, true);
            printQuantizationInfo();
        }
    }

    private void printQuantizationInfo() {
        if (quantizeStrategy == null || quantizeStrategy.isBlank()) {
            return;
        }

        String effective = effectiveQuantizationStrategy(quantizeStrategy);
        System.out.println("Quantization: " + quantizeStrategy.toLowerCase()
                + (effective.equalsIgnoreCase(quantizeStrategy) ? "" : " -> " + effective)
                + " (" + quantizeBits + "-bit)");
        System.out.println("KV cache quantization: off");
        System.out.println("--------------------------------------------------");
    }

    private String effectiveQuantizationStrategy(String raw) {
        String normalized = raw == null ? "" : raw.trim().toLowerCase();
        return switch (normalized) {
            case "turbo" -> "bnb";
            case "gptq", "autoround" -> "int4";
            default -> normalized;
        };
    }

    private void startChatLoop() {
        terminalHandler.initialize(quiet, createCompleter(), createCommandHelp());
        String prompt = uiRenderer.getPrompt(quiet);
        String secondary = uiRenderer.getSecondaryPrompt(quiet);

        while (true) {
            String input;
            try {
                input = terminalHandler.readInput(prompt, secondary);
            } catch (org.jline.reader.EndOfFileException e) {
                uiRenderer.printGoodbye(quiet);
                break;
            }

            if (input == null)
                continue;
            if (input.equalsIgnoreCase("exit") || input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("/quit")) {
                uiRenderer.printGoodbye(quiet);
                break;
            }

            if (commandHandler.handleCommand(input, sessionManager, uiRenderer)) {
                continue;
            }

            sessionManager.addMessage(tech.kayys.gollek.spi.Message.user(input));
            CliInferenceFeatures.RagContext turnRagContext = ragContextForTurn(input);

            InferenceRequest.Builder reqBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .messages(historyWithRagContext(turnRagContext))
                    .temperature(temperature)
                    .parameter("top_p", topP)
                    .parameter("top_k", topK)
                    .parameter("repeat_penalty", repeatPenalty)
                    .parameter("json_mode", jsonMode)
                    .parameter("inference_timeout_ms", inferenceTimeoutMs)
                    .maxTokens(maxTokens)
                    .plugin(pluginId)
                    .cacheBypass(noCache);

            if (mirostat > 0)
                reqBuilder.parameter("mirostat", mirostat);
            if (grammar != null && !grammar.isEmpty())
                reqBuilder.parameter("grammar", grammar);

            // JIT quantization parameters
            if (quantizeStrategy != null && !quantizeStrategy.isBlank()) {
                reqBuilder.parameter("quantize_strategy", quantizeStrategy);
                reqBuilder.parameter("quantize_bits", quantizeBits);
            }
            CliInferenceFeatures.applyTools(reqBuilder, loadedTools, loadedToolChoice);
            CliInferenceFeatures.applyRagMetadata(reqBuilder, turnRagContext, embeddingModel);

            sessionManager.executeInference(reqBuilder, stream, enableJsonSse);
        }
    }

    private CliInferenceFeatures.RagContext ragContextForTurn(String input) {
        if ((ragContexts == null || ragContexts.isEmpty()) && (ragFiles == null || ragFiles.isEmpty())) {
            return loadedRagContext;
        }
        try {
            return CliInferenceFeatures.loadRagContext(ragContexts, ragFiles, ragMaxChars, input);
        } catch (Exception e) {
            uiRenderer.printError("RAG context failed: " + e.getMessage(), quiet);
            return loadedRagContext;
        }
    }

    private List<Message> historyWithRagContext(CliInferenceFeatures.RagContext ragContext) {
        List<Message> history = new ArrayList<>(sessionManager.getHistory());
        String ragSystemPrompt = CliInferenceFeatures.ragSystemPrompt(ragContext);
        if (ragSystemPrompt == null || ragSystemPrompt.isBlank()) {
            return history;
        }

        int insertAt = history.size();
        for (int i = history.size() - 1; i >= 0; i--) {
            if (history.get(i).getRole() == Message.Role.USER) {
                insertAt = i;
                break;
            }
        }
        history.add(insertAt, Message.system(ragSystemPrompt));
        return history;
    }

    private org.jline.reader.Completer createCompleter() {
        return (reader, parsedLine, candidates) -> {
            String word = parsedLine.word();
            if (word.startsWith("/"))
                ChatTerminalHandler.COMMANDS.forEach(c -> candidates.add(new org.jline.reader.Candidate(c)));
        };
    }

    private Map<String, CmdDesc> createCommandHelp() {
        Map<String, CmdDesc> help = new HashMap<>();
        for (String c : ChatTerminalHandler.COMMANDS) {
            CmdDesc desc = new CmdDesc();
            desc.setMainDesc(List.of(new AttributedString("Command: " + c)));
            help.put(c, desc);
        }
        return help;
    }

    private boolean ensureProviderReady() { return true; }

    private void printStartupCatalog() {
        if (quiet)
            return;
        try {
            List<ModelInfo> models = sdk.listModels(0, 10);
            if (!models.isEmpty()) {
                uiRenderer.printInfo("Available models: ", quiet);
                for (var m : models)
                    System.out.println("  - " + m.getModelId());
            }
        } catch (Exception ignored) {
        }
    }

    private String inferProviderFromIndex(LocalModelIndex.Entry entry) {
        if (entry == null || entry.format == null) {
            return null;
        }
        String normalized = entry.format.toLowerCase(java.util.Locale.ROOT);
        return switch (normalized) {
            case "gguf" -> "native";
            case "safetensors", "safetensor" -> "safetensor";
            case "litert", "task", "tflite" -> "litert";
            case "onnx" -> "onnx";
            default -> null;
        };
    }
}
