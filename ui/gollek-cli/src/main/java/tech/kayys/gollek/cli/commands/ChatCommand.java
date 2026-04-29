package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import io.quarkus.arc.Unremovable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.cli.chat.*;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;

import org.jline.console.CmdDesc;
import org.jline.utils.AttributedString;

import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.*;
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
    @Inject
    tech.kayys.gollek.cli.util.ModelImporter modelImporter;

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
    public boolean enableSession = true;

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

    @Option(names = { "--quant" }, description = "Quantization type for GGUF conversion. " +
            "Options: Q4_0 (fastest, smallest), Q4_K_M (balanced), Q5_0, Q5_K_M, Q6_K, Q8_0 (best quality), " +
            "F16, F32 (no quantization). Default: Q4_K_M")
    public String quantization = "Q4_K_M";

    @Option(names = { "--quantize" }, description = "Enable JIT quantization during inference (bnb, turbo, awq, gptq, autoround)")
    public String quantizeStrategy;

    @Option(names = { "--quantize-bits" }, description = "Bit width for JIT quantization (default: 4)", defaultValue = "4")
    public int quantizeBits = 4;

    private static final String DEFAULT_CONCISE_SYSTEM_PROMPT = "Answer briefly and directly. Keep responses relevant to the question. "
            + "Prefer 1-4 short sentences unless the user asks for detail.";

    private String modelPathOverride;

    @Override
    public void run() {
        try {
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

            if (!quiet) {
                System.out.println("Platform: " + detectedPlatform.getDisplayName());
                if (detectedPlatform.isCpu()) {
                    System.out.println("⚠️  Running on CPU (GPU acceleration not available)");
                } else {
                    System.out.println("✓ GPU acceleration enabled");
                }
                System.out.println();
            }

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
            configureLogging();

            if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
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
                    filePath = modelImporter.importModel(filePath, importModel, false);
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
                    dirPath = modelImporter.importModel(dirPath, importModel, true);
                    System.out.println((importModel ? "Imported" : "Copied") + " model to: " + dirPath.toAbsolutePath());
                }
                modelPathOverride = dirPath.toAbsolutePath().toString();
                isLocal = true;
                providerId = "safetensor";
                if (modelId == null) modelId = dirPath.getFileName().toString();
            }

            if (!isLocal && !sdk.isMcpProvider(providerId) && !sdk.isCloudProvider(providerId)) {
                if (modelId == null || modelId.isBlank()) {
                    modelId = sdk.resolveDefaultModel().orElse(null);
                    if (modelId == null || modelId.isBlank()) {
                        System.err.println("Error: No model specified or found.");
                        printStartupCatalog();
                        return;
                    }
                }

                if (parentCommand != null && parentCommand.verbose) System.out.println("[ChatCommand] forceGguf=" + forceGguf + ", quantization=" + quantization + ", modelId=" + modelId);
                var resolution = sdk.ensureModelAvailable(modelId, forceGguf, quantization, progress -> {
                    if (!quiet)
                        System.out.print(
                                "\rPulling/Converting: " + progress.getPercentComplete() + "% " + progress.getProgressBar(20));
                });

                if (!quiet && resolution.getLocalPath() == null && !sdk.isCloudProvider(providerId)) {
                    System.out.println();
                }

                modelId = resolution.getModelId();
                modelPathOverride = resolution.getLocalPath();
                if (parentCommand != null && parentCommand.verbose) System.out.println("[ChatCommand] resolution: localPath=" + modelPathOverride + ", format=" + resolution.getInfo().getFormat());

                // Auto-select provider for downloaded models if not forced
                if (providerId == null) {
                    if ("safetensors".equalsIgnoreCase(resolution.getInfo().getFormat())) {
                        providerId = "safetensor";
                    } else if ("litert".equalsIgnoreCase(resolution.getInfo().getFormat())) {
                        providerId = "litert";
                    } else {
                        providerId = sdk.getPreferredProvider().orElse("native");
                    }
                }
            }

            if (providerId != null) {
                sdk.setPreferredProvider(providerId);
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


    private void configureLogging() {
        if (parentCommand != null && parentCommand.verbose) {
            System.setProperty("quarkus.log.console.level", "DEBUG");
            System.setProperty("quarkus.log.level", "DEBUG");
            System.setProperty("quarkus.log.category.\"tech.kayys.gollek\".level", "DEBUG");
        }
        // Non-verbose: console stays OFF (set in application.properties),
        // all output goes to ~/.gollek/logs/cli.log
    }

    private void setupSession() {
        uiRenderer.setJsonMode(jsonMode);
        sessionManager.initialize(modelId, providerId, modelPathOverride, enableSession, forceGguf);
        sessionManager.setInferenceParams(autoContinue, maxTokens, temperature);

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
        }
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

            InferenceRequest.Builder reqBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .temperature(temperature)
                    .parameter("top_p", topP)
                    .parameter("top_k", topK)
                    .parameter("repeat_penalty", repeatPenalty)
                    .parameter("json_mode", jsonMode)
                    .parameter("inference_timeout_ms", inferenceTimeoutMs)
                    .maxTokens(maxTokens)
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

            sessionManager.executeInference(reqBuilder, stream, enableJsonSse);
        }
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

    private boolean ensureProviderReady() {
        if (providerId == null)
            return true;
        try {
            Optional<ProviderInfo> info = sdk.listAvailableProviders().stream()
                    .filter(p -> providerId.equalsIgnoreCase(p.id())).findFirst();
            if (info.isEmpty()) {
                System.err.println("Provider not available: " + providerId);
                return false;
            }
            return info.get().healthStatus() != ProviderHealth.Status.UNHEALTHY;
        } catch (Exception e) {
            return true;
        }
    }

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
}
