package tech.kayys.gollek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;
import tech.kayys.gollek.cli.GollekCommand;
import tech.kayys.gollek.cli.chat.*;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import org.jline.console.CmdDesc;
import org.jline.utils.AttributedString;

import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.*;

/**
 * Interactive chat session using GollekSdk.
 */
@Dependent
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

    @Option(names = { "-m", "--model" }, description = "Model ID or path (optional if provider has default)")
    public String modelId;

    @Option(names = { "-p", "--provider" }, description = "Provider ID (e.g. gguf, cerebras, mistral, openai, gemini)")
    public String providerId;

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

    @Option(names = { "-v", "--verbose" }, description = "Verbose mode: show detailed internal logs")
    public boolean verbose = false;


    @Option(names = { "-o", "--output" }, description = "Save the whole conversation to a file")
    public java.io.File outputFile;

    @Option(names = { "--sse" }, description = "Output as OpenAI-compatible SSE JSON (for streaming only)")
    public boolean enableJsonSse = false;

    private static final String DEFAULT_CONCISE_SYSTEM_PROMPT = "Answer briefly and directly. Keep responses relevant to the question. "
            + "Prefer 1-4 short sentences unless the user asks for detail.";

    private String modelPathOverride;

    @Override
    public void run() {
        try {
            if (parentCommand != null) {
                parentCommand.applyRuntimeOverrides();
            }
            configureLogging();

            if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
            }

            if (modelId == null || modelId.isBlank()) {
                modelId = sdk.resolveDefaultModel().orElse(null);
                if (modelId == null || modelId.isBlank()) {
                    System.err.println("Error: No model specified or found.");
                    printStartupCatalog();
                    return;
                }
            }

            if (!isMcpProvider() && !isCloudProvider(providerId)) {
                var resolution = sdk.prepareModel(modelId, progress -> {
                    if (!quiet) System.out.print("\rPulling: " + progress.getPercentComplete() + "% " + progress.getProgressBar(20));
                });
                if (!quiet && resolution.getLocalPath() == null && !isCloudProvider(providerId)) {
                    System.out.println();
                }
                
                modelId = resolution.getModelId();
                modelPathOverride = resolution.getLocalPath();

                if (providerId == null || providerId.isBlank()) {
                    providerId = sdk.autoSelectProvider(modelId).orElse(null);
                    if (providerId != null) {
                        sdk.setPreferredProvider(providerId);
                    }
                }
            }

            if (!ensureProviderReady()) {
                return;
            }

            setupSession();
            startChatLoop();

        } catch (Exception e) {
            uiRenderer.printError("Chat session error: " + e.getMessage(), quiet);
            if (verbose)
                e.printStackTrace();
        }
    }

    private void configureLogging() {
        if (verbose) {
            System.setProperty("quarkus.log.console.level", "DEBUG");
            System.setProperty("quarkus.log.category.\"tech.kayys.gollek\".level", "DEBUG");
        }
    }

    private void setupSession() {
        sessionManager.initialize(modelId, providerId, modelPathOverride, enableSession);
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
        }

        if (!quiet) {
            uiRenderer.printBanner();
            uiRenderer.printModelInfo(modelId, providerId, outputFile != null ? outputFile.getAbsolutePath() : null);
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

            sessionManager.executeInference(reqBuilder, stream, enableJsonSse);
        }
    }

    private org.jline.reader.Completer createCompleter() {
        return (reader, parsedLine, candidates) -> {
            String word = parsedLine.word();
            if (word.startsWith("/")) {
                String[] cmds = { "/help", "/reset", "/quit", "/log", "/list", "/providers", "/provider", "/info",
                        "/extensions" };
                for (String c : cmds)
                    candidates.add(new org.jline.reader.Candidate(c));
            }
        };
    }

    private Map<String, CmdDesc> createCommandHelp() {
        Map<String, CmdDesc> help = new HashMap<>();
        String[] cmds = { "/help", "/reset", "/quit", "/log", "/list", "/providers", "/provider", "/info",
                "/extensions" };
        for (String c : cmds) {
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

    private boolean isMcpProvider() {
        return "mcp".equalsIgnoreCase(providerId);
    }

    private boolean isCloudProvider(String id) {
        if (id == null)
            return false;
        String p = id.toLowerCase();
        return p.equals("openai") || p.equals("mistral") || p.equals("anthropic") || p.equals("gemini")
                || p.equals("cerebras");
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
