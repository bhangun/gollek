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
import tech.kayys.gollek.sdk.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderInfo;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import org.jline.reader.Candidate;
import org.jline.reader.Completer;
import org.jline.reader.LineReader;
import org.jline.reader.LineReaderBuilder;
import org.jline.console.CmdDesc;
import org.jline.utils.AttributedString;
import org.jline.widget.AutosuggestionWidgets;
import org.jline.widget.TailTipWidgets;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

/**
 * Interactive chat session using GollekSdk.
 * Usage: gollek chat --model <model> [--provider ollama|openai|anthropic]
 */
@Dependent
@Unremovable
@Command(name = "chat", aliases = { "chan" }, description = "Start an interactive chat session with a model")
public class ChatCommand implements Runnable {

    @ParentCommand
    GollekCommand parentCommand;

    @Inject
    GollekSdk sdk;
    @Inject
    Instance<HuggingFaceClient> hfClientInstance;
    @Inject
    ProviderRegistry providerRegistry;
    @Inject
    ListCommand listCommand;
    @Inject
    ProvidersCommand providersCommand;
    @Inject
    InfoCommand infoCommand;
    @Inject
    ExtensionsCommand extensionsCommand;

    @Option(names = { "-m", "--model" }, description = "Model ID or path (optional if provider has default)")
    public String modelId;

    @Option(names = {
            "--provider" }, description = "Provider: litert, gguf, djl, safetensor, libtorch(experimental), ollama, gemini, openai, anthropic, cerebras")
    public String providerId;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

    @Option(names = {
            "--concise" }, description = "Prefer concise answers for lower latency", defaultValue = "true", negatable = true)
    public boolean concise;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.2")
    public double temperature;

    @Option(names = { "--top-p" }, description = "Top-p sampling", defaultValue = "0.9")
    double topP;

    @Option(names = { "--top-k" }, description = "Top-k sampling", defaultValue = "40")
    int topK;

    @Option(names = { "--repeat-penalty" }, description = "Repeat penalty", defaultValue = "1.1")
    double repeatPenalty;

    @Option(names = { "--json" }, description = "Enable JSON mode", defaultValue = "false")
    boolean jsonMode;

    @Option(names = { "--no-cache" }, description = "Bypass response cache")
    boolean noCache;

    @Option(names = { "--stream" }, description = "Stream output", defaultValue = "true")
    boolean stream;

    @Option(names = { "--session" }, description = "Enable stateful session (KV cache reuse)")
    boolean enableSession;

    @Option(names = { "-o", "--output" }, description = "Output file for assistant responses")
    java.io.File outputFile;

    @Option(names = { "-v", "--verbose", "--logs" }, description = "Enable verbose native debug output")
    boolean verbose;

    @Option(names = { "--mirostat" }, description = "Mirostat mode (0, 1, 2)", defaultValue = "0")
    int mirostat;

    @Option(names = { "--grammar" }, description = "GBNF grammar string")
    String grammar;

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate per response", defaultValue = "48")
    int maxTokens;

    @Option(names = {
            "--inference-timeout-ms" }, description = "Hard timeout for one inference call in milliseconds", defaultValue = "180000")
    long inferenceTimeoutMs;

    @Option(names = { "-q", "--quiet" }, description = "Minimal output (no banner/spinner/stats)")
    boolean quiet;

    @Option(names = {
            "--convert-mode" }, description = "Checkpoint conversion mode: ask, auto, off", defaultValue = "ask")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. q4_0, q8_0, f16, f32)")
    String ggufOutType;

    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String CYAN = "\u001B[36m";
    private static final String YELLOW = "\u001B[33m";
    private static final String DIM = "\u001B[2m";
    private static final String BOLD = "\u001B[1m";
    private static final String DEFAULT_CONCISE_SYSTEM_PROMPT = "Answer briefly and directly. Keep responses relevant to the question. "
            + "Prefer 1-4 short sentences unless the user asks for detail.";

    private String sessionId;
    private String modelPathOverride;

    private final List<Message> conversationHistory = new ArrayList<>();

    public ChatCommand() {
    }

    @Override
    public void run() {
        try {
            boolean bareChatRequested = (modelId == null || modelId.isBlank())
                    && (providerId == null || providerId.isBlank());
            if (parentCommand != null) {
                parentCommand.applyRuntimeOverrides();
            }
            // Configure logging: show GGUF debug logs only if --verbose/--logs is on
            if (verbose) {
                System.setProperty("quarkus.log.console.level", "DEBUG");
                System.setProperty("quarkus.log.category.\"tech.kayys.gollek\".level", "DEBUG");
                System.setProperty("quarkus.log.category.\"tech.kayys.gollek.inference.libtorch\".level", "DEBUG");

                java.util.logging.Logger ggufLogger = java.util.logging.Logger
                        .getLogger("tech.kayys.gollek.inference.gguf");
                ggufLogger.setLevel(java.util.logging.Level.FINE);
                // Ensure a console handler is present for FINE-level output
                java.util.logging.ConsoleHandler ch = new java.util.logging.ConsoleHandler();
                ch.setLevel(java.util.logging.Level.FINE);
                ggufLogger.addHandler(ch);
            }

            // Set preferred provider
            if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
            }

            // Resolve default model from provider if omitted
            if (modelId == null || modelId.isBlank()) {
                modelId = resolveDefaultModelForProvider().orElse(null);
                if (modelId == null || modelId.isBlank()) {
                    if ("safetensor".equalsIgnoreCase(providerId)) {
                        System.err.println(
                                "Error: No local safetensor model found. Pull one first, or pass --model <id|path>.");
                    } else {
                        System.err.println(
                                "Error: Missing required option: '--model=<modelId>' (or specify a provider with a default model)");
                    }
                    printStartupCatalog();
                    return;
                }
            }

            if (!isMcpProvider() && !isCloudProvider(providerId)) {
                if (!ensureModelAvailable()) {
                    return;
                }
                maybeAutoSelectProvider();
            }
            if (!ensureSafetensorMetadataReady()) {
                return;
            }
            if (!ensureProviderReady()) {
                return;
            }
            if (!prepareSafeRuntimeForSafetensor()) {
                return;
            }
            printCompatibilityHintBeforeInference();

            addInitialSystemPromptIfNeeded();

            if (!quiet) {
                System.out.println(BOLD + YELLOW + "  _____       _  _      _    " + RESET);
                System.out.println(BOLD + YELLOW + " / ____|     | || |    | |   " + RESET);
                System.out.println(BOLD + YELLOW + "| |  __  ___ | || | ___| | __" + RESET);
                System.out.println(BOLD + YELLOW + "| | |_ |/ _ \\| || |/ _ \\ |/ /" + RESET);
                System.out.println(BOLD + YELLOW + "| |__| | (_) | || |  __/   < " + RESET);
                System.out.println(BOLD + YELLOW + " \\_____|\\___/|_||_|\\___|_|\\_\\" + RESET);
                System.out.println();
                System.out.printf(BOLD + "Model: " + RESET + CYAN + "%s" + RESET + "%n", displayModelName(modelId));
                System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s" + RESET + "%n",
                        providerId != null ? providerId : "auto-select");
                if (outputFile != null) {
                    System.out.printf(BOLD + "Output: " + RESET + YELLOW + "%s" + RESET + "%n",
                            outputFile.getAbsolutePath());
                }
                System.out.println(DIM + "Commands: 'exit' to quit, '/reset' to clear history." + RESET);
                System.out.println(DIM + "Note: Use '\\' at the end of a line for multiline input." + RESET);
                if (bareChatRequested) {
                    printStartupCatalog();
                }
                System.out.println(DIM + "-".repeat(50) + RESET);
            }
            if (enableSession) {
                sessionId = UUID.randomUUID().toString();
                if (!quiet) {
                    System.out.printf(BOLD + "Session: " + RESET + DIM + "%s (KV cache enabled)" + RESET + "%n",
                            sessionId.substring(0, 8));
                }
            }

            boolean effectiveStream = stream;
            if (effectiveStream && ("djl".equalsIgnoreCase(providerId) || "safetensor".equalsIgnoreCase(providerId))) {
                effectiveStream = false;
                if (!quiet) {
                    System.out.println(DIM + "Provider '" + providerId
                            + "' does not support streaming; using non-streaming mode."
                            + RESET);
                }
            }

            // Initialize JLine Terminal and LineReader
            Terminal terminal = null;
            try {
                try {
                    terminal = TerminalBuilder.builder()
                            .system(true)
                            .dumb(true)
                            .build();
                } catch (Exception e) {
                    // Fallback to dumb terminal
                    terminal = TerminalBuilder.builder().dumb(true).build();
                }

                // Custom Completer for slash commands
                Completer slashCompleter = (lineReader, parsedLine, candidates) -> {
                    String word = parsedLine.word();
                    if (word.startsWith("/")) {
                        candidates.add(
                                new Candidate("/help", "/help", null, "Show available commands", null, null, true));
                        candidates.add(new Candidate("/reset", "/reset", null, "Clear conversation history", null, null,
                                true));
                        candidates
                                .add(new Candidate("/quit", "/quit", null, "Exit the chat session", null, null, true));
                        candidates.add(new Candidate("/log", "/log", null, "Show last 100 lines of CLI log", null, null,
                                true));
                        candidates
                                .add(new Candidate("/list", "/list", null, "List available models", null, null, true));
                        candidates.add(new Candidate("/providers", "/providers", null, "List available LLM providers",
                                null, null, true));
                        candidates.add(new Candidate("/provider", "/provider", null,
                                "Switch provider (e.g., /provider gemini)", null, null, true));
                        candidates.add(new Candidate("/info", "/info", null, "Display system info and adapters", null,
                                null, true));
                        candidates.add(new Candidate("/extensions", "/extensions", null,
                                "Show packaged extension modules", null, null, true));
                    }
                };

                LineReader reader = LineReaderBuilder.builder()
                        .terminal(terminal)
                        .completer(slashCompleter)
                        .variable(LineReader.HISTORY_FILE,
                                Path.of(System.getProperty("user.home"), ".gollek", "chat_history"))
                        .variable(LineReader.LIST_MAX, 50)
                        .option(LineReader.Option.AUTO_MENU, true)
                        .option(LineReader.Option.AUTO_LIST, true)
                        .option(LineReader.Option.COMPLETE_IN_WORD, true)
                        .build();

                // Enable autosuggestions (ghost text)
                AutosuggestionWidgets autosuggestionWidgets = new AutosuggestionWidgets(reader);
                autosuggestionWidgets.enable();

                // Enable TailTip (descriptions at the bottom)
                java.util.Map<String, CmdDesc> commandHelp = new java.util.HashMap<>();
                commandHelp.put("/help", createCmdDesc("Show available commands"));
                commandHelp.put("/reset", createCmdDesc("Clear conversation history"));
                commandHelp.put("/quit", createCmdDesc("Exit the chat session"));
                commandHelp.put("/log", createCmdDesc("Show last 100 lines of CLI log"));
                commandHelp.put("/list", createCmdDesc("List available models"));
                commandHelp.put("/providers", createCmdDesc("List available LLM providers"));
                commandHelp.put("/provider", createCmdDesc("Switch provider (e.g., /provider gemini)"));
                commandHelp.put("/info", createCmdDesc("Display system info and adapters"));
                commandHelp.put("/extensions", createCmdDesc("Show packaged extension modules"));

                TailTipWidgets tailTipWidgets = new TailTipWidgets(reader, commandHelp, 0,
                        TailTipWidgets.TipType.COMPLETER);
                tailTipWidgets.enable();

                // Set prompt
                String promptText = quiet ? "\n>>> " : "\n" + BOLD + CYAN + ">>> " + RESET;
                String secondaryPrompt = quiet ? "... " : DIM + "... " + RESET;

                // Open output file if specified (append mode)
                java.io.PrintWriter fileWriter = null;
                if (outputFile != null) {
                    try {
                        // Create parent dirs if needed
                        if (outputFile.getParentFile() != null) {
                            outputFile.getParentFile().mkdirs();
                        }
                        fileWriter = new java.io.PrintWriter(new java.io.FileWriter(outputFile, true), true);
                        fileWriter.println("\n--- Chat Session Started " + java.time.Instant.now() + " ---");
                    } catch (Exception e) {
                        System.err.println(YELLOW + "Failed to open output file: " + e.getMessage() + RESET);
                    }
                }

                while (true) {
                    StringBuilder inputBuffer = new StringBuilder();
                    String currentPrompt = promptText;
                    boolean interrupted = false;

                    while (true) {
                        String lineInput;
                        try {
                            lineInput = reader.readLine(currentPrompt);
                        } catch (org.jline.reader.UserInterruptException e) {
                            interrupted = true;
                            break;
                        } catch (org.jline.reader.EndOfFileException e) {
                            if (!quiet) {
                                System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
                            }
                            return;
                        }

                        if (lineInput.endsWith("\\")) {
                            inputBuffer.append(lineInput, 0, lineInput.length() - 1).append("\n");
                            currentPrompt = secondaryPrompt;
                        } else {
                            inputBuffer.append(lineInput);
                            break;
                        }
                    }

                    if (interrupted) {
                        continue;
                    }

                    String finalInput = inputBuffer.toString().trim();

                    if (finalInput.equalsIgnoreCase("exit") || finalInput.equalsIgnoreCase("quit")) {
                        if (!quiet) {
                            System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
                        }
                        break;
                    }

                    if (finalInput.isEmpty()) {
                        continue;
                    }

                    // Echo user input to file
                    if (fileWriter != null) {
                        fileWriter.println("\n>>> User: " + finalInput);
                    }

                    // Handle special commands
                    if (finalInput.equalsIgnoreCase("/reset")) {
                        conversationHistory.clear();
                        addInitialSystemPromptIfNeeded();
                        System.out.println(YELLOW + "[Conversation reset]" + RESET);
                        if (fileWriter != null)
                            fileWriter.println("[Conversation reset]");
                        continue;
                    }

                    if (finalInput.equalsIgnoreCase("/quit")) {
                        if (!quiet) {
                            System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
                        }
                        break;
                    }

                    if (finalInput.equalsIgnoreCase("/help")) {
                        System.out.println(DIM + "Available commands:" + RESET);
                        System.out.println(DIM + "  /reset      - Clear conversation history" + RESET);
                        System.out.println(DIM + "  /quit       - Exit the chat session" + RESET);
                        System.out.println(DIM + "  /log        - Show last 100 lines of log" + RESET);
                        System.out.println(DIM + "  /list       - List available models" + RESET);
                        System.out.println(DIM + "  /providers  - List available LLM providers" + RESET);
                        System.out.println(DIM + "  /provider <id> - Switch to a different provider" + RESET);
                        System.out.println(DIM + "  /info       - Display system info" + RESET);
                        System.out.println(DIM + "  /extensions - Show packaged extension modules" + RESET);
                        System.out.println(DIM + "  /help       - Show this help message" + RESET);
                        continue;
                    }

                    if (finalInput.equalsIgnoreCase("/list")) {
                        listCommand.run();
                        continue;
                    }

                    if (finalInput.equalsIgnoreCase("/providers")) {
                        providersCommand.run();
                        continue;
                    }

                    if (finalInput.toLowerCase().startsWith("/provider ")) {
                        String newProviderId = finalInput.substring(10).trim();
                        if (newProviderId.isEmpty()) {
                            System.out.println(YELLOW + "Usage: /provider <provider-id>" + RESET);
                        } else if (ensureProviderHealthy(newProviderId)) {
                            this.providerId = newProviderId;
                            sdk.setPreferredProvider(newProviderId);
                            System.out.println(GREEN + "Switched to provider: " + RESET + CYAN + providerId + RESET);
                        }
                        continue;
                    }

                    if (finalInput.equalsIgnoreCase("/info")) {
                        infoCommand.run();
                        continue;
                    }

                    if (finalInput.equalsIgnoreCase("/extensions")) {
                        extensionsCommand.run();
                        continue;
                    }

                    if (finalInput.equalsIgnoreCase("/log")) {
                        try {
                            String userHome = System.getProperty("user.home");
                            java.nio.file.Path logPath = java.nio.file.Paths.get(userHome, ".gollek", "logs",
                                    "cli.log");
                            if (java.nio.file.Files.exists(logPath)) {
                                List<String> lines = java.nio.file.Files.readAllLines(logPath);
                                int start = Math.max(0, lines.size() - 100);
                                System.out.println(DIM + "--- Last 100 log lines ---" + RESET);
                                for (int i = start; i < lines.size(); i++) {
                                    System.out.println(DIM + lines.get(i) + RESET);
                                }
                                System.out.println(DIM + "--------------------------" + RESET);
                            } else {
                                System.out.println(YELLOW + "Log file not found at: " + logPath + RESET);
                            }
                        } catch (Exception e) {
                            System.err.println(YELLOW + "Failed to read logs: " + e.getMessage() + RESET);
                        }
                        continue;
                    }

                    // Add user message to history
                    conversationHistory.add(Message.user(finalInput));

                    // Build request with conversation history
                    InferenceRequest.Builder reqBuilder = InferenceRequest.builder()
                            .requestId(UUID.randomUUID().toString())
                            .model(modelId)
                            .messages(new ArrayList<>(conversationHistory))
                            .temperature(temperature)
                            .parameter("top_p", topP)
                            .parameter("top_k", topK)
                            .parameter("repeat_penalty", repeatPenalty)
                            .parameter("json_mode", jsonMode)
                            .parameter("inference_timeout_ms", inferenceTimeoutMs)
                            .maxTokens(maxTokens)
                            .preferredProvider(providerId)
                            .cacheBypass(noCache);

                    if (mirostat > 0) {
                        reqBuilder.parameter("mirostat", mirostat);
                    }
                    if (grammar != null && !grammar.isEmpty()) {
                        reqBuilder.parameter("grammar", grammar);
                    }
                    if (modelPathOverride != null && !modelPathOverride.isBlank()) {
                        reqBuilder.parameter("model_path", modelPathOverride);
                    }

                    if (enableSession && sessionId != null) {
                        reqBuilder.parameter("session_id", sessionId);
                    }

                    InferenceRequest request = reqBuilder.build();

                    try {
                        if (fileWriter != null) {
                            if (!quiet) {
                                System.out.print(DIM + "Thinking... (Outputting to file)" + RESET);
                                System.out.flush();
                            }
                            fileWriter.println("Assistant: ");
                        } else {
                            System.out.print("\n");
                            System.out.flush();
                        }

                        if (verbose) {
                            // Dynamically adjust log level if verbose flag is set (Java Util Logging /
                            // JBoss Logging)
                            // This might not work in native if not configured for runtime init, but
                            // harmless to try
                            java.util.logging.Logger.getLogger("").setLevel(java.util.logging.Level.INFO);
                            java.util.logging.Logger.getLogger("tech.kayys.gollek")
                                    .setLevel(java.util.logging.Level.FINE);
                        } else {
                            // Ensure quiet by default
                            java.util.logging.Logger.getLogger("tech.kayys.gollek")
                                    .setLevel(java.util.logging.Level.WARNING);
                        }

                        if (effectiveStream) {
                            java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
                            java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(
                                    0);
                            long startTime = System.currentTimeMillis();
                            StringBuilder fullResponse = new StringBuilder();
                            final java.io.PrintWriter finalFileWriter = fileWriter;

                            if (!quiet && finalFileWriter == null) {
                                System.out.print(BOLD + GREEN + "Assistant: " + RESET);
                                System.out.flush();
                            }

                            sdk.streamCompletion(request)
                                    .subscribe().with(
                                            chunk -> {
                                                String delta = chunk.getDelta();
                                                if (delta == null) {
                                                    return;
                                                }
                                                if (finalFileWriter != null) {
                                                    finalFileWriter.print(delta);
                                                    finalFileWriter.flush();
                                                } else {
                                                    System.out.print(delta);
                                                    System.out.flush();
                                                }
                                                fullResponse.append(delta);
                                                tokenCount.incrementAndGet();
                                            },
                                            error -> {
                                                if (quiet) {
                                                    System.err.println("\nStream error: " + error.getMessage());
                                                } else {
                                                    System.err.println("\n" + YELLOW + "Stream error: " + RESET
                                                            + error.getMessage());
                                                }
                                                printProviderHintFromError(error);
                                                if (finalFileWriter != null) {
                                                    finalFileWriter.println("\n[Error: " + error.getMessage() + "]");
                                                }
                                                latch.countDown();
                                            },
                                            () -> {
                                                long duration = System.currentTimeMillis() - startTime;
                                                double tps = (tokenCount.get() / (duration / 1000.0));

                                                if (finalFileWriter != null) {
                                                    if (!quiet) {
                                                        System.out.printf(DIM + "\n[Done. Tokens: %d, Speed: %.2f t/s]"
                                                                + RESET + "%n", tokenCount.get(), tps);
                                                    }
                                                    finalFileWriter.println();
                                                    finalFileWriter.printf(
                                                            "\n[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]%n",
                                                            tokenCount.get(), duration / 1000.0, tps);
                                                    finalFileWriter.println("-".repeat(30));
                                                } else {
                                                    System.out.println();
                                                    if (!quiet) {
                                                        System.out.printf(
                                                                DIM + "\n[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]"
                                                                        + RESET + "%n",
                                                                tokenCount.get(), duration / 1000.0, tps);
                                                    }
                                                }
                                                conversationHistory.add(Message.assistant(fullResponse.toString()));
                                                latch.countDown();
                                            });

                            try {
                                latch.await();
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                if (!quiet) {
                                    System.out.println("\nInterrupted.");
                                }
                            }
                        } else {
                            // Non-streaming logic
                            if (verbose && !quiet) {
                                System.out.println(DIM + "Non-streaming mode..." + RESET);
                            }

                            long startTime = System.currentTimeMillis();
                            InferenceResponse response = shouldDirectProviderBypass(effectiveStream)
                                    ? inferDirectWithProvider(providerId, request)
                                    : sdk.createCompletion(request);
                            long duration = System.currentTimeMillis() - startTime;
                            String content = response.getContent();

                            // Handle tool calls if present
                            if (response.hasToolCalls() && !quiet) {
                                System.out.println();
                                for (var toolCall : response.getToolCalls()) {
                                    System.out.printf(YELLOW + "  [Tool Call] %s(%s)" + RESET + "%n",
                                            toolCall.name(), toolCall.arguments());
                                }
                            }

                            if (fileWriter != null) {
                                fileWriter.println("\nAssistant: " + content);
                                fileWriter.printf("\n[Duration: %.2fs, Tokens: %d]%n", duration / 1000.0,
                                        response.getTokensUsed());
                                fileWriter.println("-".repeat(30));
                                if (!quiet) {
                                    System.out.print(DIM + "Response written to file." + RESET);
                                }
                            } else {
                                System.out.println(content);
                                if (!quiet) {
                                    System.out.printf(DIM + "\n[Duration: %.2fs, Tokens: %d]" + RESET + "%n",
                                            duration / 1000.0, response.getTokensUsed());
                                }
                            }

                            // Add assistant response to history
                            conversationHistory.add(Message.assistant(content));
                        }

                    } catch (Exception e) {
                        if (quiet) {
                            System.err.println("\nError: " + e.getMessage());
                        } else {
                            System.err.println("\n" + YELLOW + "Error: " + RESET + e.getMessage());
                        }
                        printProviderHintFromError(e);
                    }
                }

                if (fileWriter != null)
                    fileWriter.close();
            } finally {
                if (terminal != null) {
                    try {
                        terminal.close();
                    } catch (Exception ignored) {
                    }
                }
            }

        } catch (Exception e) {
            if (quiet) {
                System.err.println("Chat failed: " + e.getMessage());
            } else {
                System.err.println("Chat failed: " + e.getMessage());
            }
            printProviderHintFromError(e);
        }
    }

    private boolean ensureModelAvailable() {
        try {
            var resolved = LocalModelResolver.resolve(sdk, modelId);
            if (resolved.isPresent()) {
                var found = resolved.get();
                modelId = found.modelId();
                modelPathOverride = found.localPath() != null ? found.localPath().toString() : null;
                maybeUpgradeDjlLayout(found.info());
                return true;
            }

            if (!isHuggingFaceSpec(modelId)) {
                System.err.println("Error: Model not found locally: " + modelId);
                return false;
            }

            configureCheckpointConversionPreference();

            boolean pulled = false;
            Exception lastPullError = null;
            for (String spec : buildPullSpecs(modelId)) {
                try {
                    if (!quiet) {
                        System.out.println("Pulling model from Hugging Face: " + spec);
                    }
                    sdk.pullModel(spec, progress -> {
                        if (quiet) {
                            return;
                        }
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
                    if (!quiet) {
                        System.out.println("\nDownload complete.");
                    }
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
                if (HuggingFaceCheckpointStore.shouldStoreOnPullFailure(reason)) {
                    var stored = HuggingFaceCheckpointStore.storeCheckpointArtifacts(
                            hfClientInstance,
                            modelId,
                            progress -> {
                                if (!quiet) {
                                    System.out.print("\r" + progress.getStatus());
                                }
                            });
                    if (stored.isPresent() && stored.get().hasWeights()) {
                        if (!quiet) {
                            System.out.println();
                        }
                        System.out.println("Checkpoint artifacts saved to: " + stored.get().rootDir().toAbsolutePath());
                        System.err.println(
                                "Model was downloaded in origin checkpoint format (.safetensors/.bin) and is not runnable yet in local Java runtime.");
                        System.err.println("Use conversion (GGUF/TorchScript) when you want to run this model.");
                        return false;
                    }
                }
                System.err.println("Failed to pull model from Hugging Face: " + reason);
                return false;
            }

            resolved = LocalModelResolver.resolve(sdk, modelId);
            if (resolved.isPresent()) {
                var found = resolved.get();
                modelId = found.modelId();
                modelPathOverride = found.localPath() != null ? found.localPath().toString() : null;
                maybeUpgradeDjlLayout(found.info());
                return true;
            }

            System.err.println("Model download finished but local registration was not found: " + modelId);
            return false;
        } catch (Exception e) {
            System.err.println("Failed to resolve model: " + e.getMessage());
            return false;
        }
    }

    private void addInitialSystemPromptIfNeeded() {
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            conversationHistory.add(Message.system(systemPrompt));
            return;
        }
        if (concise) {
            conversationHistory.add(Message.system(DEFAULT_CONCISE_SYSTEM_PROMPT));
        }
    }

    private boolean isMcpProvider() {
        return providerId != null && "mcp".equalsIgnoreCase(providerId.trim());
    }

    private boolean isCloudProvider(String providerId) {
        if (providerId == null || providerId.isBlank()) {
            return false;
        }
        String p = providerId.trim().toLowerCase();
        return p.equals("cerebras") || p.equals("ollama") || p.equals("mistral") || p.equals("gemini")
                || p.equals("openai") || p.equals("anthropic");
    }

    private boolean isHuggingFaceSpec(String id) {
        return id != null && (id.startsWith("hf:") || id.contains("/"));
    }

    private void configureCheckpointConversionPreference() {
        String mode = convertMode == null ? "ask" : convertMode.trim().toLowerCase();
        if (!mode.equals("ask") && !mode.equals("auto") && !mode.equals("off")) {
            mode = "ask";
        }

        // If a specific provider is requested and it's not GGUF, skip GGUF conversion
        // prompt
        if (providerId != null && !providerId.isEmpty() && !"gguf".equalsIgnoreCase(providerId)) {
            return;
        }

        if (mode.equals("off")) {
            System.setProperty("gollek.gguf.converter.auto", "false");
            return;
        }

        System.setProperty("gollek.gguf.converter.auto", "true");
        if (ggufOutType != null && !ggufOutType.isBlank()) {
            System.setProperty("gollek.gguf.converter.outtype", ggufOutType.trim().toLowerCase());
            return;
        }

        if (!mode.equals("ask")) {
            if (System.getProperty("gollek.gguf.converter.outtype") == null) {
                System.setProperty("gollek.gguf.converter.outtype", defaultGgufOutType());
            }
            return;
        }

        if (!isHuggingFaceSpec(modelId)) {
            return;
        }
        if (quiet || System.console() == null) {
            return;
        }
        if (System.getProperty("gollek.gguf.converter.prompted") != null) {
            return;
        }
        System.setProperty("gollek.gguf.converter.prompted", "true");

        String answer = promptConsoleLine("If this model has no GGUF artifact, convert checkpoints to GGUF? [Y/n]: ");
        if (answer != null && answer.trim().equalsIgnoreCase("n")) {
            System.setProperty("gollek.gguf.converter.auto", "false");
            return;
        }

        String current = System.getProperty("gollek.gguf.converter.outtype", defaultGgufOutType());
        String outType = promptConsoleLine("Select GGUF outtype [q4_0/q8_0/f16/f32] (default " + current + "): ");
        if (outType == null || outType.isBlank()) {
            outType = current;
        }
        outType = outType.trim().toLowerCase();
        if (!isSupportedGgufOutType(outType)) {
            outType = defaultGgufOutType();
        }
        System.setProperty("gollek.gguf.converter.outtype", outType);
    }

    private String promptConsoleLine(String prompt) {
        java.io.Console console = System.console();
        if (console == null) {
            return "";
        }
        String value = console.readLine("%s", prompt);
        return value != null ? value : "";
    }

    private String defaultGgufOutType() {
        if (isLikelyLargeModelId(modelId)) {
            return "q4_0";
        }
        return "q8_0";
    }

    private boolean isSupportedGgufOutType(String outType) {
        return outType != null && switch (outType) {
            case "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", "f16", "f32" -> true;
            default -> false;
        };
    }

    private boolean isLikelyLargeModelId(String id) {
        if (id == null) {
            return false;
        }
        String normalized = id.toUpperCase();
        return normalized.contains("4B")
                || normalized.contains("7B")
                || normalized.contains("8B")
                || normalized.contains("13B")
                || normalized.contains("14B")
                || normalized.contains("32B")
                || normalized.contains("70B");
    }

    private java.util.List<String> buildPullSpecs(String id) {
        if (id == null || id.isBlank()) {
            return java.util.List.of();
        }
        if (id.startsWith("hf:")) {
            String bare = id.substring(3);
            return java.util.List.of(id, bare);
        }
        if (id.contains("/")) {
            return java.util.List.of(id, "hf:" + id);
        }
        return java.util.List.of(id);
    }

    private void maybeAutoSelectProvider() {
        if (providerId != null && !providerId.isBlank()) {
            return;
        }
        try {
            var modelInfoOpt = LocalModelResolver.resolve(sdk, modelId).map(LocalModelResolver.ResolvedModel::info);
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
            case "PYTORCH" -> "djl";
            case "SAFETENSORS" -> "safetensor";
            case "ONNX" -> "onnx";
            default -> null;
        };
    }

    private void maybeUpgradeDjlLayout(ModelInfo info) {
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
        if (!isHuggingFaceSpec(modelId)) {
            return;
        }
        try {
            if (!quiet) {
                System.out.println("Upgrading local model layout for DJL runtime: " + modelId);
            }
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
                    .map(p -> p.healthStatus() != tech.kayys.gollek.spi.provider.ProviderHealth.Status.UNHEALTHY)
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

            var modelInfoOpt = LocalModelResolver.resolve(sdk, modelId).map(LocalModelResolver.ResolvedModel::info);
            if (modelInfoOpt.isEmpty()) {
                return true;
            }
            var modelInfo = modelInfoOpt.get();
            String format = modelInfo.getFormat();
            if (isCheckpointOnlyFormat(format)) {
                String checkpointProvider = providerForFormat(format);
                if (checkpointProvider != null && ensureProviderHealthy(checkpointProvider)) {
                    providerId = checkpointProvider;
                    sdk.setPreferredProvider(checkpointProvider);
                    return true;
                }
                if (tryRefreshCompatibleModel()) {
                    modelInfoOpt = LocalModelResolver.resolve(sdk, modelId).map(LocalModelResolver.ResolvedModel::info);
                    if (modelInfoOpt.isPresent()) {
                        format = modelInfoOpt.get().getFormat();
                    }
                }
            }
            if (isCheckpointOnlyFormat(format)) {
                String checkpointProvider = providerForFormat(format);
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

    private boolean tryRefreshCompatibleModel() {
        if (!isHuggingFaceSpec(modelId)) {
            return false;
        }
        if (!quiet) {
            System.out.println("Checkpoint-only model detected. Trying GGUF/TorchScript fallback...");
        }
        for (String spec : buildPullSpecs(modelId)) {
            try {
                sdk.pullModel(spec, null);
            } catch (Exception ignored) {
                // try next pull spec
            }
        }

        String normalized = modelId.startsWith("hf:") ? modelId.substring(3) : modelId;
        for (String candidate : java.util.List.of(modelId, normalized, modelId + "-GGUF", normalized + "-GGUF")) {
            try {
                var resolved = LocalModelResolver.resolve(sdk, candidate);
                if (resolved.isPresent() && !isCheckpointOnlyFormat(resolved.get().info().getFormat())) {
                    modelId = resolved.get().modelId();
                    modelPathOverride = resolved.get().localPath() != null ? resolved.get().localPath().toString()
                            : null;
                    if (!quiet) {
                        System.out.println("Using compatible model: " + modelId);
                    }
                    return true;
                }
            } catch (Exception ignored) {
                // continue
            }
        }
        return false;
    }

    private boolean ensureProviderHealthy(String provider) {
        if (isCloudProvider(provider)) {
            // Cloud providers might report UNHEALTHY if API keys are missing,
            // but we want to let the request go through to show the actual API error
            return true;
        }

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

    private boolean prepareSafeRuntimeForSafetensor() {
        if (!"safetensor".equalsIgnoreCase(providerId)) {
            return true;
        }

        String repo = inferCurrentHfRepo();
        if (repo == null || repo.isBlank()) {
            System.err.println(
                    "Error: Safetensor runtime is unstable in this environment and direct in-process loading is not safe.");
            System.err.println(
                    "Use a GGUF/TorchScript model instead, or pass an explicit runnable model with --model.");
            return false;
        }

        if (!quiet) {
            System.out.println("Preparing runnable runtime for checkpoint model: " + displayModelName(modelId));
        }

        try {
            // For safetensor fallback, force converter-on so pull can produce runnable
            // GGUF.
            if (System.getProperty("gollek.gguf.converter.auto") == null) {
                System.setProperty("gollek.gguf.converter.auto", "true");
            }
            if (System.getProperty("gollek.gguf.converter.outtype") == null) {
                System.setProperty("gollek.gguf.converter.outtype", defaultGgufOutType());
            }

            java.util.LinkedHashSet<String> specs = new java.util.LinkedHashSet<>();
            specs.add(repo);
            specs.add("hf:" + repo);

            for (String spec : specs) {
                try {
                    sdk.pullModel(spec, null);
                } catch (Exception ignored) {
                    // continue trying alternatives
                }
            }

            for (String candidate : runtimeModelCandidates(repo)) {
                if (tryActivateRuntimeCandidate(candidate)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
            // handled by fallback message below
        }

        if (hasOnlyEmptyGgufCacheDirs(repo)) {
            System.err.println("Note: GGUF cache folders exist but contain no runnable model files yet.");
        }
        System.err.println(
                "Error: Unable to prepare a runnable model from safetensor checkpoint in this environment.");
        System.err.println("Try: pull a GGUF variant and run with --provider gguf.");
        return false;
    }

    private List<String> runtimeModelCandidates(String repo) {
        java.util.LinkedHashSet<String> out = new java.util.LinkedHashSet<>();
        out.add(repo + "-GGUF");
        out.add("hf:" + repo + "-GGUF");
        out.add(repo);
        out.add("hf:" + repo);
        return List.copyOf(out);
    }

    private boolean tryActivateRuntimeCandidate(String candidate) {
        if (candidate == null || candidate.isBlank()) {
            return false;
        }

        // Local filesystem candidate: never route through HF resolver.
        if (looksLikeLocalPath(candidate)) {
            Path p = Path.of(candidate);
            if (!Files.exists(p)) {
                return false;
            }
            String lower = p.getFileName() != null ? p.getFileName().toString().toLowerCase() : "";
            if (!(lower.endsWith(".gguf") || lower.endsWith(".pt") || lower.endsWith(".pth"))) {
                return false;
            }
            modelId = p.toString();
            modelPathOverride = p.toString();
            providerId = lower.endsWith(".gguf") ? "gguf" : "djl";
            try {
                sdk.setPreferredProvider(providerId);
            } catch (tech.kayys.gollek.sdk.exception.SdkException e) {
                // Ignore or log
            }
            if (!quiet) {
                System.out.println("Using compatible runtime model: " + displayModelName(modelId)
                        + " (provider: " + providerId + ")");
            }
            return true;
        }

        var resolved = LocalModelResolver.resolve(sdk, candidate);
        if (resolved.isEmpty()) {
            return false;
        }
        var found = resolved.get();
        String format = found.info() != null ? found.info().getFormat() : null;
        if (isCheckpointOnlyFormat(format)) {
            return false;
        }

        modelId = found.modelId();
        modelPathOverride = found.localPath() != null ? found.localPath().toString() : null;
        String upgradedProvider = providerForFormat(format);
        if (upgradedProvider != null && !upgradedProvider.isBlank()) {
            providerId = upgradedProvider;
            try {
                sdk.setPreferredProvider(upgradedProvider);
            } catch (tech.kayys.gollek.sdk.exception.SdkException e) {
                // Ignore or log
            }
        }
        if (!quiet) {
            System.out.println("Using compatible runtime model: " + displayModelName(modelId)
                    + " (provider: " + providerId + ")");
        }
        return true;
    }

    private boolean looksLikeLocalPath(String value) {
        if (value.startsWith("/") || value.startsWith("~")) {
            return true;
        }
        return value.length() > 2 && Character.isLetter(value.charAt(0)) && value.charAt(1) == ':';
    }

    private boolean hasOnlyEmptyGgufCacheDirs(String repo) {
        Path base = Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors");
        String org = repo.contains("/") ? repo.substring(0, repo.indexOf('/')) : "";
        String name = repo.contains("/") ? repo.substring(repo.indexOf('/') + 1) : repo;
        if (org.isBlank() || name.isBlank()) {
            return false;
        }
        Path ggufDir = base.resolve(org).resolve(name + "-GGUF");
        Path ggufDir2 = base.resolve(org).resolve(name + "-GGUF-GGUF");
        return (Files.isDirectory(ggufDir) && isDirEmpty(ggufDir))
                || (Files.isDirectory(ggufDir2) && isDirEmpty(ggufDir2));
    }

    private boolean isDirEmpty(Path dir) {
        try (var files = Files.list(dir)) {
            return files.findAny().isEmpty();
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean ensureSafetensorMetadataReady() {
        if (!"safetensor".equalsIgnoreCase(providerId)) {
            return true;
        }

        Path modelPath = resolveEffectiveModelPath();
        if (modelPath == null || !Files.exists(modelPath)) {
            return true;
        }

        Path parent = modelPath.getParent();
        if (parent == null || Files.exists(parent.resolve("config.json"))) {
            return true;
        }

        if (!quiet) {
            System.out.println("Safetensor metadata is incomplete (missing config.json). Attempting auto-refresh...");
        }

        java.util.LinkedHashSet<String> refreshSpecs = new java.util.LinkedHashSet<>();
        if (isHuggingFaceSpec(modelId)) {
            refreshSpecs.addAll(buildPullSpecs(modelId));
        }
        inferHfRepoFromSafetensorPath(modelPath).ifPresent(repo -> refreshSpecs.addAll(buildPullSpecs(repo)));

        Exception lastError = null;
        for (String spec : refreshSpecs) {
            try {
                if (!quiet) {
                    System.out.println("Refreshing checkpoint metadata from Hugging Face: " + spec);
                }
                sdk.pullModel(spec, progress -> {
                    if (quiet) {
                        return;
                    }
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
                if (!quiet) {
                    System.out.println();
                }
                break;
            } catch (Exception e) {
                lastError = e;
            }
        }

        var resolved = LocalModelResolver.resolve(sdk, modelId);
        if (resolved.isPresent()) {
            var found = resolved.get();
            modelId = found.modelId();
            modelPathOverride = found.localPath() != null ? found.localPath().toString() : modelPathOverride;
            modelPath = found.localPath() != null ? found.localPath() : modelPath;
        }

        Path updatedParent = modelPath != null ? modelPath.getParent() : null;
        if (updatedParent != null && Files.exists(updatedParent.resolve("config.json"))) {
            return true;
        }

        if (!quiet && lastError != null) {
            System.err.println("Metadata auto-refresh failed: " + describeError(lastError));
        }
        System.err.println("Error: Incomplete safetensor checkpoint for " + displayModelName(modelId)
                + " (missing config.json). Re-pull model to download metadata sidecars.");
        return false;
    }

    private Path resolveEffectiveModelPath() {
        if (modelPathOverride != null && !modelPathOverride.isBlank()) {
            try {
                return Path.of(modelPathOverride);
            } catch (Exception ignored) {
                // fallback below
            }
        }
        try {
            Path candidate = Path.of(modelId);
            if (Files.exists(candidate)) {
                return candidate;
            }
        } catch (Exception ignored) {
            // fallback below
        }
        return LocalModelResolver.resolve(sdk, modelId)
                .map(LocalModelResolver.ResolvedModel::localPath)
                .orElse(null);
    }

    private Optional<String> inferHfRepoFromSafetensorPath(Path modelPath) {
        if (modelPath == null) {
            return Optional.empty();
        }
        Path base = Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors");
        try {
            Path parent = modelPath.getParent();
            if (parent == null || !parent.startsWith(base)) {
                return Optional.empty();
            }
            Path rel = base.relativize(parent);
            String repo = rel.toString().replace('\\', '/');
            if (repo.isBlank()) {
                return Optional.empty();
            }
            return Optional.of(repo);
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private String inferCurrentHfRepo() {
        if (isHuggingFaceSpec(modelId)) {
            return modelId.startsWith("hf:") ? modelId.substring(3) : modelId;
        }
        Path effective = resolveEffectiveModelPath();
        return inferHfRepoFromSafetensorPath(effective).orElse(null);
    }

    private void printGenericProviderSetupHint(String providerId) {
        if ("djl".equalsIgnoreCase(providerId)) {
            System.err.println(
                    "DJL runtime is not loaded. Ensure gollek-ext-format-djl is on classpath and DJL native runtime can initialize.");
        } else if ("libtorch".equalsIgnoreCase(providerId)) {
            System.err.println(
                    "LibTorch runtime is not loaded. Set LIBTORCH_PATH and include gollek-ext-format-libtorch.");
        } else if ("gguf".equalsIgnoreCase(providerId)) {
            System.err.println(
                    "GGUF runtime is not loaded. Set GOLEK_LLAMA_LIB_DIR/GOLEK_LLAMA_LIB_PATH and include gollek-ext-format-gguf.");
        }
    }

    private void printProviderHintFromError(Throwable throwable) {
        String detail = describeError(throwable).toLowerCase();
        if (detail.contains("provider not available: libtorch")) {
            printGenericProviderSetupHint("libtorch");
        } else if (detail.contains("provider not available: djl")) {
            printGenericProviderSetupHint("djl");
        } else if (detail.contains("provider not available: gguf")) {
            printGenericProviderSetupHint("gguf");
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

    private Optional<ProviderInfo> findProviderInfo(String id) {
        try {
            return sdk.listAvailableProviders().stream()
                    .filter(p -> id.equalsIgnoreCase(p.id()))
                    .findFirst();
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    private Optional<String> resolveDefaultModelForProvider() {
        if (providerId == null || providerId.isBlank()) {
            return findPreferredLocalModelForBareChat();
        }

        if ("safetensor".equalsIgnoreCase(providerId)) {
            return findLocalSafetensorModelPath();
        }

        return providerRegistry.getProvider(providerId)
                .map(LLMProvider::metadata)
                .map(meta -> meta.getDefaultModel())
                .filter(v -> v != null && !v.isBlank());
    }

    private Optional<String> findPreferredLocalModelForBareChat() {
        List<ModelInfo> local = listLocalModelsSafe(64);
        return local.stream()
                .max(Comparator.comparingInt(this::autoPickScore)
                        .thenComparing(m -> m.getUpdatedAt() != null ? m.getUpdatedAt() : java.time.Instant.EPOCH))
                .map(ModelInfo::getModelId)
                .filter(v -> v != null && !v.isBlank());
    }

    private int autoPickScore(ModelInfo model) {
        if (model == null || model.getFormat() == null) {
            return 0;
        }
        String format = model.getFormat().trim().toUpperCase(java.util.Locale.ROOT);
        return switch (format) {
            case "GGUF" -> 400;
            case "TORCHSCRIPT" -> 300;
            case "SAFETENSORS" -> 250;
            case "PYTORCH" -> 200;
            case "BIN" -> 100;
            default -> 50;
        };
    }

    private List<ModelInfo> listLocalModelsSafe(int limit) {
        try {
            List<ModelInfo> models = sdk.listModels(0, Math.max(limit, 32));
            LinkedHashSet<String> seen = new LinkedHashSet<>();
            List<ModelInfo> out = new ArrayList<>();
            for (ModelInfo model : models) {
                if (model == null || model.getModelId() == null) {
                    continue;
                }
                if (seen.add(model.getModelId())) {
                    out.add(model);
                }
            }
            return out;
        } catch (Exception ignored) {
            return List.of();
        }
    }

    private void printStartupCatalog() {
        if (quiet) {
            return;
        }

        List<String> localModels = listLocalModelsSafe(8).stream()
                .map(ModelInfo::getModelId)
                .filter(id -> id != null && !id.isBlank())
                .map(this::displayModelName)
                .distinct()
                .limit(8)
                .toList();

        List<String> providers;
        try {
            providers = sdk.listAvailableProviders().stream()
                    .map(ProviderInfo::id)
                    .filter(id -> id != null && !id.isBlank())
                    .distinct()
                    .sorted(String::compareToIgnoreCase)
                    .toList();
        } catch (Exception ignored) {
            providers = List.of();
        }

        if (!localModels.isEmpty()) {
            System.out.println(DIM + "Local models: " + RESET + String.join(", ", localModels));
        } else {
            System.out.println(DIM + "Local models: " + RESET + "(none)");
        }
        if (!providers.isEmpty()) {
            System.out.println(DIM + "Providers: " + RESET + String.join(", ", providers));
        }
    }

    private Optional<String> findLocalSafetensorModelPath() {
        Path base = Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors");
        if (!Files.isDirectory(base)) {
            return Optional.empty();
        }

        try (var files = Files.walk(base, 6)) {
            return files
                    .filter(Files::isRegularFile)
                    .filter(this::isSafetensorFile)
                    .max(Comparator.comparing(this::lastModifiedSafe))
                    .map(path -> path.toAbsolutePath().toString());
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private boolean isSafetensorFile(Path path) {
        String name = path.getFileName().toString().toLowerCase(java.util.Locale.ROOT);
        return name.endsWith(".safetensors") || name.endsWith(".safetensor");
    }

    private java.time.Instant lastModifiedSafe(Path path) {
        try {
            return Files.getLastModifiedTime(path).toInstant();
        } catch (Exception ignored) {
            return java.time.Instant.EPOCH;
        }
    }

    private String displayModelName(String modelRef) {
        if (modelRef == null || modelRef.isBlank()) {
            return "unknown";
        }
        try {
            Path path = Path.of(modelRef);
            Path fileName = path.getFileName();
            String file = fileName != null ? fileName.toString() : modelRef;
            String lower = file.toLowerCase(java.util.Locale.ROOT);
            if (lower.endsWith(".safetensors") || lower.endsWith(".safetensor")) {
                Path parent = path.getParent();
                if (parent != null && parent.getFileName() != null) {
                    return parent.getFileName().toString();
                }
            }
            return file;
        } catch (Exception ignored) {
            return modelRef;
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
                System.err.println(YELLOW
                        + "Hint: provider 'safetensor' is text-checkpoint oriented. This model looks multimodal/VLM and may fail."
                        + RESET);
                System.err.println(
                        YELLOW + "Hint: try a text-only checkpoint, or use a provider/runtime with multimodal support."
                                + RESET);
            } else if (model.endsWith(".gguf") || model.endsWith(".pt") || model.endsWith(".pth")) {
                System.err.println(
                        YELLOW + "Hint: provider 'safetensor' expects .safetensor/.safetensors weights, but got: "
                                + modelName + RESET);
            }
            return;
        }

        if ("gguf".equals(provider) && !(model.endsWith(".gguf") || model.contains("/models/gguf/"))) {
            System.err.println(YELLOW + "Hint: provider 'gguf' works best with GGUF models (.gguf)." + RESET);
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

    private boolean shouldDirectProviderBypass(boolean effectiveStream) {
        if (effectiveStream || providerId == null || providerId.isBlank()) {
            return false;
        }
        return "djl".equalsIgnoreCase(providerId)
                || "safetensor".equalsIgnoreCase(providerId)
                || "libtorch".equalsIgnoreCase(providerId);
    }

    private InferenceResponse inferDirectWithProvider(String id, InferenceRequest request) {
        String providerModel = (modelPathOverride != null && !modelPathOverride.isBlank())
                ? modelPathOverride
                : request.getModel();
        ProviderRequest providerRequest = ProviderRequest.builder()
                .model(providerModel)
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .streaming(false)
                .timeout(Duration.ofMillis(Math.max(inferenceTimeoutMs, 30_000L)))
                .metadata("request_id", request.getRequestId())
                .metadata("tenantId", "community")
                .build();
        try {
            return inferDirect(id, providerRequest);
        } catch (RuntimeException primary) {
            if (shouldFallbackToLibtorch(id, providerModel)) {
                if (!quiet) {
                    System.out.println(DIM + "DJL checkpoint load failed; trying libtorch (experimental)..." + RESET);
                }
                try {
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
        return providerOpt.get()
                .infer(providerRequest)
                .await()
                .atMost(Duration.ofMillis(Math.max(inferenceTimeoutMs + 60_000L, 90_000L)));
    }

    private boolean shouldFallbackToLibtorch(String providerId, String providerModel) {
        if (!"djl".equalsIgnoreCase(providerId) || providerModel == null || providerModel.isBlank()) {
            return false;
        }
        String normalized = providerModel.toLowerCase();
        if (normalized.endsWith(".safetensors")
                || normalized.endsWith(".safetensor")
                || normalized.endsWith(".bin")
                || normalized.endsWith(".pth")
                || normalized.contains("/.gollek/models/djl/")
                || normalized.contains("\\.gollek\\models\\djl\\")) {
            return true;
        }
        try {
            Path modelPath = Path.of(providerModel);
            if (Files.isDirectory(modelPath)) {
                try (var files = Files.walk(modelPath, 3)) {
                    return files
                            .filter(Files::isRegularFile)
                            .map(path -> path.getFileName().toString().toLowerCase())
                            .anyMatch(name -> name.endsWith(".safetensors")
                                    || name.endsWith(".safetensor")
                                    || name.endsWith(".bin")
                                    || name.endsWith(".pth"));
                }
            }
        } catch (Exception ignored) {
            // best effort only
        }
        return false;
    }

    private CmdDesc createCmdDesc(String description) {
        CmdDesc desc = new CmdDesc();
        desc.setMainDesc(List.of(new AttributedString(description)));
        return desc;
    }
}
