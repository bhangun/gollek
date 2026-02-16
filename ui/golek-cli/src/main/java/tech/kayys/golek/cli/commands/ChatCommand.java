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
import tech.kayys.golek.spi.provider.ProviderInfo;
import tech.kayys.golek.spi.provider.ProviderHealth;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Scanner;
import java.util.UUID;

/**
 * Interactive chat session using GolekSdk.
 * Usage: golek chat --model <model> [--provider ollama|openai|anthropic]
 */
@Dependent
@Unremovable
@Command(name = "chat", aliases = { "chan" }, description = "Start an interactive chat session with a model")
public class ChatCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Option(names = { "-m", "--model" }, description = "Model ID or path", required = true)
    String modelId;

    @Option(names = {
            "--provider" }, description = "Provider: litert, gguf, djl, libtorch(experimental), ollama, gemini, openai, anthropic, cerebras")
    String providerId;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

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

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate per response", defaultValue = "64")
    int maxTokens;

    @Option(names = { "--inference-timeout-ms" }, description = "Hard timeout for one inference call in milliseconds", defaultValue = "30000")
    long inferenceTimeoutMs;

    @Option(names = { "-q", "--quiet" }, description = "Minimal output (no banner/spinner/stats)")
    boolean quiet;

    @Option(names = { "--convert-mode" }, description = "Checkpoint conversion mode: ask, auto, off", defaultValue = "ask")
    String convertMode;

    @Option(names = { "--gguf-outtype" }, description = "GGUF converter outtype (e.g. q4_0, q8_0, f16, f32)")
    String ggufOutType;

    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String CYAN = "\u001B[36m";
    private static final String YELLOW = "\u001B[33m";
    private static final String DIM = "\u001B[2m";
    private static final String BOLD = "\u001B[1m";

    private String sessionId;

    private final List<Message> conversationHistory = new ArrayList<>();

    public ChatCommand() {
    }

    @Override
    public void run() {
        try {
            // Configure logging: show GGUF debug logs only if --verbose/--logs is on
            if (verbose) {
                java.util.logging.Logger ggufLogger = java.util.logging.Logger
                        .getLogger("tech.kayys.golek.inference.gguf");
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

            if (!ensureModelAvailable()) {
                return;
            }
            maybeAutoSelectProvider();
            if (!ensureProviderReady()) {
                return;
            }

            // Add system prompt if provided
            if (systemPrompt != null && !systemPrompt.isEmpty()) {
                conversationHistory.add(Message.system(systemPrompt));
            }

            if (!quiet) {
                System.out.println(BOLD + YELLOW + "  _____       _      _    " + RESET);
                System.out.println(BOLD + YELLOW + " / ____|     | |    | |   " + RESET);
                System.out.println(BOLD + YELLOW + "| |  __  ___ | | ___| | __" + RESET);
                System.out.println(BOLD + YELLOW + "| | |_ |/ _ \\| |/ _ \\ |/ /" + RESET);
                System.out.println(BOLD + YELLOW + "| |__| | (_) | |  __/   < " + RESET);
                System.out.println(BOLD + YELLOW + " \\_____|\\___/|_|\\___|_|\\_\\" + RESET);
                System.out.println();
                System.out.printf(BOLD + "Model: " + RESET + CYAN + "%s" + RESET + "%n", modelId);
                System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s" + RESET + "%n",
                        providerId != null ? providerId : "auto-select");
                if (outputFile != null) {
                    System.out.printf(BOLD + "Output: " + RESET + YELLOW + "%s" + RESET + "%n",
                            outputFile.getAbsolutePath());
                }
                System.out.println(DIM + "Commands: 'exit' to quit, '/reset' to clear history." + RESET);
                System.out.println(DIM + "Note: Use '\\' at the end of a line for multiline input." + RESET);
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

            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line;

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
                if (quiet) {
                    System.out.print("\n>>> ");
                } else {
                    System.out.print("\n" + BOLD + CYAN + ">>> " + RESET);
                }
                StringBuilder inputBuffer = new StringBuilder();

                while (true) {
                    line = reader.readLine();
                    if (line == null)
                        break;

                    if (line.endsWith("\\")) {
                        inputBuffer.append(line, 0, line.length() - 1).append("\n");
                        if (quiet) {
                            System.out.print("... ");
                        } else {
                            System.out.print(DIM + "... " + RESET);
                        }
                    } else {
                        inputBuffer.append(line);
                        break;
                    }
                }

                if (line == null)
                    break;

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
                    if (systemPrompt != null && !systemPrompt.isEmpty()) {
                        conversationHistory.add(Message.system(systemPrompt));
                    }
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
                    System.out.println(DIM + "  /reset  - Clear conversation history" + RESET);
                    System.out.println(DIM + "  /quit   - Exit the chat" + RESET);
                    System.out.println(DIM + "  /log    - Show last 100 lines of log" + RESET);
                    System.out.println(DIM + "  /help   - Show this help message" + RESET);
                    continue;
                }

                if (finalInput.equalsIgnoreCase("/log")) {
                    try {
                        String userHome = System.getProperty("user.home");
                        java.nio.file.Path logPath = java.nio.file.Paths.get(userHome, ".golek", "logs", "cli.log");
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

                if (enableSession && sessionId != null) {
                    reqBuilder.parameter("session_id", sessionId);
                }

                InferenceRequest request = reqBuilder.build();

                try {
                    if (fileWriter == null) {
                        if (quiet) {
                            System.out.print("\nAssistant: ");
                        } else {
                            System.out.print("\n" + BOLD + GREEN + "Assistant: " + RESET);
                        }
                    } else {
                        if (!quiet) {
                            System.out.print(DIM + "Thinking... (Outputting to file)" + RESET);
                        }
                        fileWriter.println("Assistant: ");
                    }

                    if (verbose) {
                        // Dynamically adjust log level if verbose flag is set (Java Util Logging /
                        // JBoss Logging)
                        // This might not work in native if not configured for runtime init, but
                        // harmless to try
                        java.util.logging.Logger.getLogger("").setLevel(java.util.logging.Level.INFO);
                        java.util.logging.Logger.getLogger("tech.kayys.golek").setLevel(java.util.logging.Level.FINE);
                    } else {
                        // Ensure quiet by default
                        java.util.logging.Logger.getLogger("tech.kayys.golek")
                                .setLevel(java.util.logging.Level.WARNING);
                    }

                    if (effectiveStream) {
                        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
                        java.util.concurrent.atomic.AtomicBoolean firstTokenReceived = new java.util.concurrent.atomic.AtomicBoolean(
                                false);
                        java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(
                                0);
                        long startTime = System.currentTimeMillis();
                        StringBuilder fullResponse = new StringBuilder();

                        // Synchronization object for printing
                        Object printLock = new Object();

                        // Capture standard out for potential restoration
                        final java.io.PrintWriter finalFileWriter = fileWriter;

                        sdk.streamCompletion(request)
                                .subscribe().with(
                                        chunk -> {
                                            String delta = chunk.getDelta();
                                            if (delta != null) {
                                                synchronized (printLock) {
                                                    if (!firstTokenReceived.get()) {
                                                        // Clear the spinner line completely
                                                        if (quiet && finalFileWriter == null) {
                                                            firstTokenReceived.set(true);
                                                        } else if (finalFileWriter == null) {
                                                            System.out.print("\r" + BOLD + GREEN + "Assistant: " + RESET
                                                                    + "          \r" + BOLD + GREEN + "Assistant: "
                                                                    + RESET);
                                                        } else {
                                                            if (!quiet) {
                                                                System.out.print("\r" + BOLD + GREEN
                                                                        + "Received response. Writing to file..." + RESET
                                                                        + "   ");
                                                            }
                                                        }
                                                        firstTokenReceived.set(true);
                                                    }

                                                    if (finalFileWriter != null) {
                                                        // Write to file only
                                                        finalFileWriter.print(delta);
                                                        finalFileWriter.flush();
                                                    } else {
                                                        // Write to console
                                                        System.out.print(delta);
                                                    }
                                                }
                                                fullResponse.append(delta);
                                                tokenCount.incrementAndGet();
                                            }
                                        },
                                        error -> {
                                            synchronized (printLock) {
                                                    if (!firstTokenReceived.get() && !quiet) {
                                                        System.out.print("\r" + "                                    \r"); // Clear
                                                                                                                       // line
                                                    }
                                                    if (quiet) {
                                                        System.err.println("\nStream error: " + error.getMessage());
                                                    } else {
                                                        System.err.println(
                                                                "\n" + YELLOW + "Stream error: " + RESET
                                                                        + error.getMessage());
                                                    }
                                                    if (finalFileWriter != null) {
                                                        finalFileWriter.println("\n[Error: " + error.getMessage() + "]");
                                                    }
                                            }
                                            latch.countDown();
                                        },
                                        () -> {
                                            synchronized (printLock) {
                                                if (!firstTokenReceived.get()) {
                                                    // Needed if response was empty but successful
                                                    if (quiet && finalFileWriter == null) {
                                                        // Keep output stable in quiet mode.
                                                    } else if (finalFileWriter == null) {
                                                        System.out.print("\r" + BOLD + GREEN + "Assistant: " + RESET
                                                                + "          \r" + BOLD + GREEN + "Assistant: "
                                                                + RESET);
                                                    }
                                                    firstTokenReceived.set(true);
                                                }
                                                long duration = System.currentTimeMillis() - startTime;
                                                double tps = (tokenCount.get() / (duration / 1000.0));

                                                if (finalFileWriter != null) {
                                                    if (!quiet) {
                                                        System.out.printf(DIM + "\n[Done. Tokens: %d, Speed: %.2f t/s]"
                                                                + RESET + "%n", tokenCount.get(), tps);
                                                    }
                                                    finalFileWriter.println(); // newline in file
                                                    finalFileWriter.printf(
                                                            "\n[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]%n",
                                                            tokenCount.get(), duration / 1000.0, tps);
                                                    finalFileWriter.println("-".repeat(30));
                                                } else {
                                                    System.out.println();
                                                    if (!quiet) {
                                                        System.out.printf(
                                                                DIM + "\n[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]"
                                                                        + RESET
                                                                        + "%n",
                                                                tokenCount.get(), duration / 1000.0, tps);
                                                    }
                                                }
                                            }
                                            conversationHistory.add(Message.assistant(fullResponse.toString()));
                                            latch.countDown();
                                        });

                        // Spinner loop
                        String[] spinner = { "|", "/", "-", "\\" };
                        int i = 0;

                        // Initial print
                        if (!quiet) {
                            synchronized (printLock) {
                                if (!firstTokenReceived.get()) {
                                    System.out.print(DIM + "Thinking... " + spinner[0] + RESET);
                                }
                            }
                        }

                        while (!firstTokenReceived.get() && latch.getCount() > 0) {
                            try {
                                if (latch.await(100, java.util.concurrent.TimeUnit.MILLISECONDS)) {
                                    break;
                                }
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                break;
                            }

                            if (quiet) {
                                continue;
                            }
                            synchronized (printLock) {
                                if (!firstTokenReceived.get()) {
                                    // Move cursor back to update spinner without rewriting "Assistant:"
                                    if (finalFileWriter == null) {
                                        System.out.print(
                                                "\r" + BOLD + GREEN + "Assistant: " + RESET + DIM + "Thinking... "
                                                        + spinner[i++ % spinner.length] + RESET);
                                    } else {
                                        System.out.print(
                                                "\r" + DIM + "Thinking... " + spinner[i++ % spinner.length] + RESET);
                                    }
                                }
                            }
                        }
                        // Wait for stream to complete
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
                        InferenceResponse response = sdk.createCompletion(request);
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
                }
            }

            if (fileWriter != null)
                fileWriter.close();

        } catch (Exception e) {
            if (quiet) {
                System.err.println("Chat failed: " + e.getMessage());
            } else {
                System.err.println("Chat failed: " + e.getMessage());
            }
        }
    }

    private boolean ensureModelAvailable() {
        try {
            var localInfo = sdk.getModelInfo(modelId);
            if (localInfo.isPresent()) {
                maybeUpgradeDjlLayout(localInfo.get());
                return true;
            }

            String fallbackId = modelId + "-GGUF";
            if (sdk.getModelInfo(fallbackId).isPresent()) {
                modelId = fallbackId;
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
                System.err.println("Failed to pull model from Hugging Face: " + reason);
                return false;
            }

            String normalized = modelId.startsWith("hf:") ? modelId.substring(3) : modelId;
            java.util.List<String> candidates = java.util.List.of(
                    modelId,
                    modelId + "-GGUF",
                    normalized,
                    normalized + "-GGUF");
            for (String candidate : candidates) {
                if (sdk.getModelInfo(candidate).isPresent()) {
                    modelId = candidate;
                    sdk.getModelInfo(modelId).ifPresent(this::maybeUpgradeDjlLayout);
                    return true;
                }
            }

            System.err.println("Model download finished but local registration was not found: " + modelId);
            return false;
        } catch (Exception e) {
            System.err.println("Failed to resolve model: " + e.getMessage());
            return false;
        }
    }

    private boolean isHuggingFaceSpec(String id) {
        return id != null && (id.startsWith("hf:") || id.contains("/"));
    }

    private void configureCheckpointConversionPreference() {
        String mode = convertMode == null ? "ask" : convertMode.trim().toLowerCase();
        if (!mode.equals("ask") && !mode.equals("auto") && !mode.equals("off")) {
            mode = "ask";
        }

        if (mode.equals("off")) {
            System.setProperty("golek.gguf.converter.auto", "false");
            return;
        }

        System.setProperty("golek.gguf.converter.auto", "true");
        if (ggufOutType != null && !ggufOutType.isBlank()) {
            System.setProperty("golek.gguf.converter.outtype", ggufOutType.trim().toLowerCase());
            return;
        }

        if (!mode.equals("ask")) {
            if (System.getProperty("golek.gguf.converter.outtype") == null) {
                System.setProperty("golek.gguf.converter.outtype", defaultGgufOutType());
            }
            return;
        }

        if (!isHuggingFaceSpec(modelId)) {
            return;
        }
        if (quiet || System.console() == null) {
            return;
        }
        if (System.getProperty("golek.gguf.converter.prompted") != null) {
            return;
        }
        System.setProperty("golek.gguf.converter.prompted", "true");

        Scanner scanner = new Scanner(System.in);
        System.out.print("If this model has no GGUF artifact, convert checkpoints to GGUF? [Y/n]: ");
        String answer = scanner.nextLine();
        if (answer != null && answer.trim().equalsIgnoreCase("n")) {
            System.setProperty("golek.gguf.converter.auto", "false");
            return;
        }

        String current = System.getProperty("golek.gguf.converter.outtype", defaultGgufOutType());
        System.out.print("Select GGUF outtype [q4_0/q8_0/f16/f32] (default " + current + "): ");
        String outType = scanner.nextLine();
        if (outType == null || outType.isBlank()) {
            outType = current;
        }
        outType = outType.trim().toLowerCase();
        if (!isSupportedGgufOutType(outType)) {
            outType = defaultGgufOutType();
        }
        System.setProperty("golek.gguf.converter.outtype", outType);
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
                    .map(p -> p.healthStatus() != tech.kayys.golek.spi.provider.ProviderHealth.Status.UNHEALTHY)
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
            if (isCheckpointOnlyFormat(format) && tryRefreshCompatibleModel()) {
                modelInfoOpt = sdk.getModelInfo(modelId);
                if (modelInfoOpt.isPresent()) {
                    format = modelInfoOpt.get().getFormat();
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
                var info = sdk.getModelInfo(candidate);
                if (info.isPresent() && !isCheckpointOnlyFormat(info.get().getFormat())) {
                    modelId = candidate;
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
}
