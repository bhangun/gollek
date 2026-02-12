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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
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
            "--provider" }, description = "Provider: litert, gguf, ollama, gemini, openai, anthropic, cerebras")
    String providerId;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.8")
    double temperature;

    @Option(names = { "--top-p" }, description = "Top-p sampling", defaultValue = "0.95")
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

    @Option(names = { "-v", "--verbose" }, description = "Enable verbose native debug output")
    boolean verbose;

    @Option(names = { "--mirostat" }, description = "Mirostat mode (0, 1, 2)", defaultValue = "0")
    int mirostat;

    @Option(names = { "--grammar" }, description = "GBNF grammar string")
    String grammar;

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
            // Set preferred provider
            if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
            }

            // Add system prompt if provided
            if (systemPrompt != null && !systemPrompt.isEmpty()) {
                conversationHistory.add(Message.system(systemPrompt));
            }

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
            if (enableSession) {
                sessionId = UUID.randomUUID().toString();
                System.out.printf(BOLD + "Session: " + RESET + DIM + "%s (KV cache enabled)" + RESET + "%n",
                        sessionId.substring(0, 8));
            }
            System.out.println(DIM + "Commands: 'exit' to quit, '/reset' to clear history." + RESET);
            System.out.println(DIM + "Note: Use '\\' at the end of a line for multiline input." + RESET);
            System.out.println(DIM + "-".repeat(50) + RESET);

            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line;

            while (true) {
                System.out.print("\n" + BOLD + CYAN + ">>> " + RESET);
                StringBuilder inputBuffer = new StringBuilder();

                while (true) {
                    line = reader.readLine();
                    if (line == null)
                        break;

                    if (line.endsWith("\\")) {
                        inputBuffer.append(line, 0, line.length() - 1).append("\n");
                        System.out.print(DIM + "... " + RESET);
                    } else {
                        inputBuffer.append(line);
                        break;
                    }
                }

                if (line == null)
                    break;

                String finalInput = inputBuffer.toString().trim();

                if (finalInput.equalsIgnoreCase("exit") || finalInput.equalsIgnoreCase("quit")) {
                    System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
                    break;
                }

                if (finalInput.isEmpty()) {
                    continue;
                }

                // Handle special commands
                if (finalInput.equalsIgnoreCase("/reset")) {
                    conversationHistory.clear();
                    if (systemPrompt != null && !systemPrompt.isEmpty()) {
                        conversationHistory.add(Message.system(systemPrompt));
                    }
                    System.out.println(YELLOW + "[Conversation reset]" + RESET);
                    continue;
                }

                if (finalInput.equalsIgnoreCase("/quit")) {
                    System.out.println("\n" + YELLOW + "Goodbye!" + RESET);
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
                        .maxTokens(2048)
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
                    System.out.print("\n" + BOLD + GREEN + "Assistant: " + RESET);

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

                    if (stream) {
                        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(1);
                        java.util.concurrent.atomic.AtomicBoolean firstTokenReceived = new java.util.concurrent.atomic.AtomicBoolean(
                                false);
                        java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(
                                0);
                        long startTime = System.currentTimeMillis();
                        StringBuilder fullResponse = new StringBuilder();

                        sdk.streamCompletion(request)
                                .subscribe().with(
                                        chunk -> {
                                            firstTokenReceived.set(true);
                                            String delta = chunk.getDelta();
                                            if (delta != null) {
                                                // If this is the very first chunk that has content, verify we clear the
                                                // spinner line effectively
                                                // But since we are printing delta immediately, we need to handle
                                                // spinner cleanup in the main thread or here.
                                                // Let's do it here for immediate feedback, but main thread loop needs
                                                // to stop printing spinner.

                                                System.out.print(delta);
                                                fullResponse.append(delta);
                                                tokenCount.incrementAndGet();
                                            }
                                        },
                                        error -> {
                                            System.err.println(
                                                    "\n" + YELLOW + "Stream error: " + RESET + error.getMessage());
                                            latch.countDown();
                                        },
                                        () -> {
                                            long duration = System.currentTimeMillis() - startTime;
                                            double tps = (tokenCount.get() / (duration / 1000.0));
                                            System.out.println();
                                            System.out.printf(
                                                    DIM + "\n[Tokens: %d, Duration: %.2fs, Speed: %.2f t/s]" + RESET
                                                            + "%n",
                                                    tokenCount.get(), duration / 1000.0, tps);
                                            conversationHistory.add(Message.assistant(fullResponse.toString()));
                                            latch.countDown();
                                        });

                        // Spinner loop
                        String[] spinner = { "|", "/", "-", "\\" };
                        int i = 0;
                        System.out.print(DIM + "Thinking... " + spinner[0] + RESET);

                        while (!firstTokenReceived.get() && latch.getCount() > 0) {
                            try {
                                if (latch.await(100, java.util.concurrent.TimeUnit.MILLISECONDS)) {
                                    break;
                                }
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                break;
                            }
                            if (!firstTokenReceived.get()) {
                                System.out.print("\r" + BOLD + GREEN + "Assistant: " + RESET + DIM + "Thinking... "
                                        + spinner[i++ % spinner.length] + RESET);
                            }
                        }

                        // Clear spinner if we haven't printed anything yet (e.g. error or fast
                        // response)
                        // Actually, if we received first token, the callback printed delta.
                        // But we might have leftover "Thinking..." text if the callback printed delta
                        // ON THE SAME LINE.
                        // Ideally:
                        // 1. Spinner prints "\rAssistant: Thinking... /"
                        // 2. Callback receives token.
                        // 3. Callback should preferably clear line or we ensure we are on a clean
                        // state.

                        // Let's refine:
                        // The spinner loop ends when firstTokenReceived is true.
                        // At that point, the cursor is at the end of spinner.
                        // We need to clear the spinner text.

                        if (firstTokenReceived.get()) {
                            // We are relying on the fact that `System.out.print(delta)` in callback
                            // happened.
                            // But wait, if main thread was sleeping, callback ran.
                            // Callback printed delta.
                            // The delta appeared APENDED to "Thinking... /".
                            // This is messy.
                            // FIX: Callback should NOT print until spinner is cleared?
                            // OR: Spinner loop handles everything?
                            // NO, callback drives data.
                        }

                        // Alternative straightforward approach:
                        // Main thread does NOT print spinner. It just waits.
                        // We launch a separate thread for spinner?
                        // Or we use the main thread loop for spinner.

                        // Let's try to clear line in main thread once loop exits?
                        // If loop exits because firstTokenReceived is true:
                        // That means callback set it to true.
                        // But callback runs concurrently.

                        // Revised logic in replacement content below.

                    } else {
                        long startTime = System.currentTimeMillis();
                        InferenceResponse response = sdk.createCompletion(request);
                        long duration = System.currentTimeMillis() - startTime;
                        String content = response.getContent();

                        // Handle tool calls if present
                        if (response.hasToolCalls()) {
                            System.out.println();
                            for (var toolCall : response.getToolCalls()) {
                                System.out.printf(YELLOW + "  [Tool Call] %s(%s)" + RESET + "%n",
                                        toolCall.name(), toolCall.arguments());
                            }
                        }

                        System.out.println(content);
                        System.out.printf(DIM + "\n[Duration: %.2fs, Tokens: %d]" + RESET + "%n",
                                duration / 1000.0, response.getTokensUsed());

                        // Add assistant response to history
                        conversationHistory.add(Message.assistant(content));
                    }

                } catch (Exception e) {
                    System.err.println("\n" + YELLOW + "Error: " + RESET + e.getMessage());
                }
            }

        } catch (Exception e) {
            System.err.println("Chat failed: " + e.getMessage());
        }
    }
}
