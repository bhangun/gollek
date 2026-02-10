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
@Command(name = "chat", description = "Start an interactive chat session with a model")
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

    @Option(names = { "--json" }, description = "Enable JSON mode", defaultValue = "true")
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
        System.out.println("DEBUG: ChatCommand.run() entered");
        try {
            // Set preferred provider
            if (providerId != null && !providerId.isEmpty()) {
                System.out.println("DEBUG: Setting preferred provider: " + providerId);
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
            if (providerId != null) {
                System.out.printf(BOLD + "Provider: " + RESET + YELLOW + "%s" + RESET + "%n", providerId);
            }
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

                    if (stream) {
                        System.out.print(DIM + "Thinking..." + RESET);
                        java.util.concurrent.atomic.AtomicBoolean firstToken = new java.util.concurrent.atomic.AtomicBoolean(
                                true);
                        java.util.concurrent.atomic.AtomicInteger tokenCount = new java.util.concurrent.atomic.AtomicInteger(
                                0);
                        long startTime = System.currentTimeMillis();
                        StringBuilder fullResponse = new StringBuilder();

                        sdk.streamCompletion(request)
                                .subscribe().with(
                                        chunk -> {
                                            if (firstToken.compareAndSet(true, false)) {
                                                // Clear "Thinking..."
                                                System.out.print(
                                                        "\r" + BOLD + GREEN + "Assistant: " + RESET + "            \r"
                                                                + BOLD + GREEN + "Assistant: " + RESET);
                                            }
                                            String delta = chunk.getDelta();
                                            if (delta != null) {
                                                System.out.print(delta);
                                                fullResponse.append(delta);
                                                tokenCount.incrementAndGet();
                                            }
                                        },
                                        error -> {
                                            System.err.println(
                                                    "\n" + YELLOW + "Stream error: " + RESET + error.getMessage());
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
                                        });
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
