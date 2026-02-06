package tech.kayys.golek.cli.commands;

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
@Command(name = "chat", description = "Start an interactive chat session")
@Dependent
public class ChatCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Option(names = { "-m", "--model" }, description = "Model ID or path", required = true)
    String modelId;

    @Option(names = { "-t", "--tenant" }, description = "Tenant ID", defaultValue = "default")
    String tenantId;

    @Option(names = { "--provider" }, description = "Provider: gguf, ollama, gemini, openai, anthropic, cerebras")
    String providerId;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.7")
    double temperature;

    private final List<Message> conversationHistory = new ArrayList<>();

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

            System.out.println("Golek Chat Session");
            System.out.printf("Model: %s%n", modelId);
            if (providerId != null) {
                System.out.printf("Provider: %s%n", providerId);
            }
            System.out.println("Type 'exit' or 'quit' to end the session.");
            System.out.println("-".repeat(50));

            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line;

            while (true) {
                System.out.print("\nYou: ");
                line = reader.readLine();

                if (line == null || line.trim().equalsIgnoreCase("exit")
                        || line.trim().equalsIgnoreCase("quit")) {
                    System.out.println("\nGoodbye!");
                    break;
                }

                if (line.trim().isEmpty()) {
                    continue;
                }

                // Add user message to history
                conversationHistory.add(Message.user(line.trim()));

                // Build request with conversation history
                InferenceRequest request = InferenceRequest.builder()
                        .requestId(UUID.randomUUID().toString())
                        .tenantId(tenantId)
                        .model(modelId)
                        .messages(new ArrayList<>(conversationHistory))
                        .temperature(temperature)
                        .maxTokens(2048)
                        .preferredProvider(providerId)
                        .build();

                try {
                    System.out.print("\nAssistant: ");

                    InferenceResponse response = sdk.createCompletion(request);
                    String content = response.getContent();
                    System.out.println(content);

                    // Add assistant response to history
                    conversationHistory.add(Message.assistant(content));

                } catch (Exception e) {
                    System.err.println("\nError: " + e.getMessage());
                }
            }

        } catch (Exception e) {
            System.err.println("Chat failed: " + e.getMessage());
        }
    }
}
