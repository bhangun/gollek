package tech.kayys.golek.cli.commands;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.Message;

import java.util.UUID;

/**
 * Run inference command using GolekSdk.
 * Usage: golek run --model <model> --prompt <prompt> [--provider
 * ollama|gguf|gemini] [--stream]
 */
@Command(name = "run", description = "Run inference using a specified model")
@Dependent
public class RunCommand implements Runnable {

    @Inject
    GolekSdk sdk;

    @Option(names = { "-m", "--model" }, description = "Model ID or path", required = true)
    String modelId;

    @Option(names = { "-p", "--prompt" }, description = "Input prompt", required = true)
    String prompt;

    @Option(names = { "-t", "--tenant" }, description = "Tenant ID", defaultValue = "default")
    String tenantId;

    @Option(names = { "--provider" }, description = "Provider: gguf, ollama, gemini, openai, anthropic, cerebras")
    String providerId;

    @Option(names = { "-s", "--stream" }, description = "Stream output")
    boolean stream;

    @Option(names = { "--temperature" }, description = "Sampling temperature", defaultValue = "0.7")
    double temperature;

    @Option(names = { "--max-tokens" }, description = "Maximum tokens to generate", defaultValue = "2048")
    int maxTokens;

    @Option(names = { "--system" }, description = "System prompt")
    String systemPrompt;

    @Override
    public void run() {
        long startTime = System.currentTimeMillis();

        try {
            // Set preferred provider if specified
            if (providerId != null && !providerId.isEmpty()) {
                sdk.setPreferredProvider(providerId);
            }

            // Build request
            InferenceRequest.Builder requestBuilder = InferenceRequest.builder()
                    .requestId(UUID.randomUUID().toString())
                    .tenantId(tenantId)
                    .model(modelId)
                    .temperature(temperature)
                    .maxTokens(maxTokens)
                    .streaming(stream);

            if (systemPrompt != null && !systemPrompt.isEmpty()) {
                requestBuilder.message(Message.system(systemPrompt));
            }
            requestBuilder.message(Message.user(prompt));

            if (providerId != null && !providerId.isEmpty()) {
                requestBuilder.preferredProvider(providerId);
            }

            InferenceRequest request = requestBuilder.build();

            System.out.printf("Running inference [Model: %s, Tenant: %s]%n", modelId, tenantId);
            if (providerId != null) {
                System.out.printf("Provider: %s%n", providerId);
            }

            if (stream) {
                // Streaming mode
                sdk.streamCompletion(request)
                        .subscribe().with(
                                chunk -> System.out.print(chunk.delta()),
                                error -> System.err.println("\nError: " + error.getMessage()),
                                () -> {
                                    long duration = System.currentTimeMillis() - startTime;
                                    System.out.printf("%n[Duration: %d ms]%n", duration);
                                });
            } else {
                // Sync mode
                InferenceResponse response = sdk.createCompletion(request);
                printResponse(response, startTime);
            }

        } catch (Exception e) {
            System.err.println("Inference failed: " + e.getMessage());
        }
    }

    private void printResponse(InferenceResponse response, long startTime) {
        System.out.println();
        System.out.println(response.getContent());
        System.out.printf("%n[Duration: %d ms, Tokens: %d]%n",
                response.getDurationMs(),
                response.getTokensUsed());
    }
}
