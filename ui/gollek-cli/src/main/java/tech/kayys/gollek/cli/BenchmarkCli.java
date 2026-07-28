package tech.kayys.gollek.cli;

import tech.kayys.gollek.factory.GollekSdkFactory;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.UUID;
import java.time.Duration;
import java.time.Instant;
import java.util.Locale;

public class BenchmarkCli {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: java tech.kayys.gollek.cli.BenchmarkCli <model-path> [provider]");
            System.exit(1);
        }

        String modelPath = args[0];
        String provider = args.length > 1 ? args[1] : null;

        System.out.println("== Seamless Model Orchestration Benchmark ==");
        System.out.println("Target Model Path: " + modelPath);
        
        GollekSdk sdk = GollekSdkFactory.createLocalSdk();
        
        if (provider == null || provider.isBlank()) {
            // Auto-detect format based on path extension roughly for test
            String lower = modelPath.toLowerCase(Locale.ROOT);
            if (lower.endsWith(".gguf")) provider = "gguf";
            else if (lower.endsWith(".safetensors") || lower.endsWith(".st")) provider = "safetensor";
            else if (lower.endsWith(".tflite")) provider = "litert";
            else if (lower.endsWith(".onnx")) provider = "onnx";
            else provider = "safetensor"; // Default
            System.out.println("Auto-detected provider: " + provider);
        } else {
            System.out.println("Manually selected provider: " + provider);
        }
        
        String prompt = "Who are you and what can you do?";
        System.out.println("\nPrompt: " + prompt);
        System.out.println("Generating...");
        
        InferenceRequest request = InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(modelPath)
                .parameter("model_path", modelPath)
                .preferredProvider(provider)
                .temperature(0.2)
                .maxTokens(100)
                .build();
                
        // Just adding a user message to request
        request.getMessages().add(tech.kayys.gollek.spi.Message.user(prompt));

        Instant started = Instant.now();
        InferenceResponse response = sdk.createCompletion(request);
        long elapsedMs = Math.max(1L, Duration.between(started, Instant.now()).toMillis());

        System.out.println("\nResult:\n" + response.getContent());
        System.out.println("--------------------------------------------------");
        System.out.println("Metrics:");
        
        int outputTokens = response.getContent().split("\\s+").length; // Rough fallback
        
        double tokensPerSec = (outputTokens / (double) elapsedMs) * 1000.0;
        System.out.println("Total Time : " + elapsedMs + " ms");
        System.out.printf("Throughput : %.2f tokens / sec\n", tokensPerSec);
        System.out.println("Provider   : " + provider);
        System.out.println("--------------------------------------------------");
    }
}
