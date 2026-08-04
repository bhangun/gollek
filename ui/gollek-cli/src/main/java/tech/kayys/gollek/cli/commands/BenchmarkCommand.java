package tech.kayys.gollek.cli.commands;

import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Option;
import tech.kayys.gollek.cli.GollekCommand;
import picocli.CommandLine.ParentCommand;

import java.util.UUID;
import java.time.Duration;
import java.time.Instant;
import java.util.Locale;

@Command(name = "benchmark", description = "Benchmark a local model (Safetensor, GGUF, LiteRT)")
public class BenchmarkCommand implements Runnable {

    @ParentCommand
    GollekCommand parentCommand;

    @Inject
    jakarta.enterprise.inject.Instance<tech.kayys.gollek.spi.inference.LocalInferenceEngine> engines;

    @Parameters(index = "0", description = "Model path or ID")
    String modelPath;

    @Option(names = {"--provider", "-p"}, description = "Provider (gguf, safetensor, litert, onnx)", defaultValue = "")
    String provider;

    @Override
    public void run() {
        System.out.println("== Seamless Model Orchestration Benchmark ==");
        System.out.println("Target Model Path: " + modelPath);

        if (provider == null || provider.isBlank()) {
            String lower = modelPath.toLowerCase(Locale.ROOT);
            if (lower.endsWith(".gguf")) provider = "gguf";
            else if (lower.endsWith(".safetensors") || lower.endsWith(".st")) provider = "safetensor";
            else if (lower.endsWith(".tflite") || lower.endsWith(".task")) provider = "litert";
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
                .message(tech.kayys.gollek.spi.Message.user(prompt))
                .build();

        Instant started = Instant.now();
        String outputContent = "";
        try {
            if ("gguf".equalsIgnoreCase(provider)) {
                System.out.println("DEBUG: Executing JavaNativeGgufBackend...");
                tech.kayys.gollek.cli.commands.GgufFastRun.FastArgs parsed = tech.kayys.gollek.cli.commands.GgufFastRun.FastArgs.parse(new String[]{"run", "--model", modelPath});
                java.util.Optional<java.nio.file.Path> resolved = tech.kayys.gollek.cli.commands.GgufFastRun.resolveGgufModel(parsed);
                if (resolved.isEmpty()) {
                    throw new RuntimeException("Could not resolve GGUF model path for: " + modelPath);
                }
                tech.kayys.gollek.plugin.runner.gguf.JavaNativeGgufBackend backend = 
                        new tech.kayys.gollek.plugin.runner.gguf.JavaNativeGgufBackend(resolved.get());
                tech.kayys.gollek.plugin.runner.RunnerRequest rr = tech.kayys.gollek.plugin.runner.RunnerRequest.builder()
                        .type(tech.kayys.gollek.plugin.runner.RequestType.INFER)
                        .inferenceRequest(request)
                        .build();
                tech.kayys.gollek.plugin.runner.RunnerResult<?> result = backend.execute(rr);
                if (result.isSuccess()) {
                    outputContent = String.valueOf(result.getData());
                } else {
                    throw new RuntimeException(result.getErrorMessage().orElse("GGUF execution failed"));
                }
            } else {
                System.out.println("DEBUG: Executing Safetensor via LocalInferenceEngine...");
                tech.kayys.gollek.spi.inference.LocalInferenceEngine safetensorEngine = null;
                System.out.println("DEBUG: Available engines in Instance<LocalInferenceEngine>:");
                for (tech.kayys.gollek.spi.inference.LocalInferenceEngine engine : engines) {
                    String className = engine.getClass().getName();
                    System.out.println("  - " + className);
                    if (className.contains("SafetensorEngine") || className.contains("DirectSafetensorBackend")) {
                        safetensorEngine = engine;
                    }
                }
                if (safetensorEngine == null) {
                    System.out.println("DEBUG: SafetensorEngine not found, falling back to first available engine");
                    safetensorEngine = engines.iterator().next();
                }
                InferenceResponse response = safetensorEngine.infer(request).await().indefinitely();
                if (response != null) {
                    outputContent = response.getContent();
                } else {
                    throw new RuntimeException("Safetensor engine returned null");
                }
            }
        } catch (Exception e) {
            System.err.println("Error running benchmark: " + e.getMessage());
            e.printStackTrace();
            return;
        }
        long elapsedMs = Math.max(1L, Duration.between(started, Instant.now()).toMillis());

        System.out.println("\nResult:\n" + outputContent);
        System.out.println("--------------------------------------------------");
        System.out.println("Metrics:");
        
        int outputTokens = outputContent.split("\\s+").length; // Rough fallback
        double tokensPerSec = (outputTokens / (double) elapsedMs) * 1000.0;
        
        System.out.println("Total Time : " + elapsedMs + " ms");
        System.out.printf("Throughput : %.2f tokens / sec\n", tokensPerSec);
        System.out.println("Provider   : " + provider);
        System.out.println("--------------------------------------------------");
    }
}
