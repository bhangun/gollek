package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.api.GollekClient;
import tech.kayys.gollek.spi.model.ModelInfo;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class CliGollekClient implements GollekClient {

    private final String model;
    private final String backend;
    private final int maxTokens;
    private final float temperature;
    private final ObjectMapper mapper = new ObjectMapper();

    public CliGollekClient(String model, String backend, int maxTokens, float temperature) {
        this.model = model;
        this.backend = backend;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
    }

    @Override
    public GenerationResult generate(String prompt) {
        return generate(GenerationRequest.of(prompt));
    }

    @Override
    public GenerationResult generate(GenerationRequest request) {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "gollek-cli", "run",
                    "--model", model,
                    "--prompt", request.prompt(),
                    "--max-tokens", String.valueOf(request.maxTokens() > 0 ? request.maxTokens() : maxTokens),
                    "--temperature", String.valueOf(request.temperature() > 0 ? request.temperature() : temperature),
                    "--json", "--no-banner", "--stream=false"
            );

            if (backend != null && !backend.isBlank()) {
                pb.command().add("--backend");
                pb.command().add(backend);
            }

            Process process = pb.start();
            int exitCode = process.waitFor();
            
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            String errorOutput = new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);

            if (exitCode != 0) {
                throw new RuntimeException("CLI inference failed (exit " + exitCode + "): " + errorOutput);
            }

            try {
                return mapper.readValue(output, GenerationResult.class);
            } catch (Exception e) {
                // Fallback parsing if JSON isn't perfect
                return new GenerationResult(output.trim(), 0, 0, 0);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to run CLI inference", e);
        }
    }

    @Override
    public GenerationStream generateStream(String prompt) {
        EmbeddedGenerationStream stream = new EmbeddedGenerationStream();
        CompletableFuture.runAsync(() -> {
            try {
                GenerationResult res = generate(prompt);
                stream.emitToken(res.text());
                stream.emitComplete();
            } catch (Exception e) {
                stream.emitError(e);
            }
        });
        return stream;
    }

    @Override
    public List<GenerationResult> generateBatch(List<String> prompts) {
        return prompts.stream().map(this::generate).toList();
    }

    @Override
    public float[] embed(String text) {
        return new float[0];
    }

    @Override
    public List<float[]> embedBatch(List<String> texts) {
        return texts.stream().map(this::embed).toList();
    }

    @Override
    public ModelInfo modelInfo() {
        return ModelInfo.builder().modelId(model).format(backend).build();
    }

    @Override
    public boolean supports(Feature feature) {
        return feature == Feature.BATCH_INFERENCE;
    }

    @Override
    public void close() {
    }
}
