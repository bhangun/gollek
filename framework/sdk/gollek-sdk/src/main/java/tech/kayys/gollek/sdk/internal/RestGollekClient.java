package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.api.GollekClient;
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class RestGollekClient implements GollekClient {

    private final HttpClient httpClient;
    private final String endpoint;
    private final String model;
    private final String backend;
    private final int maxTokens;
    private final float temperature;
    private final ObjectMapper mapper = new ObjectMapper();

    public RestGollekClient(String endpoint, String model, String backend, int maxTokens, float temperature) {
        this.endpoint = endpoint;
        this.model = model;
        this.backend = backend;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    @Override
    public GenerationResult generate(String prompt) {
        return generate(GenerationRequest.of(prompt));
    }

    @Override
    public GenerationResult generate(GenerationRequest request) {
        String baseUri = endpoint.endsWith("/") ? endpoint : endpoint + "/";
        try {
            String requestBody = mapper.writeValueAsString(request);
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUri + "v1/generate"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new RuntimeException("Inference failed: " + response.body());
            }
            
            // Assuming response is something like {"text": "...", "tokenCount": 10, "promptTokens": 5, "durationMs": 100}
            // For now, parse it to GenerationResult if we have the right model mapping, otherwise build it manually
            // We'll use a basic stub parser for demonstration
            return mapper.readValue(response.body(), GenerationResult.class);
        } catch (Exception e) {
            throw new RuntimeException("REST Generation failed", e);
        }
    }

    @Override
    public GenerationStream generateStream(String prompt) {
        // HTTP Streaming via SSE goes here
        // We will just use the Embedded stream logic over HTTP for now as a placeholder
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
        // Mocking embedding REST call
        return new float[768];
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
        return feature == Feature.STREAMING || feature == Feature.BATCH_INFERENCE;
    }

    @Override
    public void close() {
        // HttpClient handles resources
    }
}
