package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.sdk.model.*;
import tech.kayys.gollek.sdk.internal.util.Json;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Remote (HTTP) implementation of {@link GollekClient}.
 *
 * <p>Communicates with a Gollek sidecar or standalone server via REST endpoints.
 * Uses the JDK {@link java.net.http.HttpClient} for a minimal dependency footprint.
 *
 * <p>Instances are created by {@link GollekClientBuilderImpl} and should not be
 * instantiated directly.
 */
public class RemoteGollekClient implements GollekClient {

    private final HttpClient httpClient;
    /** Base URL of the remote Gollek server (e.g. {@code https://gollek.example.com}). */
    private final String endpoint;
    private final String apiKey;
    private final Duration timeout;

    /**
     * Package-private constructor — use {@link GollekClient#builder()} to obtain instances.
     *
     * @param builder the configured builder carrying endpoint, apiKey, and timeout
     */
    RemoteGollekClient(GollekClientBuilderImpl builder) {
        this.endpoint = builder.endpoint;
        this.apiKey = builder.apiKey;
        this.timeout = Duration.ofMillis(builder.timeoutMillis);
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(timeout)
                .build();
    }

    @Override
    public GenerationResponse generate(GenerationRequest request) {
        try {
            return generateAsync(request).join();
        } catch (Exception e) {
            throw new RuntimeException("Generation failed", e);
        }
    }

    @Override
    public CompletableFuture<GenerationResponse> generateAsync(GenerationRequest request) {
        String baseUri = endpoint.endsWith("/") ? endpoint : endpoint + "/";
        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(baseUri + "v1/generate"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .timeout(timeout)
                .POST(HttpRequest.BodyPublishers.ofString(Json.stringify(request)))
                .build();

        return httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() != 200) {
                        throw new RuntimeException("Inference failed: " + response.body());
                    }
                    return Json.parse(response.body(), GenerationResponse.class);
                });
    }

    @Override
    public GenerationStream generateStream(GenerationRequest request) {
        String baseUri = endpoint.endsWith("/") ? endpoint : endpoint + "/";
        URI uri = URI.create(baseUri + "v1/generate/stream");
        return new HttpStreamClient(httpClient, uri, apiKey, timeout).stream(request);
    }

    @Override
    public EmbeddingResponse embed(EmbeddingRequest request) {
        String baseUri = endpoint.endsWith("/") ? endpoint : endpoint + "/";
        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(baseUri + "v1/embeddings"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .timeout(timeout)
                .POST(HttpRequest.BodyPublishers.ofString(Json.stringify(request)))
                .build();

        try {
            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new RuntimeException("Embedding failed: " + response.body());
            }
            return Json.parse(response.body(), EmbeddingResponse.class);
        } catch (Exception e) {
            throw new RuntimeException("Embedding request failed", e);
        }
    }

    @Override
    public List<ModelInfo> listModels() {
        String baseUri = endpoint.endsWith("/") ? endpoint : endpoint + "/";
        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(baseUri + "v1/models"))
                .header("Authorization", "Bearer " + apiKey)
                .timeout(timeout)
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new RuntimeException("Failed to list models: " + response.body());
            }
            return Json.parseList(response.body(), ModelInfo.class);
        } catch (Exception e) {
            throw new RuntimeException("Model list request failed", e);
        }
    }

    @Override
    public ModelInfo getModelInfo(String modelId) {
        String baseUri = endpoint.endsWith("/") ? endpoint : endpoint + "/";
        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(baseUri + "v1/models/" + modelId))
                .header("Authorization", "Bearer " + apiKey)
                .timeout(timeout)
                .GET()
                .build();

        try {
            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new RuntimeException("Failed to get model info: " + response.body());
            }
            return Json.parse(response.body(), ModelInfo.class);
        } catch (Exception e) {
            throw new RuntimeException("Model info request failed", e);
        }
    }

    @Override
    public void close() {
        // HttpClient handles its own resources
    }
}
