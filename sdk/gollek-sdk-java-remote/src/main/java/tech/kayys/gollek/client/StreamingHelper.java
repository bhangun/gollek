package tech.kayys.gollek.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;

import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.client.exception.GollekClientException;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.spi.auth.ApiKeyConstants;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.function.Consumer;

/**
 * Helper for consuming Server-Sent Events (SSE) streams from the Gollek HTTP API.
 *
 * <p>Used internally by {@link GollekClient} to handle streaming inference responses
 * and model-pull progress updates. Each public method returns a Mutiny {@link Multi}
 * that emits typed objects parsed from {@code data:} SSE lines.
 *
 * <p>Cancellation is cooperative: once the subscriber cancels, the SSE loop exits
 * on the next line read.
 */
public class StreamingHelper {

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String apiKey;

    /**
     * Constructs a streaming helper.
     *
     * @param httpClient   shared HTTP client
     * @param objectMapper Jackson mapper for deserializing SSE payloads
     * @param baseUrl      base URL of the Gollek server (e.g. {@code "http://localhost:8080"})
     * @param apiKey       bearer token for authentication; {@code null} or blank uses the community key
     */
    public StreamingHelper(HttpClient httpClient, ObjectMapper objectMapper, String baseUrl, String apiKey) {
        this.httpClient = httpClient;
        this.objectMapper = objectMapper;
        this.baseUrl = baseUrl;
        this.apiKey = normalizeApiKey(apiKey);
    }

    /**
     * Creates a {@link Multi} that streams {@link StreamingInferenceChunk} objects
     * from the {@code /v1/inference/completions/stream} endpoint.
     *
     * @param requestBody JSON-serialized {@link tech.kayys.gollek.spi.inference.InferenceRequest}
     * @return a {@link Multi} emitting inference chunks until {@code [DONE]} or cancellation
     */
    public Multi<StreamingInferenceChunk> createStreamPublisher(String requestBody) {
        return createSseMulti(
                baseUrl + "/v1/inference/completions/stream",
                requestBody,
                StreamingInferenceChunk.class);
    }

    /**
     * Creates a {@link Multi} that streams {@link PullProgress} updates while a model
     * is being pulled from the {@code /v1/models/pull/stream} endpoint.
     *
     * <p>Each emitted {@link PullProgress} is also forwarded to {@code progressCallback}
     * if non-null, allowing callers to receive updates outside the reactive pipeline.
     *
     * @param modelSpec        model specification to pull (e.g. {@code "hf:TheBloke/Llama-2-7B-GGUF"})
     * @param progressCallback optional callback invoked for each progress update; may be {@code null}
     * @return a {@link Multi} emitting pull progress events until completion or cancellation
     */
    public Multi<PullProgress> createModelPullStreamPublisher(
            String modelSpec,
            Consumer<PullProgress> progressCallback) {

        return Multi.createFrom().emitter(emitter -> {
            try {
                // Create request body
                java.util.Map<String, Object> requestBodyMap = new java.util.HashMap<>();
                requestBodyMap.put("model", modelSpec);

                String requestBody = objectMapper.writeValueAsString(requestBodyMap);

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/v1/models/pull/stream"))
                        .header("Content-Type", "application/json")
                        .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                        .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                        .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                        .timeout(Duration.ofMinutes(30)) // Longer timeout for model pulling
                        .build();

                HttpResponse<java.io.InputStream> response = httpClient.send(request,
                        HttpResponse.BodyHandlers.ofInputStream());

                if (response.statusCode() != 200) {
                    emitter.fail(new GollekClientException(
                            "Model pull stream request failed with status: " + response.statusCode()));
                    return;
                }

                try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
                    String line;
                    while (!emitter.isCancelled() && (line = reader.readLine()) != null) {
                        if (line.startsWith("data: ")) {
                            String data = line.substring(6); // Remove "data: " prefix

                            if ("[DONE]".equals(data)) {
                                break; // End of stream
                            }

                            try {
                                PullProgress progress = objectMapper.readValue(data,
                                        PullProgress.class);

                                if (progressCallback != null) {
                                    progressCallback.accept(progress);
                                }

                                emitter.emit(progress);
                            } catch (Exception e) {
                                emitter.fail(new GollekClientException(
                                        "Error parsing model pull progress: " + e.getMessage(), e));
                                return;
                            }
                        }
                    }
                }

                if (!emitter.isCancelled()) {
                    emitter.complete();
                }
            } catch (IOException | InterruptedException e) {
                if (!emitter.isCancelled()) {
                    emitter.fail(
                            new GollekClientException("Error during model pull streaming: " + e.getMessage(), e));
                }
            }
        });
    }

    /**
     * Generic SSE consumer that POSTs {@code requestBody} to {@code endpoint} and
     * deserializes each {@code data:} line into an instance of {@code responseType}.
     *
     * @param <T>          the element type
     * @param endpoint     full URL of the SSE endpoint
     * @param requestBody  JSON request body
     * @param responseType class to deserialize each SSE data line into
     * @return a {@link Multi} emitting deserialized objects
     */
    private <T> Multi<T> createSseMulti(String endpoint, String requestBody, Class<T> responseType) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(endpoint))
                        .header("Content-Type", "application/json")
                        .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                        .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                        .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                        .timeout(Duration.ofMinutes(10)) // Longer timeout for streaming
                        .build();

                HttpResponse<java.io.InputStream> response = httpClient.send(request,
                        HttpResponse.BodyHandlers.ofInputStream());

                if (response.statusCode() != 200) {
                    emitter.fail(
                            new GollekClientException(
                                    "Streaming request failed with status: " + response.statusCode()));
                    return;
                }

                try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
                    String line;
                    while (!emitter.isCancelled() && (line = reader.readLine()) != null) {
                        if (line.startsWith("data: ")) {
                            String data = line.substring(6); // Remove "data: " prefix

                            if ("[DONE]".equals(data)) {
                                break; // End of stream
                            }

                            try {
                                T chunk = objectMapper.readValue(data, responseType);
                                emitter.emit(chunk);
                            } catch (Exception e) {
                                emitter.fail(
                                        new GollekClientException("Error parsing stream chunk: " + e.getMessage(), e));
                                return;
                            }
                        }
                    }
                }

                if (!emitter.isCancelled()) {
                    emitter.complete();
                }
            } catch (IOException | InterruptedException e) {
                if (!emitter.isCancelled()) {
                    emitter.fail(new GollekClientException("Error during streaming: " + e.getMessage(), e));
                }
            }
        });
    }

    private static String normalizeApiKey(String apiKey) {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }
}
