package tech.kayys.gollek.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.stream.StreamChunk;
import tech.kayys.gollek.client.exception.GollekClientException;
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
 * Helper class for handling Server-Sent Events (SSE) for streaming inference
 * and model operations.
 */
public class StreamingHelper {

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String apiKey;

    public StreamingHelper(HttpClient httpClient, ObjectMapper objectMapper, String baseUrl, String apiKey) {
        this.httpClient = httpClient;
        this.objectMapper = objectMapper;
        this.baseUrl = baseUrl;
        this.apiKey = normalizeApiKey(apiKey);
    }

    /**
     * Creates a publisher for streaming inference chunks.
     *
     * @param requestBody The JSON request body for the inference request
     * @return A Multi that emits StreamChunk objects
     */
    public Multi<StreamChunk> createStreamPublisher(String requestBody) {
        return createSseMulti(
                baseUrl + "/v1/inference/completions/stream",
                requestBody,
                StreamChunk.class);
    }

    /**
     * Creates a publisher for streaming model pull progress updates.
     *
     * @param modelSpec        The model specification to pull
     * @param progressCallback Callback to receive progress updates
     * @return A Multi that emits PullProgress objects
     */
    public Multi<tech.kayys.gollek.sdk.core.model.PullProgress> createModelPullStreamPublisher(
            String modelSpec,
            Consumer<tech.kayys.gollek.sdk.core.model.PullProgress> progressCallback) {

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
                                tech.kayys.gollek.sdk.core.model.PullProgress progress = objectMapper.readValue(data,
                                        tech.kayys.gollek.sdk.core.model.PullProgress.class);

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
     * Generic method to create an SSE multi for different response types.
     *
     * @param endpoint     The API endpoint to call
     * @param requestBody  The JSON request body
     * @param responseType The class of the response type
     * @return A Multi that emits objects of the specified type
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
                            new GollekClientException("Streaming request failed with status: " + response.statusCode()));
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
