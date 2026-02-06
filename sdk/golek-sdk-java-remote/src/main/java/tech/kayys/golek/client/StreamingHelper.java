package tech.kayys.golek.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.reactivex.Flowable;
import org.reactivestreams.Publisher;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.golek.client.exception.GolekClientException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.Flow;
import java.util.function.Consumer;

/**
 * Helper class for handling Server-Sent Events (SSE) for streaming inference and model operations.
 */
public class StreamingHelper {

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String apiKey;
    private final String defaultTenantId;

    public StreamingHelper(HttpClient httpClient, ObjectMapper objectMapper, String baseUrl, String apiKey, String defaultTenantId) {
        this.httpClient = httpClient;
        this.objectMapper = objectMapper;
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
        this.defaultTenantId = defaultTenantId;
    }

    /**
     * Creates a publisher for streaming inference chunks.
     *
     * @param requestBody The JSON request body for the inference request
     * @return A Publisher that emits StreamChunk objects
     */
    public Publisher<StreamChunk> createStreamPublisher(String requestBody) {
        return createSsePublisher(
            baseUrl + "/v1/inference/completions/stream",
            requestBody,
            StreamChunk.class
        );
    }

    /**
     * Creates a publisher for streaming model pull progress updates.
     *
     * @param modelSpec The model specification to pull
     * @param progressCallback Callback to receive progress updates
     * @return A Publisher that emits PullProgress objects
     */
    public Publisher<tech.kayys.golek.sdk.core.model.PullProgress> createModelPullStreamPublisher(
            String modelSpec,
            Consumer<tech.kayys.golek.sdk.core.model.PullProgress> progressCallback) {

        return Flowable.fromPublisher(subscriber -> {
            try {
                // Create request body
                java.util.Map<String, Object> requestBodyMap = new java.util.HashMap<>();
                requestBodyMap.put("model", modelSpec);

                String requestBody = objectMapper.writeValueAsString(requestBodyMap);

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/v1/models/pull/stream"))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + apiKey)
                        .header("X-Tenant-ID", defaultTenantId)
                        .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                        .timeout(Duration.ofMinutes(30)) // Longer timeout for model pulling
                        .build();

                HttpResponse<java.io.InputStream> response = httpClient.send(request,
                        HttpResponse.BodyHandlers.ofInputStream());

                if (response.statusCode() != 200) {
                    subscriber.onError(new GolekClientException("Model pull stream request failed with status: " + response.statusCode()));
                    return;
                }

                BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()));
                String line;

                while ((line = reader.readLine()) != null && !subscriber.isCancelled()) {
                    if (line.startsWith("data: ")) {
                        String data = line.substring(6); // Remove "data: " prefix

                        if ("[DONE]".equals(data)) {
                            break; // End of stream
                        }

                        try {
                            tech.kayys.golek.sdk.core.model.PullProgress progress =
                                objectMapper.readValue(data, tech.kayys.golek.sdk.core.model.PullProgress.class);

                            if (progressCallback != null) {
                                progressCallback.accept(progress);
                            }

                            subscriber.onNext(progress);
                        } catch (Exception e) {
                            subscriber.onError(new GolekClientException("Error parsing model pull progress: " + e.getMessage(), e));
                            return;
                        }
                    }
                }

                subscriber.onComplete();
            } catch (IOException | InterruptedException e) {
                if (!subscriber.isCancelled()) {
                    subscriber.onError(new GolekClientException("Error during model pull streaming: " + e.getMessage(), e));
                }
            }
        });
    }

    /**
     * Generic method to create an SSE publisher for different response types.
     *
     * @param endpoint The API endpoint to call
     * @param requestBody The JSON request body
     * @param responseType The class of the response type
     * @return A Publisher that emits objects of the specified type
     */
    private <T> Publisher<T> createSsePublisher(String endpoint, String requestBody, Class<T> responseType) {
        return Flowable.fromPublisher(subscriber -> {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(endpoint))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + apiKey)
                        .header("X-Tenant-ID", defaultTenantId)
                        .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                        .timeout(Duration.ofMinutes(10)) // Longer timeout for streaming
                        .build();

                HttpResponse<java.io.InputStream> response = httpClient.send(request,
                        HttpResponse.BodyHandlers.ofInputStream());

                if (response.statusCode() != 200) {
                    subscriber.onError(new GolekClientException("Streaming request failed with status: " + response.statusCode()));
                    return;
                }

                BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()));
                String line;

                while ((line = reader.readLine()) != null && !subscriber.isCancelled()) {
                    if (line.startsWith("data: ")) {
                        String data = line.substring(6); // Remove "data: " prefix

                        if ("[DONE]".equals(data)) {
                            break; // End of stream
                        }

                        try {
                            T chunk = objectMapper.readValue(data, responseType);
                            subscriber.onNext(chunk);
                        } catch (Exception e) {
                            subscriber.onError(new GolekClientException("Error parsing stream chunk: " + e.getMessage(), e));
                            return;
                        }
                    }
                }

                subscriber.onComplete();
            } catch (IOException | InterruptedException e) {
                if (!subscriber.isCancelled()) {
                    subscriber.onError(new GolekClientException("Error during streaming: " + e.getMessage(), e));
                }
            }
        });
    }
}