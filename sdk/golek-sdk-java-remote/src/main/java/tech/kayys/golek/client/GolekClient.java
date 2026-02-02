package tech.kayys.golek.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.client.exception.GolekClientException;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.exception.SdkException;
import tech.kayys.golek.sdk.core.model.AsyncJobStatus;
import tech.kayys.golek.sdk.core.model.BatchInferenceRequest;

import javax.net.ssl.SSLContext;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Remote client for interacting with the Golek inference engine via HTTP API.
 * Implements the core GolekSdk interface for remote access.
 */
public class GolekClient implements GolekSdk {

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String apiKey;
    private final String defaultTenantId;

    private GolekClient(Builder builder) {
        this.baseUrl = builder.baseUrl;
        this.apiKey = builder.apiKey;
        this.defaultTenantId = builder.defaultTenantId;

        HttpClient.Builder clientBuilder = HttpClient.newBuilder()
                .connectTimeout(builder.connectTimeout)
                .followRedirects(HttpClient.Redirect.NORMAL);

        if (builder.sslContext != null) {
            clientBuilder.sslContext(builder.sslContext);
        }

        this.httpClient = clientBuilder.build();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Creates a new inference request synchronously.
     *
     * @param request The inference request
     * @return The inference response
     * @throws SdkException if the request fails
     */
    @Override
    public InferenceResponse createCompletion(InferenceRequest request) throws SdkException {
        try {
            String requestBody = objectMapper.writeValueAsString(request);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/inference/completions"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .header("X-Tenant-ID", defaultTenantId)
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(Duration.ofSeconds(60)) // Add timeout for sync requests
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response, InferenceResponse.class);
        } catch (GolekClientException e) {
            // Convert GolekClientException to SdkException
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("Failed to create completion", e);
        }
    }

    /**
     * Creates a new inference request asynchronously.
     *
     * @param request The inference request
     * @return A CompletableFuture that will complete with the inference response
     */
    @Override
    public CompletableFuture<InferenceResponse> createCompletionAsync(InferenceRequest request) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return createCompletion(request);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    /**
     * Submits an async inference job.
     *
     * @param request The inference request
     * @return The job ID
     * @throws SdkException if the request fails
     */
    @Override
    public String submitAsyncJob(InferenceRequest request) throws SdkException {
        try {
            String requestBody = objectMapper.writeValueAsString(request);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/inference/jobs"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .header("X-Tenant-ID", defaultTenantId)
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(Duration.ofSeconds(10)) // Short timeout for job submission
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            // For job submission, we expect a response with the job ID
            java.util.Map<String, Object> jsonResponse = handleResponse(response, java.util.Map.class);

            return (String) jsonResponse.get("jobId");
        } catch (GolekClientException e) {
            throw e; // Re-throw client exceptions
        } catch (Exception e) {
            throw new GolekClientException("Failed to submit async job", e);
        }
    }

    /**
     * Gets the status of an async inference job.
     *
     * @param jobId The job ID
     * @return The job status
     * @throws SdkException if the request fails
     */
    @Override
    public AsyncJobStatus getJobStatus(String jobId) throws SdkException {
        try {
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/inference/jobs/" + URLEncoder.encode(jobId, StandardCharsets.UTF_8)))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("X-Tenant-ID", defaultTenantId)
                    .timeout(Duration.ofSeconds(10)) // Short timeout for status check
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response, AsyncJobStatus.class);
        } catch (GolekClientException e) {
            throw e; // Re-throw client exceptions
        } catch (Exception e) {
            throw new GolekClientException("Failed to get job status", e);
        }
    }

    /**
     * Waits for an async job to complete.
     *
     * @param jobId The job ID
     * @param maxWaitTime Maximum time to wait
     * @param pollInterval Interval between status checks
     * @return The final job status
     * @throws SdkException if the request fails or times out
     */
    @Override
    public AsyncJobStatus waitForJob(String jobId, Duration maxWaitTime, Duration pollInterval) throws SdkException {
        long startTime = System.currentTimeMillis();
        long maxWaitMillis = maxWaitTime.toMillis();

        while (System.currentTimeMillis() - startTime < maxWaitMillis) {
            AsyncJobStatus status = getJobStatus(jobId);

            if (status.isComplete()) {
                return status;
            }

            try {
                Thread.sleep(pollInterval.toMillis());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new GolekClientException("Job polling interrupted", e);
            }
        }

        throw new GolekClientException("Job " + jobId + " did not complete within the specified time");
    }

    /**
     * Performs batch inference.
     *
     * @param batchRequest The batch inference request
     * @return List of inference responses
     * @throws SdkException if the request fails
     */
    @Override
    public List<InferenceResponse> batchInference(BatchInferenceRequest batchRequest) throws SdkException {
        try {
            String requestBody = objectMapper.writeValueAsString(batchRequest);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/inference/batch"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + apiKey)
                    .header("X-Tenant-ID", defaultTenantId)
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(Duration.ofMinutes(5)) // Longer timeout for batch processing
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response,
                objectMapper.getTypeFactory().constructCollectionType(java.util.List.class, InferenceResponse.class)
            );
        } catch (GolekClientException e) {
            throw e; // Re-throw client exceptions
        } catch (Exception e) {
            throw new GolekClientException("Failed to perform batch inference", e);
        }
    }

    /**
     * Creates a streaming inference request.
     *
     * @param request The inference request
     * @return A Multi that emits StreamChunk objects
     */
    @Override
    public Multi<StreamChunk> streamCompletion(InferenceRequest request) {
        try {
            String requestBody = objectMapper.writeValueAsString(request);

            StreamingHelper helper = new StreamingHelper(httpClient, objectMapper, baseUrl, apiKey, defaultTenantId);

            // Convert the Flow.Publisher to Mutiny Multi
            return Multi.createFrom().publisher(helper.createStreamPublisher(requestBody));
        } catch (Exception e) {
            return Multi.createFrom().failure(new GolekClientException("Failed to initiate streaming completion", e));
        }
    }

    /**
     * Handles HTTP responses and maps them to appropriate objects or exceptions.
     *
     * @param response The HTTP response
     * @param responseType The class of the expected response type
     * @return The parsed response object
     * @throws GolekClientException if the response indicates an error
     */
    private <T> T handleResponse(HttpResponse<String> response, Class<T> responseType) throws GolekClientException {
        int statusCode = response.statusCode();
        String responseBody = response.body();

        if (statusCode >= 200 && statusCode < 300) {
            try {
                return objectMapper.readValue(responseBody, responseType);
            } catch (Exception e) {
                throw new GolekClientException("Failed to parse response: " + e.getMessage(), e);
            }
        }

        // Handle specific error codes
        switch (statusCode) {
            case 400:
                throw new GolekClientException("Bad request: " + responseBody);
            case 401:
            case 403:
                throw new AuthenticationException("Authentication failed: " + responseBody);
            case 429:
                // Extract retry-after header if present
                int retryAfter = response.headers().firstValue("Retry-After")
                    .map(Integer::parseInt)
                    .orElse(60);
                throw new RateLimitException("Rate limit exceeded: " + responseBody, retryAfter);
            case 404:
                throw new GolekClientException("Resource not found: " + responseBody);
            case 422:
                throw new ModelException(null, "Unprocessable entity: " + responseBody);
            case 500:
                throw new GolekClientException("Internal server error: " + responseBody);
            case 503:
                throw new GolekClientException("Service unavailable: " + responseBody);
            default:
                throw new GolekClientException("Request failed with status: " + statusCode + ", body: " + responseBody);
        }
    }

    /**
     * Builder for creating GolekClient instances.
     */
    public static class Builder {
        private String baseUrl = "http://localhost:8080";
        private String apiKey;
        private String defaultTenantId = "default";
        private Duration connectTimeout = Duration.ofSeconds(30);
        private SSLContext sslContext;

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder defaultTenantId(String tenantId) {
            this.defaultTenantId = tenantId;
            return this;
        }

        public Builder connectTimeout(Duration timeout) {
            this.connectTimeout = timeout;
            return this;
        }

        public Builder sslContext(SSLContext sslContext) {
            this.sslContext = sslContext;
            return this;
        }

        public GolekClient build() {
            if (apiKey == null || apiKey.trim().isEmpty()) {
                throw new IllegalArgumentException("API key is required");
            }
            return new GolekClient(this);
        }
    }
}