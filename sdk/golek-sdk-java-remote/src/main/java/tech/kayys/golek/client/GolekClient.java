package tech.kayys.golek.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.ProviderInfo;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.golek.client.exception.GolekClientException;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.exception.SdkException;
import tech.kayys.golek.sdk.core.model.AsyncJobStatus;
import tech.kayys.golek.sdk.core.model.BatchInferenceRequest;
import tech.kayys.golek.spi.auth.ApiKeyConstants;

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
    private String preferredProvider;

    private GolekClient(Builder builder) {
        this.baseUrl = builder.baseUrl;
        this.apiKey = normalizeApiKey(builder.apiKey);
        this.preferredProvider = builder.preferredProvider;

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
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
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
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(Duration.ofSeconds(10)) // Short timeout for job submission
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            // For job submission, we expect a response with the job ID
            java.util.Map<String, Object> jsonResponse = handleResponse(response, java.util.Map.class);

            return (String) jsonResponse.get("jobId");
        } catch (GolekClientException e) {
            // Convert GolekClientException to SdkException
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("Failed to submit async job", e);
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
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .timeout(Duration.ofSeconds(10)) // Short timeout for status check
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response, AsyncJobStatus.class);
        } catch (GolekClientException e) {
            // Convert GolekClientException to SdkException
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("Failed to get job status", e);
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
                throw new SdkException("Job polling interrupted", e);
            }
        }

        throw new SdkException("Job " + jobId + " did not complete within the specified time");
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
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(Duration.ofMinutes(5)) // Longer timeout for batch processing
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response,
                objectMapper.getTypeFactory().constructCollectionType(java.util.List.class, InferenceResponse.class)
            );
        } catch (GolekClientException e) {
            // Convert GolekClientException to SdkException
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("Failed to perform batch inference", e);
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

            StreamingHelper helper = new StreamingHelper(httpClient, objectMapper, baseUrl, apiKey);

            // Convert the Flow.Publisher to Mutiny Multi
            return Multi.createFrom().publisher(helper.createStreamPublisher(requestBody));
        } catch (Exception e) {
            return Multi.createFrom().failure(new SdkException("Failed to initiate streaming completion", e));
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
     * Lists all available inference providers.
     *
     * @return List of provider information
     * @throws SdkException if the request fails
     */
    @Override
    public List<ProviderInfo> listAvailableProviders() throws SdkException {
        try {
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/providers"))
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .timeout(Duration.ofSeconds(10))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response,
                objectMapper.getTypeFactory().constructCollectionType(java.util.List.class, ProviderInfo.class)
            );
        } catch (GolekClientException e) {
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_PROVIDER_LIST", "Failed to list providers", e);
        }
    }

    /**
     * Gets detailed information about a specific provider.
     *
     * @param providerId The provider ID
     * @return Provider information
     * @throws SdkException if the provider is not found
     */
    @Override
    public ProviderInfo getProviderInfo(String providerId) throws SdkException {
        try {
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/providers/" + URLEncoder.encode(providerId, StandardCharsets.UTF_8)))
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .timeout(Duration.ofSeconds(10))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response, ProviderInfo.class);
        } catch (GolekClientException e) {
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_PROVIDER_INFO", "Failed to get provider info", e);
        }
    }

    /**
     * Sets the preferred provider for subsequent requests.
     *
     * @param providerId The provider ID
     * @throws SdkException if the provider is not available
     */
    @Override
    public void setPreferredProvider(String providerId) throws SdkException {
        // Validate provider exists by fetching its info
        getProviderInfo(providerId);
        this.preferredProvider = providerId;
    }

    /**
     * Gets the currently preferred provider ID.
     *
     * @return The preferred provider ID, or empty if none is set
     */
    @Override
    public java.util.Optional<String> getPreferredProvider() {
        return java.util.Optional.ofNullable(preferredProvider);
    }

    /**
     * Builder for creating GolekClient instances.
     */
    public static class Builder {
        private String baseUrl = "http://localhost:8080";
        private String apiKey;
        private String preferredProvider;
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

        public Builder preferredProvider(String preferredProvider) {
            this.preferredProvider = preferredProvider;
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
            return new GolekClient(this);
        }
    }

    // ==================== Model Operations ====================

    @Override
    public List<tech.kayys.golek.sdk.core.model.ModelInfo> listModels() throws SdkException {
        try {
            String url = String.format("%s/v1/models", baseUrl);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(30))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response,
                objectMapper.getTypeFactory().constructCollectionType(java.util.List.class, tech.kayys.golek.sdk.core.model.ModelInfo.class)
            );
        } catch (GolekClientException e) {
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_LIST", "Failed to list models", e);
        }
    }

    @Override
    public List<tech.kayys.golek.sdk.core.model.ModelInfo> listModels(int offset, int limit) throws SdkException {
        try {
            String url = String.format("%s/v1/models?offset=%d&limit=%d", baseUrl, offset, limit);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(30))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            return handleResponse(response,
                objectMapper.getTypeFactory().constructCollectionType(java.util.List.class, tech.kayys.golek.sdk.core.model.ModelInfo.class)
            );
        } catch (GolekClientException e) {
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_LIST", "Failed to list models", e);
        }
    }

    @Override
    public java.util.Optional<tech.kayys.golek.sdk.core.model.ModelInfo> getModelInfo(String modelId) throws SdkException {
        try {
            String encodedModelId = URLEncoder.encode(modelId, StandardCharsets.UTF_8);
            String url = String.format("%s/v1/models/%s", baseUrl, encodedModelId);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(30))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 404) {
                return java.util.Optional.empty(); // Model not found
            }

            tech.kayys.golek.sdk.core.model.ModelInfo modelInfo = handleResponse(response, tech.kayys.golek.sdk.core.model.ModelInfo.class);
            return java.util.Optional.of(modelInfo);
        } catch (GolekClientException e) {
            if (e.getErrorCode().equals("CLIENT_ERROR") && e.getMessage().contains("404")) {
                return java.util.Optional.empty(); // Model not found
            }
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_INFO", "Failed to get model info", e);
        }
    }

    @Override
    public void pullModel(String modelSpec, java.util.function.Consumer<tech.kayys.golek.sdk.core.model.PullProgress> progressCallback) throws SdkException {
        try {
            // First, try to initiate the model pull
            java.util.Map<String, Object> requestBodyMap = new java.util.HashMap<>();
            requestBodyMap.put("model", modelSpec);

            String requestBody = objectMapper.writeValueAsString(requestBodyMap);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/models/pull"))
                    .header("Content-Type", "application/json")
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(Duration.ofMinutes(5)) // Shorter timeout for initial request
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            // If the server immediately returns success, call the callback
            if (response.statusCode() == 200) {
                if (progressCallback != null) {
                    tech.kayys.golek.sdk.core.model.PullProgress progress = new tech.kayys.golek.sdk.core.model.PullProgress(
                        modelSpec, "completed", 100.0, "Model pulled successfully"
                    );
                    progressCallback.accept(progress);
                }
                return;
            } else if (response.statusCode() == 202) {
                // Server accepted the request, now we need to stream progress updates
                StreamingHelper helper = new StreamingHelper(httpClient, objectMapper, baseUrl, apiKey);

                // Create a publisher for the streaming progress
                Publisher<tech.kayys.golek.sdk.core.model.PullProgress> publisher =
                    helper.createModelPullStreamPublisher(modelSpec, progressCallback);

                // Subscribe to the publisher to process the stream
                io.reactivex.Flowable.fromPublisher(publisher)
                    .blockingSubscribe(
                        progress -> {
                            // Progress updates are handled by the callback in the publisher
                        },
                        error -> {
                            throw new SdkException("SDK_ERR_MODEL_PULL", "Error during model pull streaming: " + error.getMessage(), error);
                        },
                        () -> {
                            // Completed
                        }
                    );
            } else {
                // Parse error response
                java.util.Map<String, Object> errorResponse = handleResponse(response, java.util.Map.class);
                String errorMessage = (String) errorResponse.getOrDefault("error", "Unknown error during model pull");
                throw new SdkException("SDK_ERR_MODEL_PULL", "Failed to pull model: " + errorMessage);
            }
        } catch (GolekClientException e) {
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_PULL", "Failed to pull model", e);
        }
    }

    @Override
    public void deleteModel(String modelId) throws SdkException {
        try {
            String encodedModelId = URLEncoder.encode(modelId, StandardCharsets.UTF_8);
            String url = String.format("%s/v1/models/%s", baseUrl, encodedModelId);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header(ApiKeyConstants.HEADER_API_KEY, apiKey)
                    .header(ApiKeyConstants.HEADER_AUTHORIZATION, ApiKeyConstants.authorizationValue(apiKey))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(30))
                    .DELETE()
                    .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200 || response.statusCode() == 204) {
                // Success - model deleted
                return;
            } else {
                // Parse error response
                java.util.Map<String, Object> errorResponse = handleResponse(response, java.util.Map.class);
                String errorMessage = (String) errorResponse.getOrDefault("error", "Unknown error during model deletion");
                throw new SdkException("SDK_ERR_MODEL_DELETE", "Failed to delete model: " + errorMessage);
            }
        } catch (GolekClientException e) {
            throw new SdkException(e.getErrorCode(), e.getMessage(), e);
        } catch (Exception e) {
            throw new SdkException("SDK_ERR_MODEL_DELETE", "Failed to delete model", e);
        }
    }

    private static String normalizeApiKey(String apiKey) {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }
}
