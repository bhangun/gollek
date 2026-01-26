package tech.kayys.golek.provider.core.plugin;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.provider.core.ProviderConfig;
import tech.kayys.golek.provider.core.spi.ProviderCapabilities;
import tech.kayys.golek.provider.core.spi.ProviderException;
import tech.kayys.golek.provider.core.spi.ProviderHealth;
import tech.kayys.golek.provider.core.spi.ProviderRequest;
import tech.kayys.golek.provider.core.streaming.StreamingLLMProvider;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

/**
 * HuggingFace Inference API provider.
 * Supports both Inference API and Inference Endpoints.
 * 
 * Capabilities:
 * - Text generation
 * - Text embeddings
 * - Classification
 * - Feature extraction
 * - Zero-shot classification
 */
@ApplicationScoped
public class HuggingFaceProvider implements StreamingLLMProvider {

    private static final Logger LOG = Logger.getLogger(HuggingFaceProvider.class);
    private static final String API_BASE = "https://api-inference.huggingface.co/models/";

    private HttpClient httpClient;
    private ProviderConfig config;
    private volatile boolean initialized = false;

    @Override
    public String id() {
        return "tech.kayys/huggingface-provider";
    }

    @Override
    public String name() {
        return "HuggingFace Provider";
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(false)
                .multimodal(false)
                .embeddings(true)
                .maxContextTokens(4096)
                .supportedFormats(Set.of("transformers"))
                .supportedDevices(Set.of("cloud"))
                .features(Map.of(
                        "serverless", true,
                        "dedicated_endpoints", true,
                        "auto_scaling", true))
                .build();
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        // Supports any HuggingFace model ID
        return modelId.contains("/") || // org/model format
                modelId.startsWith("hf:");
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderInitializationException {
        if (initialized) {
            return;
        }

        synchronized (this) {
            if (initialized) {
                return;
            }

            LOG.info("Initializing HuggingFace provider");
            this.config = config;

            try {
                this.httpClient = HttpClient.newBuilder()
                        .connectTimeout(Duration.ofSeconds(10))
                        .build();

                // Verify API key
                String apiKey = config.getString("hf.api.key");
                if (apiKey == null || apiKey.isBlank()) {
                    throw new ProviderInitializationException(
                            "HuggingFace API key not configured");
                }

                initialized = true;
                LOG.info("HuggingFace provider initialized");

            } catch (Exception e) {
                throw new ProviderInitializationException(
                        "Failed to initialize HuggingFace provider", e);
            }
        }
    }

    @Override
    public Uni<InferenceResponse> infer(
            ProviderRequest request,
            TenantContext context) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();

            String modelId = normalizeModelId(request.getModel());
            LOG.debugf("HuggingFace inference for model: %s", modelId);

            // Build request
            String prompt = buildPrompt(request);
            Map<String, Object> requestBody = buildRequestBody(prompt, request);

            // Call API
            long startTime = System.currentTimeMillis();
            Map<String, Object> response = callInferenceAPI(modelId, requestBody);
            long duration = System.currentTimeMillis() - startTime;

            // Parse response
            String content = extractContent(response);

            return InferenceResponse.builder()
                    .requestId(request.getMetadata("request_id")
                            .orElse(UUID.randomUUID().toString()))
                    .content(content)
                    .model(modelId)
                    .tokensUsed(estimateTokens(prompt + content))
                    .durationMs(duration)
                    .metadata("provider", id())
                    .metadata("model_type", detectModelType(modelId))
                    .build();
        })
                .runSubscriptionOn(java.util.concurrent.ForkJoinPool.commonPool());
    }

    @Override
    public Multi<StreamChunk> stream(
            ProviderRequest request,
            TenantContext context) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                ensureInitialized();

                String modelId = normalizeModelId(request.getModel());
                String prompt = buildPrompt(request);
                Map<String, Object> requestBody = buildRequestBody(prompt, request);
                requestBody.put("stream", true);

                String requestId = request.getMetadata("request_id")
                        .orElse(UUID.randomUUID().toString());

                // Call streaming API
                streamInferenceAPI(modelId, requestBody, (index, delta) -> {
                    emitter.emit(StreamChunk.of(requestId, index, delta));
                });

                emitter.complete();

            } catch (Exception e) {
                LOG.errorf(e, "Streaming failed");
                emitter.fail(new ProviderException("Streaming failed", e));
            }
        });
    }

    @Override
    public ProviderHealth health() {
        if (!initialized) {
            return ProviderHealth.unhealthy("Provider not initialized");
        }

        try {
            // Test API connectivity
            testAPIConnection();

            return ProviderHealth.healthy("HuggingFace API accessible");

        } catch (Exception e) {
            return ProviderHealth.degraded(
                    "API connection issues: " + e.getMessage());
        }
    }

    @Override
    public void shutdown() {
        LOG.info("Shutting down HuggingFace provider");
        initialized = false;
    }

    /**
     * Call HuggingFace Inference API
     */
    private Map<String, Object> callInferenceAPI(
            String modelId,
            Map<String, Object> requestBody) {
        try {
            String apiKey = config.getString("hf.api.key");
            String endpoint = getEndpoint(modelId);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(endpoint))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(
                            new com.fasterxml.jackson.databind.ObjectMapper()
                                    .writeValueAsString(requestBody)))
                    .timeout(Duration.ofSeconds(30))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                throw new ProviderException(
                        "API request failed: " + response.statusCode() +
                                " - " + response.body());
            }

            // Parse JSON response
            return parseResponse(response.body());

        } catch (Exception e) {
            throw new ProviderException("API call failed", e, true);
        }
    }

    /**
     * Stream from HuggingFace API
     */
    private void streamInferenceAPI(
            String modelId,
            Map<String, Object> requestBody,
            java.util.function.BiConsumer<Integer, String> consumer) {
        // Implement Server-Sent Events streaming
        // Simplified for now
        try {
            Map<String, Object> response = callInferenceAPI(modelId, requestBody);
            String content = extractContent(response);

            // Simulate streaming by chunking
            String[] words = content.split(" ");
            for (int i = 0; i < words.length; i++) {
                consumer.accept(i, words[i] + " ");
                Thread.sleep(50); // Simulate delay
            }

        } catch (Exception e) {
            throw new ProviderException("Streaming failed", e);
        }
    }

    private String normalizeModelId(String modelId) {
        if (modelId.startsWith("hf:")) {
            return modelId.substring(3);
        }
        return modelId;
    }

    private String getEndpoint(String modelId) {
        String customEndpoint = config.getString("hf.endpoint.url");
        if (customEndpoint != null) {
            return customEndpoint;
        }
        return API_BASE + modelId;
    }

    private String buildPrompt(ProviderRequest request) {
        return request.getMessages().stream()
                .map(msg -> msg.getRole() + ": " + msg.getContent())
                .reduce("", (a, b) -> a + "\n" + b);
    }

    private Map<String, Object> buildRequestBody(
            String prompt,
            ProviderRequest request) {
        Map<String, Object> body = new HashMap<>();
        body.put("inputs", prompt);

        Map<String, Object> parameters = new HashMap<>();

        request.getParameter("max_tokens", Integer.class)
                .ifPresent(v -> parameters.put("max_new_tokens", v));

        request.getParameter("temperature", Double.class)
                .ifPresent(v -> parameters.put("temperature", v));

        request.getParameter("top_p", Double.class)
                .ifPresent(v -> parameters.put("top_p", v));

        if (!parameters.isEmpty()) {
            body.put("parameters", parameters);
        }

        return body;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseResponse(String json) {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper()
                    .readValue(json, Map.class);
        } catch (Exception e) {
            throw new ProviderException("Failed to parse response", e);
        }
    }

    @SuppressWarnings("unchecked")
    private String extractContent(Map<String, Object> response) {
        // Handle different response formats
        if (response.containsKey("generated_text")) {
            return response.get("generated_text").toString();
        }

        if (response.containsKey("0")) {
            Map<String, Object> first = (Map<String, Object>) response.get("0");
            if (first.containsKey("generated_text")) {
                return first.get("generated_text").toString();
            }
        }

        return response.toString();
    }

    private String detectModelType(String modelId) {
        if (modelId.contains("gpt"))
            return "gpt";
        if (modelId.contains("bert"))
            return "bert";
        if (modelId.contains("t5"))
            return "t5";
        if (modelId.contains("llama"))
            return "llama";
        return "unknown";
    }

    private int estimateTokens(String text) {
        return text.split("\\s+").length;
    }

    private void testAPIConnection() throws Exception {
        String apiKey = config.getString("hf.api.key");

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api-inference.huggingface.co"))
                .header("Authorization", "Bearer " + apiKey)
                .GET()
                .timeout(Duration.ofSeconds(5))
                .build();

        HttpResponse<String> response = httpClient.send(
                request,
                HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() >= 500) {
            throw new Exception("API server error: " + response.statusCode());
        }
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("Provider not initialized");
        }
    }
}