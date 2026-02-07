package tech.kayys.golek.provider.huggingface;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.provider.*;
import tech.kayys.golek.spi.exception.ProviderException;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;

/**
 * HuggingFace Inference API provider.
 */
@ApplicationScoped
public class HuggingFaceProvider implements StreamingProvider {

    private static final Logger LOG = Logger.getLogger(HuggingFaceProvider.class);
    private static final String API_BASE = "https://api-inference.huggingface.co/models/";
    private static final String PROVIDER_ID = "huggingface";
    private static final String PROVIDER_NAME = "HuggingFace";
    private static final String VERSION = "1.0.0";

    private HttpClient httpClient;
    private ProviderConfig config;
    private volatile boolean initialized = false;

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return PROVIDER_NAME;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(id())
                .name(name())
                .description("HuggingFace Inference API provider")
                .vendor("HuggingFace")
                .version(VERSION)
                .homepage("https://huggingface.co")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .embeddings(true)
                .maxContextTokens(4096)
                .maxOutputTokens(2048)
                .supportedLanguages(List.of("en"))
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
        if (initialized)
            return;

        synchronized (this) {
            if (initialized)
                return;

            LOG.info("Initializing HuggingFace provider");
            this.config = config;

            try {
                this.httpClient = HttpClient.newBuilder()
                        .connectTimeout(Duration.ofSeconds(10))
                        .build();

                String apiKey = config.getString("hf.api.key");
                if (apiKey == null || apiKey.isBlank()) {
                    apiKey = config.getString("hf_api_key");
                }

                if (apiKey == null || apiKey.isBlank()) {
                    throw new ProviderException.ProviderInitializationException(id(),
                            "HuggingFace API key not configured", null);
                }

                initialized = true;
                LOG.info("HuggingFace provider initialized");
            } catch (Exception e) {
                throw new ProviderException.ProviderInitializationException("Failed to initialize HuggingFace provider",
                        e);
            }
        }
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        return modelId.contains("/") || modelId.startsWith("hf:");
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();
            String modelId = normalizeModelId(request.getModel());
            Map<String, Object> requestBody = buildRequestBody(request);

            long startTime = System.currentTimeMillis();
            Map<String, Object> response = callInferenceAPI(modelId, requestBody);
            long duration = System.currentTimeMillis() - startTime;

            String content = extractContent(response);

            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .content(content)
                    .model(modelId)
                    .durationMs(duration)
                    .build();
        });
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request, TenantContext context) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                ensureInitialized();
                String modelId = normalizeModelId(request.getModel());
                Map<String, Object> requestBody = buildRequestBody(request);
                requestBody.put("stream", true);

                String requestId = request.getRequestId();

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
    public Uni<ProviderHealth> health() {
        if (!initialized) {
            return Uni.createFrom().item(ProviderHealth.unhealthy("Provider not initialized"));
        }
        return Uni.createFrom().item(ProviderHealth.healthy("HuggingFace API accessible"));
    }

    @Override
    public void shutdown() {
        initialized = false;
    }

    private Map<String, Object> callInferenceAPI(String modelId, Map<String, Object> requestBody) {
        try {
            String apiKey = config.getString("hf.api.key");
            if (apiKey == null || apiKey.isBlank()) {
                apiKey = config.getString("hf_api_key");
            }
            String endpoint = API_BASE + modelId;

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(endpoint))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(
                            new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(requestBody)))
                    .timeout(Duration.ofSeconds(30))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new ProviderException("API request failed: " + response.statusCode());
            }

            return new com.fasterxml.jackson.databind.ObjectMapper().readValue(response.body(), Map.class);
        } catch (Exception e) {
            throw new ProviderException("API call failed", e);
        }
    }

    private void streamInferenceAPI(String modelId, Map<String, Object> requestBody,
            java.util.function.BiConsumer<Integer, String> consumer) {
        try {
            Map<String, Object> response = callInferenceAPI(modelId, requestBody);
            String content = extractContent(response);
            String[] words = content.split(" ");
            for (int i = 0; i < words.length; i++) {
                consumer.accept(i, words[i] + " ");
                Thread.sleep(50);
            }
        } catch (Exception e) {
            throw new ProviderException("Streaming failed", e);
        }
    }

    private String normalizeModelId(String modelId) {
        return modelId.startsWith("hf:") ? modelId.substring(3) : modelId;
    }

    private Map<String, Object> buildRequestBody(ProviderRequest request) {
        Map<String, Object> body = new HashMap<>();
        String prompt = request.getMessages().get(request.getMessages().size() - 1).getContent();
        body.put("inputs", prompt);
        return body;
    }

    private String extractContent(Map<String, Object> response) {
        if (response.containsKey("generated_text"))
            return response.get("generated_text").toString();
        return response.toString();
    }

    private void ensureInitialized() {
        if (!initialized)
            throw new IllegalStateException("Provider not initialized");
    }
}