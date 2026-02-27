package tech.kayys.gollek.provider.mistral;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.ProxySelector;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Mistral provider adapter for cloud LLM inference.
 */
@ApplicationScoped
public class MistralProvider implements StreamingProvider {

    private static final String PROVIDER_ID = "mistral";
    private static final String PROVIDER_NAME = "Mistral AI";
    private static final String VERSION = "1.0.0";
    private static final Logger log = Logger.getLogger(MistralProvider.class);

    @Inject
    com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    @Inject
    MistralConfig configDetails;

    private HttpClient httpClient;

    @jakarta.annotation.PostConstruct
    void init() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .proxy(ProxySelector.getDefault())
                .build();
    }

    private final AtomicInteger requestCounter = new AtomicInteger(0);

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return PROVIDER_NAME;
    }

    @Override
    public void initialize(ProviderConfig config) {
        log.info("Mistral provider initialized");
    }

    @Override
    public void shutdown() {
        log.info("Mistral provider shutting down");
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            String currentApiKey = getApiKey(null);
            return ProviderHealth.healthy(
                    currentApiKey != null && !currentApiKey.isBlank() ? "Mistral API available"
                            : "Mistral initialized (API key missing)");
        });
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(true)
                .multimodal(false)
                .embeddings(true)
                .maxContextTokens(32768)
                .build();
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(PROVIDER_NAME)
                .description("Mistral AI - high-performance open models")
                .version(VERSION)
                .vendor("Mistral AI")
                .homepage("https://mistral.ai")
                .defaultModel("mistral-small-latest")
                .build();
    }

    @Override
    public boolean supports(String model, ProviderRequest request) {
        return model != null
                && (model.startsWith("mistral") || model.startsWith("pixtral") || model.startsWith("codestral"));
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        long startTime = System.currentTimeMillis();
        int requestId = requestCounter.incrementAndGet();

        MistralRequest mistralRequest = buildMistralRequest(request);
        String currentApiKey = getApiKey(request);

        if (currentApiKey == null || currentApiKey.isBlank()) {
            return Uni.createFrom().failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                    "Mistral API key not configured. Set MISTRAL_API_KEY environment variable."));
        }

        String baseUrl = configDetails.baseUrl();
        if (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
        }
        // Ensure /v1 is present if not already in baseUrl
        String url = baseUrl.contains("/v1") ? baseUrl + "/chat/completions" : baseUrl + "/v1/chat/completions";

        try {
            String body = objectMapper.writeValueAsString(mistralRequest);
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + currentApiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            return Uni.createFrom()
                    .completionStage(httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString()))
                    .map(resp -> {
                        if (resp.statusCode() != 200) {
                            throw new RuntimeException("Mistral failed: " + resp.statusCode() + " " + resp.body());
                        }
                        try {
                            MistralResponse response = objectMapper.readValue(resp.body(), MistralResponse.class);
                            long duration = System.currentTimeMillis() - startTime;
                            return InferenceResponse.builder()
                                    .requestId(request.getRequestId() != null ? request.getRequestId()
                                            : String.valueOf(requestId))
                                    .content(extractContent(response))
                                    .model(request.getModel())
                                    .durationMs(duration)
                                    .metadata("provider", PROVIDER_ID)
                                    .build();
                        } catch (Exception e) {
                            throw new RuntimeException("Mistral deserialization failed", e);
                        }
                    });
        } catch (Exception e) {
            return Uni.createFrom().failure(e);
        }
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request) {
        AtomicInteger chunkIndex = new AtomicInteger(0);
        MistralRequest mistralRequest = buildMistralRequest(request);
        mistralRequest.setStream(true);
        String currentApiKey = getApiKey(request);

        if (currentApiKey == null || currentApiKey.isBlank()) {
            return Multi.createFrom().failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                    "Mistral API key not configured. Set MISTRAL_API_KEY environment variable."));
        }

        String baseUrl = configDetails.baseUrl();
        if (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
        }
        String url = baseUrl.contains("/v1") ? baseUrl + "/chat/completions" : baseUrl + "/v1/chat/completions";

        try {
            String body = objectMapper.writeValueAsString(mistralRequest);
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + currentApiKey)
                    .header("Accept", "text/event-stream")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            return Multi.createFrom().emitter(emitter -> {
                httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofInputStream())
                        .thenAccept(resp -> {
                            if (resp.statusCode() != 200) {
                                emitter.fail(new RuntimeException("Mistral streaming failed: " + resp.statusCode()));
                                return;
                            }
                            try (BufferedReader reader = new BufferedReader(new InputStreamReader(resp.body()))) {
                                String line;
                                while ((line = reader.readLine()) != null) {
                                    if (line.startsWith("data: ")) {
                                        String data = line.substring(6).trim();
                                        if (!data.isEmpty() && !"[DONE]".equals(data)) {
                                            try {
                                                MistralStreamResponse chunk = objectMapper.readValue(data,
                                                        MistralStreamResponse.class);
                                                String content = extractContent(chunk);
                                                int index = chunkIndex.getAndIncrement();
                                                emitter.emit(StreamChunk.of(request.getRequestId(), index, content));
                                            } catch (Exception e) {
                                                log.warn("Failed to parse Mistral chunk: " + data, e);
                                            }
                                        }
                                    }
                                }
                                emitter.complete();
                            } catch (Exception e) {
                                emitter.fail(e);
                            }
                        })
                        .exceptionally(t -> {
                            emitter.fail(t);
                            return null;
                        });
            });
        } catch (Exception e) {
            return Multi.createFrom().failure(e);
        }
    }

    private MistralRequest buildMistralRequest(ProviderRequest request) {
        MistralRequest mistralRequest = new MistralRequest();
        String model = request.getModel() != null ? request.getModel().trim() : "mistral-small-latest";
        mistralRequest.setModel(model);

        if (request.getMessages() != null) {
            List<MistralMessage> messages = request.getMessages().stream()
                    .map(msg -> new MistralMessage(msg.getRole().toString().toLowerCase(), msg.getContent()))
                    .collect(Collectors.toList());
            mistralRequest.setMessages(messages);
        }

        if (request.getParameters() != null) {
            if (request.getParameters().containsKey("temperature")) {
                mistralRequest.setTemperature(((Number) request.getParameters().get("temperature")).doubleValue());
            }
            if (request.getParameters().containsKey("max_tokens")) {
                mistralRequest.setMaxTokens(((Number) request.getParameters().get("max_tokens")).intValue());
            }
            if (request.getParameters().containsKey("top_p")) {
                mistralRequest.setTopP(((Number) request.getParameters().get("top_p")).doubleValue());
            }
        }

        return mistralRequest;
    }

    private String getApiKey(ProviderRequest request) {
        if (request != null && request.getApiKey().isPresent()) {
            return request.getApiKey().get();
        }
        String key = configDetails.apiKey();
        if (key != null && !key.isBlank() && !"dummy".equals(key)) {
            return key;
        }
        // Fallback to standard environment variable
        return System.getenv("MISTRAL_API_KEY");
    }

    private String extractContent(MistralResponse response) {
        if (response.getChoices() != null && !response.getChoices().isEmpty()) {
            return response.getChoices().get(0).getMessage().getContent();
        }
        return "";
    }

    private String extractContent(MistralStreamResponse chunk) {
        if (chunk.getChoices() != null && !chunk.getChoices().isEmpty()) {
            return chunk.getChoices().get(0).getDelta().getContent();
        }
        return "";
    }
}
