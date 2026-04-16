package tech.kayys.gollek.provider.anthropic;

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
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.StreamingProvider;

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
 * Anthropic provider adapter for cloud LLM inference.
 * Supports Claude family of models.
 */
@ApplicationScoped
public class AnthropicProvider implements StreamingProvider {

    private static final String PROVIDER_ID = "anthropic";
    private static final String PROVIDER_NAME = "Anthropic";
    private static final String VERSION = "1.0.0";
    private static final String API_VERSION = "2023-06-01";
    private static final Logger log = Logger.getLogger(AnthropicProvider.class);

    @Inject
    com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    @Inject
    AnthropicConfig configDetails;

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
    public boolean isEnabled() {
        return configDetails != null && configDetails.enabled();
    }

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
        log.info("Anthropic provider initialized");
    }

    @Override
    public void shutdown() {
        log.info("Anthropic provider shutting down");
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            String currentApiKey = getApiKey(null);
            return ProviderHealth.healthy(
                    currentApiKey != null && !currentApiKey.isBlank() ? "Anthropic API available"
                            : "Anthropic initialized (API key missing)");
        });
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(true)
                .multimodal(true)
                .maxContextTokens(200000)
                .build();
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(PROVIDER_NAME)
                .description("Anthropic - Claude family of models")
                .version(VERSION)
                .vendor("Anthropic")
                .homepage("https://anthropic.com")
                .defaultModel("claude-3-sonnet-20240229")
                .build();
    }

    @Override
    public boolean supports(String model, ProviderRequest request) {
        if (model == null)
            return false;
        String lower = model.toLowerCase();
        return lower.startsWith("claude") || lower.contains("anthropic");
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        long startTime = System.currentTimeMillis();
        int requestId = requestCounter.incrementAndGet();

        AnthropicRequest anthropicRequest = buildAnthropicRequest(request);
        String currentApiKey = getApiKey(request);

        if (currentApiKey == null || currentApiKey.isBlank()) {
            return Uni.createFrom().failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                    "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."));
        }

        String baseUrl = configDetails.baseUrl();
        if (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
        }
        String url = baseUrl + "/v1/messages";

        try {
            String body = objectMapper.writeValueAsString(anthropicRequest);
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .header("x-api-key", currentApiKey)
                    .header("anthropic-version", API_VERSION)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            return Uni.createFrom()
                    .completionStage(httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString()))
                    .map(resp -> {
                        if (resp.statusCode() != 200) {
                            throw new RuntimeException("Anthropic failed: " + resp.statusCode() + " " + resp.body());
                        }
                        try {
                            com.fasterxml.jackson.databind.JsonNode root = objectMapper.readTree(resp.body());
                            AnthropicResponse response = objectMapper.treeToValue(root, AnthropicResponse.class);
                            long duration = System.currentTimeMillis() - startTime;
                            return InferenceResponse.builder()
                                    .requestId(request.getRequestId() != null ? request.getRequestId()
                                            : String.valueOf(requestId))
                                    .content(extractContent(root))
                                    .model(response.getModel() != null ? response.getModel() : request.getModel())
                                    .inputTokens(
                                            response.getUsage() != null ? response.getUsage().getInputTokens() : 0)
                                    .outputTokens(
                                            response.getUsage() != null ? response.getUsage().getOutputTokens() : 0)
                                    .durationMs(duration)
                                    .metadata("provider", PROVIDER_ID)
                                    .build();
                        } catch (Exception e) {
                            throw new RuntimeException("Anthropic deserialization failed", e);
                        }
                    });
        } catch (Exception e) {
            return Uni.createFrom().failure(e);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        AtomicInteger chunkIndex = new AtomicInteger(0);
        AnthropicRequest anthropicRequest = buildAnthropicRequest(request);
        anthropicRequest.setStream(true);
        String currentApiKey = getApiKey(request);

        if (currentApiKey == null || currentApiKey.isBlank()) {
            return Multi.createFrom().failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                    "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."));
        }

        String baseUrl = configDetails.baseUrl();
        if (baseUrl.endsWith("/")) {
            baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
        }
        String url = baseUrl + "/v1/messages";

        log.debug("Anthropic streaming URL: " + url);

        try {
            String body = objectMapper.writeValueAsString(anthropicRequest);
            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("Content-Type", "application/json")
                    .header("x-api-key", currentApiKey)
                    .header("anthropic-version", API_VERSION)
                    .header("Accept", "text/event-stream")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            return Multi.createFrom().emitter(emitter -> {
                httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofInputStream())
                        .thenAccept(resp -> {
                            if (resp.statusCode() != 200) {
                                emitter.fail(new RuntimeException("Anthropic streaming failed: " + resp.statusCode()));
                                return;
                            }
                            try (BufferedReader reader = new BufferedReader(new InputStreamReader(resp.body()))) {
                                String line;
                                while ((line = reader.readLine()) != null) {
                                    if (line.startsWith("data: ")) {
                                        String data = line.substring(6).trim();
                                        if (!data.isEmpty()) {
                                            try {
                                                com.fasterxml.jackson.databind.JsonNode root = objectMapper
                                                        .readTree(data);
                                                String eventType = root.path("type").asText("");
                                                if ("content_block_delta".equals(eventType)) {
                                                    String content = root.path("delta").path("text").asText("");
                                                    if (content != null && !content.isBlank()) {
                                                        int index = chunkIndex.getAndIncrement();
                                                        emitter.emit(
                                                                StreamingInferenceChunk.of(request.getRequestId(), index, content));
                                                    }
                                                }
                                            } catch (Exception e) {
                                                log.warn("Failed to parse Anthropic chunk: " + data, e);
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

    private AnthropicRequest buildAnthropicRequest(ProviderRequest request) {
        AnthropicRequest anthropicRequest = new AnthropicRequest();
        String model = request.getModel() != null ? request.getModel().trim() : "claude-3-sonnet-20240229";
        anthropicRequest.setModel(model);

        if (request.getMessages() != null) {
            // Convert messages to Anthropic format
            List<AnthropicMessage> messages = request.getMessages().stream()
                    .map(msg -> new AnthropicMessage(msg.getRole().toString().toLowerCase(), msg.getContent()))
                    .collect(Collectors.toList());
            anthropicRequest.setMessages(messages);
        }

        if (request.getParameters() != null) {
            if (request.getParameters().containsKey("max_tokens")) {
                anthropicRequest.setMaxTokens(((Number) request.getParameters().get("max_tokens")).intValue());
            } else {
                // Anthropic requires max_tokens
                anthropicRequest.setMaxTokens(4096);
            }
            if (request.getParameters().containsKey("temperature")) {
                anthropicRequest.setTemperature(((Number) request.getParameters().get("temperature")).doubleValue());
            }
            if (request.getParameters().containsKey("top_p")) {
                anthropicRequest.setTopP(((Number) request.getParameters().get("top_p")).doubleValue());
            }
            if (request.getParameters().containsKey("top_k")) {
                anthropicRequest.setTopK(((Number) request.getParameters().get("top_k")).intValue());
            }
        } else {
            // Default max_tokens for Anthropic
            anthropicRequest.setMaxTokens(4096);
        }

        return anthropicRequest;
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
        return System.getenv("ANTHROPIC_API_KEY");
    }

    private String extractContent(com.fasterxml.jackson.databind.JsonNode root) {
        if (root == null) {
            return "";
        }
        com.fasterxml.jackson.databind.JsonNode content = root.get("content");
        if (content == null || !content.isArray() || content.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (com.fasterxml.jackson.databind.JsonNode block : content) {
            if ("text".equals(block.path("type").asText())) {
                sb.append(block.path("text").asText(""));
            }
        }
        return sb.toString();
    }
}
