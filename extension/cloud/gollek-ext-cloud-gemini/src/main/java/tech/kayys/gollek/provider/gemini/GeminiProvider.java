package tech.kayys.gollek.provider.gemini;

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
 * Google Gemini provider adapter for cloud LLM inference.
 */
@ApplicationScoped
public class GeminiProvider implements StreamingProvider {

        private static final String PROVIDER_ID = "gemini";
        private static final String PROVIDER_NAME = "Google Gemini";
        private static final String VERSION = "1.0.0";
        private static final Logger log = Logger.getLogger(GeminiProvider.class);

        @Inject
        com.fasterxml.jackson.databind.ObjectMapper objectMapper;

        @Inject
        GeminiConfig configDetails;

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
                log.info("Gemini provider initialized");
        }

        @Override
        public void shutdown() {
                log.info("Gemini provider shutting down");
        }

        @Override
        public Uni<ProviderHealth> health() {
                return Uni.createFrom().item(() -> {
                        String currentApiKey = getApiKey(null);
                        return ProviderHealth.healthy(
                                        currentApiKey != null && !currentApiKey.isBlank() ? "Gemini API available"
                                                        : "Gemini initialized (API key missing)");
                });
        }

        @Override
        public ProviderCapabilities capabilities() {
                return ProviderCapabilities.builder()
                                .streaming(true)
                                .functionCalling(true)
                                .multimodal(true)
                                .maxContextTokens(1000000)
                                .build();
        }

        @Override
        public ProviderMetadata metadata() {
                return ProviderMetadata.builder()
                                .providerId(PROVIDER_ID)
                                .name(PROVIDER_NAME)
                                .description("Google Gemini - multimodal AI with large context window")
                                .version(VERSION)
                                .vendor("Google")
                                .homepage("https://ai.google.dev/docs")
                                .defaultModel("gemini-1.5-flash")
                                .build();
        }

        @Override
        public boolean supports(String model, ProviderRequest request) {
                return model != null && model.startsWith("gemini");
        }

        private String getApiKey(ProviderRequest request) {
                if (request != null && request.getApiKey().isPresent()) {
                        return request.getApiKey().get();
                }
                if (configDetails == null)
                        return null;
                String key = configDetails.apiKey();
                return (key != null && key.equals("dummy")) ? null : key;
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request) {
                long startTime = System.currentTimeMillis();
                int requestId = requestCounter.incrementAndGet();

                GeminiRequest geminiRequest = buildGeminiRequest(request);
                String currentApiKey = getApiKey(request);

                if (currentApiKey == null || currentApiKey.isBlank()) {
                        return Uni.createFrom()
                                        .failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                                                        "Gemini API key not configured."));
                }

                String model = request.getModel() != null ? request.getModel().trim() : "gemini-1.5-flash";
                String url = String.format(
                                "https://generativelanguage.googleapis.com/v1/models/%s:generateContent?key=%s",
                                model, currentApiKey);

                try {
                        String body = objectMapper.writeValueAsString(geminiRequest);
                        HttpRequest httpRequest = HttpRequest.newBuilder()
                                        .uri(URI.create(url))
                                        .header("Content-Type", "application/json")
                                        .POST(HttpRequest.BodyPublishers.ofString(body))
                                        .build();

                        return Uni.createFrom()
                                        .completionStage(httpClient.sendAsync(httpRequest,
                                                        HttpResponse.BodyHandlers.ofString()))
                                        .map(resp -> {
                                                if (resp.statusCode() != 200) {
                                                        throw new RuntimeException("Gemini failed: " + resp.statusCode()
                                                                        + " " + resp.body());
                                                }
                                                try {
                                                        GeminiResponse response = objectMapper.readValue(resp.body(),
                                                                        GeminiResponse.class);
                                                        long duration = System.currentTimeMillis() - startTime;
                                                        return InferenceResponse.builder()
                                                                        .requestId(request.getRequestId() != null
                                                                                        ? request.getRequestId()
                                                                                        : String.valueOf(requestId))
                                                                        .content(extractContent(response))
                                                                        .model(model)
                                                                        .durationMs(duration)
                                                                        .metadata("provider", PROVIDER_ID)
                                                                        .build();
                                                } catch (Exception e) {
                                                        throw new RuntimeException("Gemini deserialization failed", e);
                                                }
                                        });
                } catch (Exception e) {
                        return Uni.createFrom().failure(e);
                }
        }

        @Override
        public Multi<StreamChunk> inferStream(ProviderRequest request) {
                AtomicInteger chunkIndex = new AtomicInteger(0);
                GeminiRequest geminiRequest = buildGeminiRequest(request);
                String currentApiKey = getApiKey(request);

                if (currentApiKey == null || currentApiKey.isBlank()) {
                        return Multi.createFrom()
                                        .failure(new ProviderException.ProviderAuthenticationException(PROVIDER_ID,
                                                        "Gemini API key not configured."));
                }

                String model = request.getModel() != null ? request.getModel().trim() : "gemini-1.5-flash";
                String url = String.format(
                                "https://generativelanguage.googleapis.com/v1/models/%s:streamGenerateContent?alt=sse&key=%s",
                                model, currentApiKey);

                log.debug("Gemini streaming URL: " + url.replaceAll("key=.*", "key=REDACTED"));

                try {
                        String body = objectMapper.writeValueAsString(geminiRequest);
                        HttpRequest httpRequest = HttpRequest.newBuilder()
                                        .uri(URI.create(url))
                                        .header("Content-Type", "application/json")
                                        .header("Accept", "text/event-stream")
                                        .POST(HttpRequest.BodyPublishers.ofString(body))
                                        .build();

                        return Multi.createFrom().emitter(emitter -> {
                                httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofInputStream())
                                                .thenAccept(resp -> {
                                                        if (resp.statusCode() != 200) {
                                                                emitter.fail(new RuntimeException(
                                                                                "Gemini streaming failed: "
                                                                                                + resp.statusCode()));
                                                                return;
                                                        }
                                                        try (BufferedReader reader = new BufferedReader(
                                                                        new InputStreamReader(resp.body()))) {
                                                                String line;
                                                                while ((line = reader.readLine()) != null) {
                                                                        if (line.startsWith("data:")) {
                                                                                String data = line.startsWith("data: ")
                                                                                                ? line.substring(6)
                                                                                                                .trim()
                                                                                                : line.substring(5)
                                                                                                                .trim();
                                                                                if (!data.isEmpty()) {
                                                                                        try {
                                                                                                GeminiResponse chunk = objectMapper
                                                                                                                .readValue(data, GeminiResponse.class);
                                                                                                String content = extractContent(
                                                                                                                chunk);
                                                                                                int index = chunkIndex
                                                                                                                .getAndIncrement();

                                                                                                boolean isFinal = false;
                                                                                                if (chunk.getCandidates() != null
                                                                                                                && !chunk.getCandidates()
                                                                                                                                .isEmpty()) {
                                                                                                        String reason = chunk
                                                                                                                        .getCandidates()
                                                                                                                        .get(0)
                                                                                                                        .getFinishReason();
                                                                                                        isFinal = reason != null
                                                                                                                        && !reason.isEmpty();
                                                                                                }

                                                                                                if (isFinal) {
                                                                                                        emitter.emit(StreamChunk
                                                                                                                        .finalChunk(request
                                                                                                                                        .getRequestId(),
                                                                                                                                        index,
                                                                                                                                        content));
                                                                                                } else {
                                                                                                        emitter.emit(StreamChunk
                                                                                                                        .of(request.getRequestId(),
                                                                                                                                        index,
                                                                                                                                        content));
                                                                                                }
                                                                                        } catch (Exception e) {
                                                                                                log.warn("Failed to parse Gemini chunk: "
                                                                                                                + data,
                                                                                                                e);
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

        private GeminiRequest buildGeminiRequest(ProviderRequest request) {
                GeminiRequest geminiRequest = new GeminiRequest();
                if (request.getMessages() != null) {
                        List<GeminiContent> contents = request.getMessages().stream()
                                        .map(msg -> {
                                                GeminiContent content = new GeminiContent();
                                                content.setRole(mapRole(msg.getRole().toString()));
                                                content.setParts(List.of(new GeminiPart(msg.getContent())));
                                                return content;
                                        })
                                        .collect(Collectors.toList());
                        geminiRequest.setContents(contents);
                }
                GeminiGenerationConfig genConfig = new GeminiGenerationConfig();
                if (request.getParameters() != null) {
                        if (request.getParameters().containsKey("temperature")) {
                                genConfig.setTemperature(
                                                ((Number) request.getParameters().get("temperature")).doubleValue());
                        }
                        if (request.getParameters().containsKey("max_tokens")) {
                                genConfig.setMaxOutputTokens(
                                                ((Number) request.getParameters().get("max_tokens")).intValue());
                        }
                }
                geminiRequest.setGenerationConfig(genConfig);
                return geminiRequest;
        }

        private String mapRole(String role) {
                return switch (role.toLowerCase()) {
                        case "system", "user" -> "user";
                        case "assistant" -> "model";
                        default -> "user";
                };
        }

        private String extractContent(GeminiResponse response) {
                if (response.getCandidates() == null || response.getCandidates().isEmpty())
                        return "";
                GeminiCandidate candidate = response.getCandidates().get(0);
                if (candidate.getContent() == null || candidate.getContent().getParts() == null)
                        return "";
                return candidate.getContent().getParts().stream()
                                .map(GeminiPart::getText)
                                .filter(java.util.Objects::nonNull)
                                .collect(Collectors.joining());
        }
}