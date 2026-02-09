package tech.kayys.golek.provider.gemini;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.ProviderCapabilities;
import tech.kayys.golek.spi.provider.ProviderConfig;
import tech.kayys.golek.spi.provider.ProviderHealth;
import tech.kayys.golek.spi.provider.ProviderMetadata;
import tech.kayys.golek.spi.provider.ProviderRequest;
import tech.kayys.golek.spi.provider.StreamingProvider;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.wayang.tenant.TenantContext;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Google Gemini provider adapter for cloud LLM inference.
 * Supports Gemini Pro, Gemini Ultra, and Flash models.
 */
@ApplicationScoped
public class GeminiProvider implements StreamingProvider {

        private static final String PROVIDER_ID = "gemini";
        private static final String PROVIDER_NAME = "Google Gemini";
        private static final String VERSION = "1.0.0";

        private static final Logger log = Logger.getLogger(GeminiProvider.class);

        @Inject
        @RestClient
        GeminiClient client;

        private String apiKey;
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
                // Required parameters
                String key = config.getString("api.key");
                if (key == null || key.isBlank()) {
                        key = config.getRequiredSecret("api.key");
                }
                this.apiKey = key;
                log.info("Gemini provider initialized");
        }

        @Override
        public void shutdown() {
                log.info("Gemini provider shutting down");
        }

        @Override
        public Uni<ProviderHealth> health() {
                if (apiKey == null || apiKey.isBlank()) {
                        return Uni.createFrom().item(ProviderHealth.unhealthy("API key not configured"));
                }
                return Uni.createFrom().item(ProviderHealth.healthy(id()));
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
                                .build();
        }

        @Override
        public boolean supports(String model, TenantContext context) {
                return model.startsWith("gemini");
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
                long startTime = System.currentTimeMillis();
                int requestId = requestCounter.incrementAndGet();

                log.debugf("[%d] Gemini inference: model=%s", requestId, request.getModel());

                GeminiRequest geminiRequest = buildGeminiRequest(request);

                return client.generateContent(request.getModel(), apiKey, geminiRequest)
                                .map(response -> {
                                        long duration = System.currentTimeMillis() - startTime;
                                        String content = extractContent(response);

                                        return InferenceResponse.builder()
                                                        .requestId(request.getRequestId() != null
                                                                        ? request.getRequestId()
                                                                        : String.valueOf(requestId))
                                                        .content(content)
                                                        .model(request.getModel())
                                                        .durationMs(duration)
                                                        .tokensUsed(response.getUsageMetadata() != null ? response
                                                                        .getUsageMetadata().getTotalTokenCount() : 0)
                                                        .metadata("total_tokens",
                                                                        response.getUsageMetadata() != null ? response
                                                                                        .getUsageMetadata()
                                                                                        .getTotalTokenCount() : 0)
                                                        .build();
                                })
                                .onFailure().transform(this::wrapException);
        }

        @Override
        public Multi<StreamChunk> inferStream(ProviderRequest request, TenantContext context) {
                AtomicInteger chunkIndex = new AtomicInteger(0);
                String reqId = request.getRequestId() != null ? request.getRequestId()
                                : String.valueOf(requestCounter.incrementAndGet());

                GeminiRequest geminiRequest = buildGeminiRequest(request);

                return client.streamGenerateContent(request.getModel(), apiKey, geminiRequest)
                                .map(chunk -> {
                                        int index = chunkIndex.getAndIncrement();
                                        String content = extractContent(chunk);
                                        return StreamChunk.of(reqId, index, content);
                                })
                                .onFailure().transform(this::wrapException);
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
                if (response.getCandidates() == null || response.getCandidates().isEmpty()) {
                        return "";
                }
                GeminiCandidate candidate = response.getCandidates().get(0);
                if (candidate.getContent() == null || candidate.getContent().getParts() == null) {
                        return "";
                }
                return candidate.getContent().getParts().stream()
                                .map(GeminiPart::getText)
                                .filter(java.util.Objects::nonNull)
                                .collect(Collectors.joining());
        }

        private Throwable wrapException(Throwable ex) {
                return new RuntimeException("Gemini request failed: " + ex.getMessage(), ex);
        }
}