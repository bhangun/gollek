package tech.kayys.golek.provider.gemini;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.provider.ProviderCapabilities;
import tech.kayys.golek.api.provider.ProviderConfig;
import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.ProviderMetadata;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.provider.core.adapter.CloudProviderAdapter;
import tech.kayys.golek.provider.core.spi.StreamingProvider;
import tech.kayys.wayang.tenant.TenantContext;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Google Gemini provider adapter for cloud LLM inference.
 * Supports Gemini Pro, Gemini Ultra, and Flash models.
 */
@ApplicationScoped
public class GeminiProvider extends CloudProviderAdapter implements StreamingProvider {

        private static final String PROVIDER_ID = "gemini";
        private static final String PROVIDER_NAME = "Google Gemini";
        private static final String VERSION = "1.0.0";

        @Inject
        @RestClient
        GeminiClient client;

        @Inject
        GeminiConfig config;

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
        public String version() {
                return VERSION;
        }

        @Override
        protected String getDefaultBaseUrl() {
                return "https://generativelanguage.googleapis.com";
        }

        @Override
        protected String getApiKeyEnvironmentVariable() {
                return "GEMINI_API_KEY";
        }

        @Override
        protected Uni<Boolean> performHealthCheckRequest() {
                return client.listModels(getApiKey())
                                .map(models -> true)
                                .onFailure().recoverWithItem(false);
        }

        @Override
        public ProviderCapabilities capabilities() {
                return ProviderCapabilities.builder()
                                .streaming(true)
                                .functionCalling(true)
                                .multimodal(true)
                                .embeddings(true)
                                .maxContextTokens(1000000) // Gemini 1.5 Pro
                                .maxOutputTokens(8192)
                                .supportedModels(Set.of(
                                                "gemini-2.0-flash-exp",
                                                "gemini-1.5-pro",
                                                "gemini-1.5-flash",
                                                "gemini-1.5-flash-8b",
                                                "gemini-1.0-pro",
                                                "gemini-pro-vision",
                                                "text-embedding-004"))
                                .supportedLanguages(List.of("en", "zh", "es", "fr", "de", "ja", "ko", "pt", "ru", "ar"))
                                .toolCalling(true)
                                .structuredOutputs(true)
                                .build();
        }

        @Override
        public ProviderMetadata metadata() {
                return ProviderMetadata.builder()
                                .providerId(PROVIDER_ID)
                                .displayName(PROVIDER_NAME)
                                .description("Google Gemini - multimodal AI with 1M token context window")
                                .version(VERSION)
                                .vendor("Google")
                                .documentationUrl("https://ai.google.dev/docs")
                                .metadata("deployment", "cloud")
                                .metadata("requires_api_key", "true")
                                .metadata("pricing_url", "https://ai.google.dev/pricing")
                                .build();
        }

        @Override
        public boolean supports(String model, TenantContext context) {
                return capabilities().getSupportedModels().contains(model) ||
                                model.startsWith("gemini");
        }

        @Override
        protected Uni<InferenceResponse> doInfer(ProviderRequest request) {
                trackRequest();
                long startTime = System.currentTimeMillis();
                int requestId = requestCounter.incrementAndGet();

                log.debugf("[%d] Gemini inference: model=%s", requestId, request.getModel());

                GeminiRequest geminiRequest = buildGeminiRequest(request);
                String model = request.getModel();

                return client.generateContent(model, getApiKey(), geminiRequest)
                                .map(response -> {
                                        long duration = System.currentTimeMillis() - startTime;
                                        log.debugf("[%d] Gemini response in %dms", requestId, duration);

                                        String content = extractContent(response);

                                        return InferenceResponse.builder()
                                                        .requestId(request.getRequestId())
                                                        .content(content)
                                                        .model(model)
                                                        .durationMs(duration)
                                                        .metadata("provider", PROVIDER_ID)
                                                        .metadata("prompt_tokens",
                                                                        response.getUsageMetadata() != null ? response
                                                                                        .getUsageMetadata()
                                                                                        .getPromptTokenCount() : 0)
                                                        .metadata("completion_tokens",
                                                                        response.getUsageMetadata() != null ? response
                                                                                        .getUsageMetadata()
                                                                                        .getCandidatesTokenCount() : 0)
                                                        .metadata("total_tokens",
                                                                        response.getUsageMetadata() != null ? response
                                                                                        .getUsageMetadata()
                                                                                        .getTotalTokenCount() : 0)
                                                        .metadata("finish_reason",
                                                                        (response.getCandidates() != null && !response
                                                                                        .getCandidates().isEmpty())
                                                                                                        ? response.getCandidates()
                                                                                                                        .get(0)
                                                                                                                        .getFinishReason()
                                                                                                        : "unknown")
                                                        .build();
                                })
                                .onFailure().invoke(this::trackError)
                                .onFailure().transform(this::wrapException);
        }

        @Override
        public Multi<StreamChunk> stream(ProviderRequest request, TenantContext context) {
                // trackRequest();
                AtomicInteger chunkIndex = new AtomicInteger(0);

                GeminiRequest geminiRequest = buildGeminiRequest(request);
                String model = request.getModel();

                return client.streamGenerateContent(model, getApiKey(), geminiRequest)
                                .map(chunk -> {
                                        int index = chunkIndex.getAndIncrement();
                                        String content = extractContent(chunk);
                                        boolean isFinal = chunk.getCandidates() != null &&
                                                        !chunk.getCandidates().isEmpty() &&
                                                        "STOP".equals(chunk.getCandidates().get(0).getFinishReason());

                                        return StreamChunk.builder()
                                                        .index(index)
                                                        .delta(content)
                                                        .model(model)
                                                        .isFinal(isFinal)
                                                        .build();
                                })
                                .onFailure().invoke(this::trackError)
                                .onFailure().transform(this::wrapException);
        }

        private GeminiRequest buildGeminiRequest(ProviderRequest request) {
                GeminiRequest geminiRequest = new GeminiRequest();

                // Convert messages to Gemini format
                if (request.getMessages() != null) {
                        List<GeminiContent> contents = request.getMessages().stream()
                                        .map(msg -> {
                                                GeminiContent content = new GeminiContent();
                                                content.setRole(mapRole(msg.getRole()));
                                                content.setParts(List.of(new GeminiPart(msg.getContent())));
                                                return content;
                                        })
                                        .collect(Collectors.toList());
                        geminiRequest.setContents(contents);
                }

                // Set generation config
                GeminiGenerationConfig genConfig = new GeminiGenerationConfig();
                if (request.getParameters().containsKey("temperature")) {
                        genConfig.setTemperature(((Number) request.getParameters().get("temperature")).doubleValue());
                }
                if (request.getParameters().containsKey("max_tokens")) {
                        genConfig.setMaxOutputTokens(((Number) request.getParameters().get("max_tokens")).intValue());
                }
                if (request.getParameters().containsKey("top_p")) {
                        genConfig.setTopP(((Number) request.getParameters().get("top_p")).doubleValue());
                }
                if (request.getParameters().containsKey("top_k")) {
                        genConfig.setTopK(((Number) request.getParameters().get("top_k")).intValue());
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