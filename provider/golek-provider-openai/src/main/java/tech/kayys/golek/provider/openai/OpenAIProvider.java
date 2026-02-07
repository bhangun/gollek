package tech.kayys.golek.provider.openai;

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
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * OpenAI provider implementation for cloud LLM inference.
 */
@ApplicationScoped
public class OpenAIProvider implements StreamingProvider {

        private static final Logger log = Logger.getLogger(OpenAIProvider.class);
        private static final String PROVIDER_ID = "openai";
        private static final String PROVIDER_NAME = "OpenAI";
        private static final String VERSION = "1.0.0";

        @Inject
        @RestClient
        OpenAIClient client;

        @Inject
        OpenAIConfig config;

        private ProviderConfig providerConfig;
        private final AtomicInteger requestCounter = new AtomicInteger(0);

        @Override
        public void initialize(ProviderConfig config) {
                this.providerConfig = config;
                log.info("OpenAI provider initialized");
        }

        @Override
        public void shutdown() {
                log.info("OpenAI provider shutting down");
        }

        @Override
        public Uni<ProviderHealth> health() {
                String apiKey = getApiKey();
                if (apiKey == null || apiKey.isEmpty()) {
                        return Uni.createFrom().item(ProviderHealth.unhealthy("API key is missing"));
                }
                return client.listModels("Bearer " + apiKey)
                                .map(models -> ProviderHealth.healthy("OpenAI is reachable"))
                                .onFailure().recoverWithItem(
                                                t -> ProviderHealth.unhealthy("OpenAI unreachable: " + t.getMessage()));
        }

        private String getApiKey() {
                if (providerConfig == null)
                        return null;
                return providerConfig.getSecret("api.key")
                                .orElseGet(() -> providerConfig.getString("api_key"));
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
        public String version() {
                return VERSION;
        }

        @Override
        public ProviderCapabilities capabilities() {
                return ProviderCapabilities.builder()
                                .streaming(true)
                                .functionCalling(true)
                                .multimodal(true)
                                .embeddings(true)
                                .maxContextTokens(128000)
                                .maxOutputTokens(4096)
                                .supportedModels(Set.of(
                                                "gpt-4o",
                                                "gpt-4o-mini",
                                                "gpt-4-turbo",
                                                "gpt-3.5-turbo"))
                                .supportedLanguages(List.of("en", "zh", "es", "fr", "de", "ja", "ko"))
                                .build();
        }

        @Override
        public ProviderMetadata metadata() {
                return ProviderMetadata.builder()
                                .providerId(PROVIDER_ID)
                                .name(PROVIDER_NAME)
                                .description("OpenAI GPT models - industry standard for LLM inference")
                                .version(VERSION)
                                .vendor("OpenAI")
                                .homepage("https://platform.openai.com/docs")
                                .build();
        }

        @Override
        public boolean supports(String model, TenantContext context) {
                return capabilities().getSupportedModels().contains(model) ||
                                model.startsWith("gpt-") ||
                                model.startsWith("text-embedding");
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
                long startTime = System.currentTimeMillis();
                int requestId = requestCounter.incrementAndGet();

                log.debugf("[%d] OpenAI inference: model=%s", requestId, request.getModel());

                OpenAIRequest openaiRequest = buildOpenAIRequest(request);
                String apiKey = getApiKey();

                return client.chatCompletions("Bearer " + apiKey, openaiRequest)
                                .map(response -> {
                                        long duration = System.currentTimeMillis() - startTime;
                                        log.debugf("[%d] OpenAI response in %dms", requestId, duration);

                                        String content = response.getChoices().get(0).getMessage().getContent();

                                        return InferenceResponse.builder()
                                                        .requestId(request.getRequestId())
                                                        .content(content)
                                                        .model(response.getModel())
                                                        .durationMs(duration)
                                                        .metadata("provider", PROVIDER_ID)
                                                        .metadata("prompt_tokens",
                                                                        response.getUsage() != null ? response
                                                                                        .getUsage().getPromptTokens()
                                                                                        : 0)
                                                        .metadata("completion_tokens",
                                                                        response.getUsage() != null ? response
                                                                                        .getUsage()
                                                                                        .getCompletionTokens() : 0)
                                                        .metadata("total_tokens",
                                                                        response.getUsage() != null ? response
                                                                                        .getUsage().getTotalTokens()
                                                                                        : 0)
                                                        .build();
                                });
        }

        @Override
        public Multi<StreamChunk> inferStream(ProviderRequest request, TenantContext context) {
                AtomicInteger chunkIndex = new AtomicInteger(0);

                OpenAIRequest openaiRequest = buildOpenAIRequest(request);
                openaiRequest.setStream(true);
                String apiKey = getApiKey();

                return client.chatCompletionsStream("Bearer " + apiKey, openaiRequest)
                                .map(chunk -> {
                                        int index = chunkIndex.getAndIncrement();
                                        String content = "";
                                        boolean isFinal = false;

                                        if (chunk.getChoices() != null && !chunk.getChoices().isEmpty()) {
                                                OpenAIStreamChoice choice = chunk.getChoices().get(0);
                                                if (choice.getDelta() != null
                                                                && choice.getDelta().getContent() != null) {
                                                        content = choice.getDelta().getContent();
                                                }
                                                isFinal = "stop".equals(choice.getFinishReason());
                                        }

                                        return isFinal
                                                        ? StreamChunk.finalChunk(request.getRequestId(), index, content)
                                                        : StreamChunk.of(request.getRequestId(), index, content);
                                });
        }

        private OpenAIRequest buildOpenAIRequest(ProviderRequest request) {
                OpenAIRequest openaiRequest = new OpenAIRequest();
                openaiRequest.setModel(request.getModel());
                openaiRequest.setStream(request.isStreaming());

                // Convert messages
                if (request.getMessages() != null) {
                        List<OpenAIMessage> messages = request.getMessages().stream()
                                        .map(msg -> new OpenAIMessage(msg.getRole().name().toLowerCase(),
                                                        msg.getContent()))
                                        .collect(Collectors.toList());
                        openaiRequest.setMessages(messages);
                }

                // Set parameters
                if (request.getParameters().containsKey("temperature")) {
                        openaiRequest.setTemperature(
                                        ((Number) request.getParameters().get("temperature")).doubleValue());
                }
                if (request.getParameters().containsKey("max_tokens")) {
                        openaiRequest.setMaxTokens(((Number) request.getParameters().get("max_tokens")).intValue());
                }

                return openaiRequest;
        }
}