package tech.kayys.golek.provider.openai;

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
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.provider.core.adapter.CloudProviderAdapter;
import tech.kayys.golek.provider.core.spi.StreamingProvider;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * OpenAI provider adapter for cloud LLM inference.
 * Supports GPT-4, GPT-3.5-Turbo, and other OpenAI models.
 */
@ApplicationScoped
public class OpenAIProvider extends CloudProviderAdapter implements StreamingProvider {

        private static final String PROVIDER_ID = "openai";
        private static final String PROVIDER_NAME = "OpenAI";
        private static final String VERSION = "1.0.0";

        @Inject
        @RestClient
        OpenAIClient client;

        @Inject
        OpenAIConfig config;

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
                return "https://api.openai.com";
        }

        @Override
        protected String getApiKeyEnvironmentVariable() {
                return "OPENAI_API_KEY";
        }

        @Override
        protected Uni<Boolean> performHealthCheckRequest() {
                return client.listModels("Bearer " + getApiKey())
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
                                .maxContextTokens(128000)
                                .maxOutputTokens(4096)
                                .supportedModels(Set.of(
                                                "gpt-4o",
                                                "gpt-4o-mini",
                                                "gpt-4-turbo",
                                                "gpt-4-turbo-preview",
                                                "gpt-4",
                                                "gpt-4-32k",
                                                "gpt-3.5-turbo",
                                                "gpt-3.5-turbo-16k",
                                                "text-embedding-3-small",
                                                "text-embedding-3-large",
                                                "text-embedding-ada-002"))
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
                                .description("OpenAI GPT models - industry standard for LLM inference")
                                .version(VERSION)
                                .vendor("OpenAI")
                                .documentationUrl("https://platform.openai.com/docs")
                                .metadata("deployment", "cloud")
                                .metadata("requires_api_key", "true")
                                .metadata("pricing_url", "https://openai.com/pricing")
                                .build();
        }

        @Override
        public boolean supports(String model, TenantContext context) {
                return capabilities().getSupportedModels().contains(model) ||
                                model.startsWith("gpt-") ||
                                model.startsWith("text-embedding");
        }

        @Override
        protected Uni<InferenceResponse> doInfer(ProviderRequest request) {
                trackRequest();
                long startTime = System.currentTimeMillis();
                int requestId = requestCounter.incrementAndGet();

                log.debugf("[%d] OpenAI inference: model=%s", requestId, request.getModel());

                OpenAIRequest openaiRequest = buildOpenAIRequest(request);

                return client.chatCompletions("Bearer " + getApiKey(), openaiRequest)
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
                                                        .metadata("finish_reason",
                                                                        (response.getChoices() != null && !response
                                                                                        .getChoices().isEmpty())
                                                                                                        ? response.getChoices()
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

                OpenAIRequest openaiRequest = buildOpenAIRequest(request);
                openaiRequest.setStream(true);

                return client.chatCompletionsStream("Bearer " + getApiKey(), openaiRequest)
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

                                        return StreamChunk.builder()
                                                        .index(index)
                                                        .delta(content)
                                                        .model(chunk.getModel())
                                                        .isFinal(isFinal)
                                                        .build();
                                })
                                .onFailure().invoke(this::trackError)
                                .onFailure().transform(this::wrapException);
        }

        private OpenAIRequest buildOpenAIRequest(ProviderRequest request) {
                OpenAIRequest openaiRequest = new OpenAIRequest();
                openaiRequest.setModel(request.getModel());

                // Convert messages
                if (request.getMessages() != null) {
                        List<OpenAIMessage> messages = request.getMessages().stream()
                                        .map(msg -> new OpenAIMessage(msg.getRole(), msg.getContent()))
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
                if (request.getParameters().containsKey("top_p")) {
                        openaiRequest.setTopP(((Number) request.getParameters().get("top_p")).doubleValue());
                }
                if (request.getParameters().containsKey("frequency_penalty")) {
                        openaiRequest.setFrequencyPenalty(
                                        ((Number) request.getParameters().get("frequency_penalty")).doubleValue());
                }
                if (request.getParameters().containsKey("presence_penalty")) {
                        openaiRequest.setPresencePenalty(
                                        ((Number) request.getParameters().get("presence_penalty")).doubleValue());
                }

                return openaiRequest;
        }

        private Throwable wrapException(Throwable ex) {
                return new RuntimeException("OpenAI request failed: " + ex.getMessage(), ex);
        }
}