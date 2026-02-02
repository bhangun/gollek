package tech.kayys.golek.provider.cerebras;

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
 * Cerebras provider adapter for high-speed cloud inference.
 * Supports Llama 3.1 8B/70B at extreme speeds.
 */
@ApplicationScoped
public class CerebrasProvider extends CloudProviderAdapter implements StreamingProvider {

    private static final String PROVIDER_ID = "cerebras";
    private static final String PROVIDER_NAME = "Cerebras";
    private static final String VERSION = "1.0.0";

    @Inject
    @RestClient
    CerebrasClient client;

    @Inject
    CerebrasConfig config;

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
        return "https://api.cerebras.ai";
    }

    @Override
    protected String getApiKeyEnvironmentVariable() {
        return "CEREBRAS_API_KEY";
    }

    @Override
    protected Uni<Boolean> performHealthCheckRequest() {
        // Cerebras doesn't have a simple models list in some versions, but let's try
        return Uni.createFrom().item(true);
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(true)
                .multimodal(false)
                .embeddings(false)
                .maxContextTokens(128000)
                .supportedModels(Set.of(
                        "llama3.1-70b",
                        "llama3.1-8b",
                        "llama-3.1-70b-versatile",
                        "llama-3.1-8b-instant"))
                .supportedLanguages(List.of("en"))
                .metadata("speed", "extreme")
                .toolCalling(true)
                .build();
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .displayName(PROVIDER_NAME)
                .description("Cerebras - The world's fastest inference engine")
                .version(VERSION)
                .vendor("Cerebras")
                .documentationUrl("https://inference.cerebras.ai/docs")
                .metadata("deployment", "cloud")
                .metadata("requires_api_key", "true")
                .metadata("pricing_url", "https://cerebras.ai/pricing")
                .build();
    }

    @Override
    public boolean supports(String model, TenantContext context) {
        return capabilities().getSupportedModels().contains(model) ||
                model.contains("cerebras") ||
                (model.contains("llama") && config.preferForLlama());
    }

    @Override
    protected Uni<InferenceResponse> doInfer(ProviderRequest request) {
        trackRequest();
        long startTime = System.currentTimeMillis();
        int requestId = requestCounter.incrementAndGet();

        log.debugf("[%d] Cerebras inference: model=%s", requestId, request.getModel());

        CerebrasRequest cerebrasRequest = buildCerebrasRequest(request);

        return client.chatCompletions("Bearer " + getApiKey(), cerebrasRequest)
                .map(response -> {
                    long duration = System.currentTimeMillis() - startTime;
                    log.debugf("[%d] Cerebras response in %dms", requestId, duration);

                    String content = response.getChoices().get(0).getMessage().getContent();

                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .content(content)
                            .model(response.getModel())
                            .durationMs(duration)
                            .metadata("provider", PROVIDER_ID)
                            .metadata("prompt_tokens",
                                    response.getUsage() != null ? response.getUsage().getPromptTokens() : 0)
                            .metadata("completion_tokens",
                                    response.getUsage() != null ? response.getUsage().getCompletionTokens() : 0)
                            .metadata("total_tokens",
                                    response.getUsage() != null ? response.getUsage().getTotalTokens() : 0)
                            .metadata("finish_reason",
                                    (response.getChoices() != null && !response.getChoices().isEmpty())
                                            ? response.getChoices().get(0).getFinishReason()
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

        CerebrasRequest cerebrasRequest = buildCerebrasRequest(request);
        cerebrasRequest.setStream(true);

        return client.chatCompletionsStream("Bearer " + getApiKey(), cerebrasRequest)
                .map(chunk -> {
                    int index = chunkIndex.getAndIncrement();
                    String content = "";
                    boolean isFinal = false;

                    if (chunk.getChoices() != null && !chunk.getChoices().isEmpty()) {
                        CerebrasStreamChoice choice = chunk.getChoices().get(0);
                        if (choice.getDelta() != null && choice.getDelta().getContent() != null) {
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

    private CerebrasRequest buildCerebrasRequest(ProviderRequest request) {
        CerebrasRequest cerebrasRequest = new CerebrasRequest();
        cerebrasRequest.setModel(request.getModel());

        // Convert messages
        if (request.getMessages() != null) {
            List<CerebrasMessage> messages = request.getMessages().stream()
                    .map(msg -> new CerebrasMessage(msg.getRole(), msg.getContent()))
                    .collect(Collectors.toList());
            cerebrasRequest.setMessages(messages);
        }

        // Set parameters
        if (request.getParameters().containsKey("temperature")) {
            cerebrasRequest.setTemperature(((Number) request.getParameters().get("temperature")).doubleValue());
        }
        if (request.getParameters().containsKey("max_tokens")) {
            cerebrasRequest.setMaxTokens(((Number) request.getParameters().get("max_tokens")).intValue());
        }
        if (request.getParameters().containsKey("top_p")) {
            cerebrasRequest.setTopP(((Number) request.getParameters().get("top_p")).doubleValue());
        }

        return cerebrasRequest;
    }

    private Throwable wrapException(Throwable ex) {
        return new RuntimeException("Cerebras request failed: " + ex.getMessage(), ex);
    }
}