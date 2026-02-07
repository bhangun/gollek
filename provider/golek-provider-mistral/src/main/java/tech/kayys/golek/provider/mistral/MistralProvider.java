package tech.kayys.golek.provider.mistral;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.*;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.wayang.tenant.TenantContext;

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

@ApplicationScoped
public class MistralProvider implements StreamingProvider {

    private static final Logger log = Logger.getLogger(MistralProvider.class);
    private static final String PROVIDER_ID = "mistral";
    private static final String PROVIDER_NAME = "Mistral AI";
    private static final String VERSION = "1.0.0";

    @Inject
    @RestClient
    MistralClient client;

    private ProviderConfig providerConfig;
    private final AtomicInteger requestCounter = new AtomicInteger(0);

    @Override
    public void initialize(ProviderConfig config) {
        this.providerConfig = config;
        log.info("Mistral provider initialized");
    }

    @Override
    public void shutdown() {
        log.info("Mistral provider shutting down");
    }

    @Override
    public Uni<ProviderHealth> health() {
        String apiKey = getApiKey();
        if (apiKey == null || apiKey.isEmpty()) {
            return Uni.createFrom().item(ProviderHealth.unhealthy("API key is missing"));
        }
        // Simplified health check for Mistral
        return Uni.createFrom().item(ProviderHealth.healthy("Mistral provider initialized"));
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
                .multimodal(false)
                .embeddings(true)
                .maxContextTokens(128000)
                .maxOutputTokens(4096)
                .supportedModels(Set.of(
                        "mistral-tiny",
                        "mistral-small",
                        "mistral-medium",
                        "mistral-large-latest",
                        "open-mistral-7b",
                        "open-mixtral-8x7b",
                        "open-mixtral-8x22b"))
                .supportedLanguages(List.of("en", "fr", "de", "es", "it"))
                .build();
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(PROVIDER_NAME)
                .description("Mistral AI - High-performance open-source models")
                .version(VERSION)
                .vendor("Mistral AI")
                .homepage("https://mistral.ai")
                .build();
    }

    @Override
    public boolean supports(String model, TenantContext context) {
        return capabilities().getSupportedModels().contains(model) ||
                model.startsWith("mistral-") ||
                model.startsWith("open-");
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        long startTime = System.currentTimeMillis();
        int requestId = requestCounter.incrementAndGet();

        MistralRequest mistralRequest = buildMistralRequest(request);
        String apiKey = getApiKey();

        return client.chatCompletions("Bearer " + apiKey, mistralRequest)
                .map(response -> {
                    long duration = System.currentTimeMillis() - startTime;
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
                            .build();
                });
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request, TenantContext context) {
        AtomicInteger chunkIndex = new AtomicInteger(0);
        MistralRequest mistralRequest = buildMistralRequest(request);
        mistralRequest.setStream(true);
        String apiKey = getApiKey();

        return client.chatCompletionsStream("Bearer " + apiKey, mistralRequest)
                .map(chunk -> {
                    int index = chunkIndex.getAndIncrement();
                    String content = "";
                    boolean isFinal = false;

                    if (chunk.getChoices() != null && !chunk.getChoices().isEmpty()) {
                        MistralStreamChoice choice = chunk.getChoices().get(0);
                        if (choice.getDelta() != null && choice.getDelta().getContent() != null) {
                            content = choice.getDelta().getContent();
                        }
                        isFinal = "stop".equals(choice.getFinishReason());
                    }

                    return isFinal
                            ? StreamChunk.finalChunk(request.getRequestId(), index, content)
                            : StreamChunk.of(request.getRequestId(), index, content);
                });
    }

    private MistralRequest buildMistralRequest(ProviderRequest request) {
        MistralRequest mistralRequest = new MistralRequest();
        mistralRequest.setModel(request.getModel());
        mistralRequest.setStream(request.isStreaming());

        if (request.getMessages() != null) {
            List<MistralMessage> messages = request.getMessages().stream()
                    .map(msg -> new MistralMessage(msg.getRole().name().toLowerCase(), msg.getContent()))
                    .collect(Collectors.toList());
            mistralRequest.setMessages(messages);
        }

        if (request.getParameters().containsKey("temperature")) {
            mistralRequest.setTemperature(((Number) request.getParameters().get("temperature")).doubleValue());
        }
        if (request.getParameters().containsKey("max_tokens")) {
            mistralRequest.setMaxTokens(((Number) request.getParameters().get("max_tokens")).intValue());
        }

        return mistralRequest;
    }
}
