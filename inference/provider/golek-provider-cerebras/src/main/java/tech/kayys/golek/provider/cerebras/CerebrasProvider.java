package tech.kayys.golek.provider.cerebras;

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
 * Cerebras provider adapter for high-speed cloud inference.
 * Supports Llama 3.1 8B/70B at extreme speeds.
 */
@ApplicationScoped
public class CerebrasProvider implements StreamingProvider {

    private static final String PROVIDER_ID = "cerebras";
    private static final String PROVIDER_NAME = "Cerebras";
    private static final String VERSION = "1.0.0";

    private static final Logger log = Logger.getLogger(CerebrasProvider.class);

    @Inject
    @RestClient
    CerebrasClient client;

    @Inject
    CerebrasConfig configDetails; // Renamed to avoid confusion with ProviderConfig

    private final AtomicInteger requestCounter = new AtomicInteger(0);
    private String apiKey;

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
        log.info("Cerebras provider initialized");
    }

    @Override
    public void shutdown() {
        log.info("Cerebras provider shutting down");
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(true)
                .multimodal(false)
                .maxContextTokens(128000)
                .build();
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(PROVIDER_NAME)
                .description("Cerebras - The world's fastest inference engine")
                .version(VERSION)
                .vendor("Cerebras")
                .homepage("https://inference.cerebras.ai/docs")
                .build();
    }

    @Override
    public boolean supports(String model, TenantContext context) {
        // Hardcoded support for now, or check config
        return model.contains("cerebras") ||
                model.contains("llama3.1-70b") ||
                model.contains("llama3.1-8b");
    }

    @Override
    public Uni<ProviderHealth> health() {
        // Simple health check: if we have an API key, we assume we're healthy for now.
        // In a real implementation, we might make a lightweight call to the API.
        return Uni.createFrom().item(() -> apiKey != null && !apiKey.isBlank()
                ? ProviderHealth.healthy(id())
                : ProviderHealth.unhealthy("API key not configured"));
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        long startTime = System.currentTimeMillis();
        int requestId = requestCounter.incrementAndGet();

        log.debugf("[%d] Cerebras inference: model=%s", requestId, request.getModel());

        CerebrasRequest cerebrasRequest = buildCerebrasRequest(request);

        return client.chatCompletions("Bearer " + apiKey, cerebrasRequest)
                .map(response -> {
                    long duration = System.currentTimeMillis() - startTime;
                    log.debugf("[%d] Cerebras response in %dms", requestId, duration);

                    String content = "";
                    if (response.getChoices() != null && !response.getChoices().isEmpty()) {
                        content = response.getChoices().get(0).getMessage().getContent();
                    }

                    return InferenceResponse.builder()
                            .requestId(
                                    request.getRequestId() != null ? request.getRequestId() : String.valueOf(requestId))
                            .content(content)
                            .model(response.getModel())
                            .inputTokens(response.getUsage() != null ? response.getUsage().getPromptTokens() : 0)
                            .outputTokens(response.getUsage() != null ? response.getUsage().getCompletionTokens() : 0)
                            .tokensUsed(response.getUsage() != null ? response.getUsage().getTotalTokens() : 0)
                            .durationMs(duration)
                            .metadata("provider", PROVIDER_ID)
                            .build();
                })
                .onFailure().transform(this::wrapException);
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request, TenantContext context) {
        AtomicInteger chunkIndex = new AtomicInteger(0);
        // Ensure requestId is available; if request.getRequestId() is null, generate
        // one?
        // ProviderRequest usually has one. If not, we can use the atomic counter.
        String reqId = request.getRequestId() != null ? request.getRequestId()
                : String.valueOf(requestCounter.incrementAndGet());

        CerebrasRequest cerebrasRequest = buildCerebrasRequest(request);
        cerebrasRequest.setStream(true);

        return client.chatCompletionsStream("Bearer " + apiKey, cerebrasRequest)
                .map(chunk -> {
                    int index = chunkIndex.getAndIncrement();
                    String content = "";
                    // boolean isFinal = false; // Unused for now

                    if (chunk.getChoices() != null && !chunk.getChoices().isEmpty()) {
                        CerebrasStreamChoice choice = chunk.getChoices().get(0);
                        if (choice.getDelta() != null && choice.getDelta().getContent() != null) {
                            content = choice.getDelta().getContent();
                        }
                    }

                    return StreamChunk.of(reqId, index, content);
                })
                .onFailure().transform(this::wrapException);
    }

    private CerebrasRequest buildCerebrasRequest(ProviderRequest request) {
        CerebrasRequest cerebrasRequest = new CerebrasRequest();
        cerebrasRequest.setModel(request.getModel());

        // Convert messages
        if (request.getMessages() != null) {
            List<CerebrasMessage> messages = request.getMessages().stream()
                    .map(msg -> new CerebrasMessage(msg.getRole().toString().toLowerCase(), msg.getContent()))
                    .collect(Collectors.toList());
            cerebrasRequest.setMessages(messages);
        }

        // Set parameters
        if (request.getParameters() != null) {
            if (request.getParameters().containsKey("temperature")) {
                cerebrasRequest.setTemperature(((Number) request.getParameters().get("temperature")).doubleValue());
            }
            if (request.getParameters().containsKey("max_tokens")) {
                cerebrasRequest.setMaxTokens(((Number) request.getParameters().get("max_tokens")).intValue());
            }
            if (request.getParameters().containsKey("top_p")) {
                cerebrasRequest.setTopP(((Number) request.getParameters().get("top_p")).doubleValue());
            }
        }

        return cerebrasRequest;
    }

    private Throwable wrapException(Throwable ex) {
        return new RuntimeException("Cerebras request failed: " + ex.getMessage(), ex);
    }
}