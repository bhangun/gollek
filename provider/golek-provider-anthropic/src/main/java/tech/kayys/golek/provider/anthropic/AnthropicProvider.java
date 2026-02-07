package tech.kayys.golek.provider.anthropic;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.jboss.logging.Logger;
import tech.kayys.golek.spi.Message;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.*;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.exception.ProviderException;
import java.util.concurrent.atomic.AtomicInteger;

import java.util.*;
import java.util.stream.Collectors;

@ApplicationScoped
public class AnthropicProvider implements StreamingProvider {

    private static final String PROVIDER_ID = "anthropic";
    private static final String PROVIDER_NAME = "Anthropic Claude";
    private static final String API_VERSION = "2023-06-01"; // Constant for fixed version

    private static final Logger log = Logger.getLogger(AnthropicProvider.class);

    @Inject
    @RestClient
    AnthropicClient anthropicClient;

    @Inject
    @RestClient
    AnthropicStreamingClient streamingClient;

    private String apiKey;
    private String baseUrl = "https://api.anthropic.com";
    private String apiVersion = API_VERSION; // Instance variable for configurable version

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return PROVIDER_NAME;
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name("Anthropic")
                .version(API_VERSION)
                .description("Anthropic Claude models integration")
                .vendor("Anthropic")
                .homepage("https://www.anthropic.com")
                .build();
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
    public void initialize(ProviderConfig config) {
        // Required parameters
        String key = config.getString("api.key");
        if (key == null || key.isBlank()) {
            // Check required secret
            key = config.getRequiredSecret("api.key");
        }
        this.apiKey = key;

        // Optional parameters
        this.baseUrl = config.getString("api.base-url", "https://api.anthropic.com");
        this.apiVersion = config.getString("api.version", API_VERSION);

        log.info("Anthropic provider initialized");
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        // Check if the model is in our known list of Anthropic models
        List<String> supportedModels = Arrays.asList(
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1",
                "claude-2.0");

        return supportedModels.contains(modelId.toLowerCase());
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        return Uni.createFrom().deferred(() -> {
            try {
                // Create a non-streaming request
                AnthropicRequest anthropicRequest = mapToAnthropicRequest(request);

                // Ensure streaming is disabled for regular inference
                anthropicRequest = new AnthropicRequest(
                        anthropicRequest.model(),
                        anthropicRequest.messages(),
                        anthropicRequest.maxTokens(),
                        anthropicRequest.temperature(),
                        false, // Disable streaming
                        anthropicRequest.systemPrompt());

                return anthropicClient
                        .createMessage(anthropicRequest, apiKey, apiVersion)
                        .onItem().transform(response -> mapToInferenceResponse(response, request))
                        .onFailure().transform(throwable -> {
                            // Log the error
                            log.error("Error calling Anthropic API", throwable);

                            // Wrap in provider-specific exception
                            throw new ProviderException(
                                    "Anthropic API call failed: " + throwable.getMessage(),
                                    throwable);
                        });
            } catch (Exception e) {
                log.error("Error preparing Anthropic request", e);
                return Uni.createFrom().failure(
                        new ProviderException(
                                "Failed to prepare Anthropic request: " + e.getMessage(),
                                e));
            }
        });
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request, TenantContext context) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                // Create a streaming request by setting stream=true
                AnthropicRequest anthropicRequest = mapToAnthropicRequest(request);

                // Set streaming to true in the request
                anthropicRequest = new AnthropicRequest(
                        anthropicRequest.model(),
                        anthropicRequest.messages(),
                        anthropicRequest.maxTokens(),
                        anthropicRequest.temperature(),
                        true, // Enable streaming
                        anthropicRequest.systemPrompt());

                AtomicInteger chunkIndex = new AtomicInteger(0);

                // Use the streaming client to get the stream
                streamingClient.streamMessage(anthropicRequest, apiKey, apiVersion)
                        .subscribe()
                        .with(
                                chunk -> {
                                    // Process each chunk from the stream
                                    String content = "";
                                    boolean isFinal = false;

                                    if ("content_block_delta".equals(chunk.getType())) {
                                        // This is a delta chunk with incremental content
                                        if (chunk.getDelta() != null && chunk.getDelta().getText() != null) {
                                            content = chunk.getDelta().getText();
                                        }
                                    } else if ("content_block_stop".equals(chunk.getType())) {
                                        // This indicates the end of a content block
                                        isFinal = true;
                                    } else if ("message_stop".equals(chunk.getType())) {
                                        // This indicates the end of the entire message
                                        isFinal = true;
                                    }

                                    // Create and emit the stream chunk
                                    StreamChunk streamChunk;
                                    if (isFinal) {
                                        streamChunk = StreamChunk.finalChunk(request.getRequestId(),
                                                chunkIndex.getAndIncrement(), content);
                                    } else {
                                        streamChunk = StreamChunk.of(request.getRequestId(),
                                                chunkIndex.getAndIncrement(), content);
                                    }

                                    emitter.emit(streamChunk);
                                },
                                throwable -> {
                                    log.error("Error in Anthropic streaming", throwable);
                                    emitter.fail(new ProviderException(
                                            "Anthropic streaming failed: " + throwable.getMessage(),
                                            throwable));
                                },
                                () -> {
                                    // Stream completed normally
                                    emitter.complete();
                                });
            } catch (Exception e) {
                log.error("Error preparing Anthropic streaming request", e);
                emitter.fail(new ProviderException(
                        "Failed to prepare Anthropic streaming request: " + e.getMessage(),
                        e));
            }
        });
    }

    @Override
    public Uni<ProviderHealth> health() {
        // Create a minimal request to test the API connection
        ProviderRequest testRequest = ProviderRequest.builder()
                .model("claude-3-haiku-20240307")
                .message(Message.user("health check"))
                .parameter("max_tokens", 10)
                .build();

        return infer(testRequest, null)
                .map(response -> ProviderHealth.healthy(id()))
                .onFailure().recoverWithItem(throwable -> {
                    log.error("Health check failed", throwable);
                    return ProviderHealth.unhealthy("Health check failed: " + throwable.getMessage());
                });
    }

    @Override
    public void shutdown() {
        // No specific cleanup needed for this provider
    }

    private AnthropicRequest mapToAnthropicRequest(ProviderRequest request) {
        // Separate system messages from other messages
        String systemPrompt = "";
        List<Message> nonSystemMessages = new ArrayList<>();

        for (Message msg : request.getMessages()) {
            if (msg.getRole() == Message.Role.SYSTEM) {
                // Concatenate system messages if there are multiple
                if (!systemPrompt.isEmpty()) {
                    systemPrompt += "\n";
                }
                systemPrompt += msg.getContent();
            } else {
                nonSystemMessages.add(msg);
            }
        }

        // Map non-system messages to Anthropic format
        List<AnthropicRequest.Message> anthropicMessages = nonSystemMessages.stream()
                .map(msg -> new AnthropicRequest.Message(
                        mapRoleToAnthropic(msg.getRole()),
                        msg.getContent() != null ? msg.getContent() : ""))
                .collect(Collectors.toList());

        // Extract parameters
        double temperature = request.getTemperature();
        int maxTokens = request.getMaxTokens();

        return new AnthropicRequest(
                request.getModel(),
                anthropicMessages,
                maxTokens,
                temperature,
                request.isStreaming(),
                systemPrompt);
    }

    private String mapRoleToAnthropic(Message.Role role) {
        switch (role) {
            case SYSTEM:
                // Anthropic handles system messages differently - they go as a separate
                // parameter
                return "user"; // For now, treat as user, but in practice system messages need special handling
            case USER:
                return "user";
            case ASSISTANT:
                return "assistant";
            case TOOL:
                return "user"; // Anthropic treats tool responses as user messages
            default:
                return "user";
        }
    }

    private InferenceResponse mapToInferenceResponse(AnthropicResponse anthropicResponse, ProviderRequest request) {
        // Extract the content from the response
        String content = "";
        if (anthropicResponse.content() != null && !anthropicResponse.content().isEmpty()) {
            content = anthropicResponse.content().get(0).text();
        }

        // Calculate token usage
        int tokensUsed = 0;
        if (anthropicResponse.usage() != null) {
            tokensUsed = anthropicResponse.usage().outputTokens();
        }

        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .content(content)
                .model(anthropicResponse.model())
                .tokensUsed(tokensUsed)
                .build();
    }
}