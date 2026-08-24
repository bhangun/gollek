package tech.kayys.gollek.server.api.v1;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.jboss.resteasy.reactive.RestSseElementType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.server.SdkProvider;
import tech.kayys.gollek.sdk.core.ChatParams;
import tech.kayys.gollek.sdk.core.GollekChatService;
import tech.kayys.gollek.server.api.v1.dto.ChatCompletionMapper;
import tech.kayys.gollek.server.api.v1.dto.ChatCompletionMessage;
import tech.kayys.gollek.server.api.v1.dto.ChatCompletionRequest;
import tech.kayys.gollek.server.api.v1.dto.ChatCompletionResponse;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

@Path("/v1/gollek/chat")
public class ChatResource {

    private static final Logger LOG = LoggerFactory.getLogger(ChatResource.class);

    @Inject
    SdkProvider sdkProvider;
    
    @Inject
    MeterRegistry meterRegistry;

    private final GollekChatService chatService = new GollekChatService();

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Uni<ChatCompletionResponse> chatCompletion(ChatCompletionRequest req) {
        if (req.stream()) {
            throw new IllegalArgumentException("For streaming, use Accept: text/event-stream");
        }
        
        Timer.Sample sample = Timer.start(meterRegistry);
        LOG.info("Received REST chat request for model: {}", req.model());
        
        List<Message> history = ChatCompletionMapper.toMessages(req.messages());
        List<ToolDefinition> tools = req.tools() != null ? req.tools() : List.of();
        ChatParams params = ChatParams.of(req.temperature(), req.maxTokens());

        return chatService.chat(sdkProvider.getSdk(), req.model(), null, history, tools, params)
                .map(resp -> {
                    List<tech.kayys.gollek.spi.tool.ToolCall> mappedToolCalls = null;
                    if (resp.getToolCalls() != null && !resp.getToolCalls().isEmpty()) {
                        mappedToolCalls = resp.getToolCalls().stream()
                                .map(tc -> new tech.kayys.gollek.spi.tool.ToolCall(UUID.randomUUID().toString(), tc.name(), tc.arguments(), "function"))
                                .toList();
                    }
                    ChatCompletionMessage message = new ChatCompletionMessage(
                            "assistant", resp.getContent(), null, mappedToolCalls);
                    String finishReasonStr = resp.getFinishReason() != null ? resp.getFinishReason().name().toLowerCase() : null;
                    ChatCompletionResponse.Choice choice = new ChatCompletionResponse.Choice(
                            0, message, finishReasonStr);
                    
                    ChatCompletionResponse.Usage usage = null;
                    if (resp.getInputTokens() > 0 || resp.getOutputTokens() > 0) {
                        usage = new ChatCompletionResponse.Usage(
                                resp.getInputTokens(),
                                resp.getOutputTokens(),
                                resp.getInputTokens() + resp.getOutputTokens()
                        );
                    }

                    sample.stop(meterRegistry.timer("gollek.rest.chat.duration"));
                    meterRegistry.counter("gollek.rest.chat.count").increment();
                    if (usage != null) {
                        meterRegistry.counter("gollek.rest.chat.tokens").increment(usage.totalTokens());
                    }
                    LOG.info("REST Chat response completed for model: {}, finish reason: {}", req.model(), finishReasonStr);

                    return new ChatCompletionResponse(
                            UUID.randomUUID().toString(),
                            "chat.completion",
                            Instant.now().getEpochSecond(),
                            req.model(),
                            List.of(choice),
                            usage
                    );
                })
                .onFailure().invoke(err -> {
                    LOG.error("Failed REST chat request for model: {}", req.model(), err);
                    meterRegistry.counter("gollek.rest.chat.error").increment();
                });
    }

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.SERVER_SENT_EVENTS)
    @RestSseElementType(MediaType.APPLICATION_JSON)
    public Multi<StreamingInferenceChunk> chatStream(ChatCompletionRequest req) {
        LOG.info("Received REST chat stream request for model: {}", req.model());
        
        List<Message> history = ChatCompletionMapper.toMessages(req.messages());
        List<ToolDefinition> tools = req.tools() != null ? req.tools() : List.of();
        ChatParams params = ChatParams.of(req.temperature(), req.maxTokens());
        
        return chatService.streamChat(sdkProvider.getSdk(), req.model(), null, history, tools, params)
                .onItem().invoke(chunk -> {
                    meterRegistry.counter("gollek.rest.chat.stream.chunks").increment();
                })
                .onFailure().invoke(err -> {
                    LOG.error("Failed REST chat stream request for model: {}", req.model(), err);
                    meterRegistry.counter("gollek.rest.chat.stream.error").increment();
                });
    }
}
