package tech.kayys.gollek.sdk.core;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Shared chat service facade that abstracts away the construction of
 * InferenceRequests, allowing a unified multi-turn chat and tool-use
 * experience across CLI, REST API, gRPC, and embedded environments (Wayang).
 */
import tech.kayys.gollek.sdk.core.observability.SdkObservabilityProvider;

public final class GollekChatService {
    private final SdkObservabilityProvider obs;

    public GollekChatService() {
        this.obs = SdkObservabilityProvider.noop();
    }

    public GollekChatService(SdkObservabilityProvider obs) {
        this.obs = obs != null ? obs : SdkObservabilityProvider.noop();
    }

    public Multi<StreamingInferenceChunk> streamChat(
            GollekSdk sdk, 
            String modelId, 
            String systemPrompt,
            List<Message> history, 
            List<ToolDefinition> tools, 
            ChatParams p) {

        InferenceRequest req = buildRequest(modelId, systemPrompt, history, tools, p, true);
        obs.getAuditLogger().logClientRequestStart(req.getRequestId(), modelId);
        long startMs = System.currentTimeMillis();
        
        return sdk.streamCompletion(req)
                .onItem().invoke(chunk -> {
                    obs.getMetricsCollector().recordClientChunk(modelId);
                    // Approximation of TTFT for stream
                    if (chunk.getText() != null && !chunk.getText().isEmpty()) {
                        obs.getMetricsCollector().recordClientTtft(modelId, System.currentTimeMillis() - startMs);
                    }
                })
                .onCompletion().invoke(() -> {
                    long duration = System.currentTimeMillis() - startMs;
                    obs.getAuditLogger().logClientRequestComplete(req.getRequestId(), modelId, 0, duration);
                })
                .onFailure().invoke(err -> {
                    obs.getMetricsCollector().recordClientFailure(modelId, err.getClass().getSimpleName());
                    obs.getAuditLogger().logClientRequestFailure(req.getRequestId(), modelId, err.getClass().getSimpleName());
                });
    }

    public Uni<InferenceResponse> chat(
            GollekSdk sdk, 
            String modelId, 
            String systemPrompt,
            List<Message> history, 
            List<ToolDefinition> tools, 
            ChatParams p) {

        InferenceRequest req = buildRequest(modelId, systemPrompt, history, tools, p, false);
        obs.getAuditLogger().logClientRequestStart(req.getRequestId(), modelId);
        long startMs = System.currentTimeMillis();
        
        return Uni.createFrom().completionStage(() -> sdk.createCompletionAsync(req))
                .onItem().invoke(resp -> {
                    long duration = System.currentTimeMillis() - startMs;
                    obs.getMetricsCollector().recordClientSuccess(modelId, duration, resp.getTokensUsed());
                    obs.getAuditLogger().logClientRequestComplete(req.getRequestId(), modelId, resp.getTokensUsed(), duration);
                })
                .onFailure().invoke(err -> {
                    obs.getMetricsCollector().recordClientFailure(modelId, err.getClass().getSimpleName());
                    obs.getAuditLogger().logClientRequestFailure(req.getRequestId(), modelId, err.getClass().getSimpleName());
                });
    }

    private InferenceRequest buildRequest(
            String modelId, 
            String systemPrompt,
            List<Message> history, 
            List<ToolDefinition> tools,
            ChatParams p, 
            boolean streaming) {

        List<Message> messages = new ArrayList<>();
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            messages.add(Message.system(systemPrompt));
        }
        if (history != null) {
            messages.addAll(history);
        }

        InferenceRequest.Builder reqBuilder = InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(modelId)
                .messages(messages)
                .streaming(streaming)
                .parameter("temperature", p.temperature())
                .parameter("max_tokens", p.maxTokens())
                .parameter("top_p", p.topP())
                .parameter("repeat_penalty", p.repeatPenalty());

        if (tools != null && !tools.isEmpty()) {
            reqBuilder.tools(tools);
        }

        return reqBuilder.build();
    }
}
