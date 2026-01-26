package tech.kayys.golek.provider.core.adapter;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.provider.core.spi.StreamingProvider;
import tech.kayys.golek.provider.core.streaming.ChunkProcessor;
import tech.kayys.golek.provider.core.streaming.StreamHandler;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Adapter that adds streaming capabilities to providers
 */
public abstract class StreamingAdapter extends AbstractProvider implements StreamingProvider {

    protected StreamHandler streamHandler;
    protected ChunkProcessor chunkProcessor;

    @Override
    public Multi<StreamChunk> stream(ProviderRequest request) {
        if (!isInitialized()) {
            return Multi.createFrom().failure(
                    new IllegalStateException("Provider not initialized"));
        }

        String tenantId = request.getTenantContext() != null
                ? request.getTenantContext().getTenantId()
                : "default";

        return checkRateLimit(tenantId)
                .onItem().transformToMulti(v -> doStream(request))
                .onFailure().transform(this::handleFailure);
    }

    @Override
    protected Uni<InferenceResponse> doInfer(ProviderRequest request) {
        if (request.isStreaming()) {
            // Collect streaming chunks into single response
            return collectStreamedResponse(request);
        } else {
            return doNonStreamingInfer(request);
        }
    }

    /**
     * Provider-specific streaming implementation
     */
    protected abstract Multi<StreamChunk> doStream(ProviderRequest request);

    /**
     * Provider-specific non-streaming implementation
     */
    protected abstract Uni<InferenceResponse> doNonStreamingInfer(ProviderRequest request);

    /**
     * Collect streamed chunks into single response
     */
    protected Uni<InferenceResponse> collectStreamedResponse(ProviderRequest request) {
        StringBuilder content = new StringBuilder();
        AtomicInteger chunkCount = new AtomicInteger();

        long startTime = System.currentTimeMillis();

        return doStream(request)
                .onItem().invoke(chunk -> {
                    if (!chunk.isFinal()) {
                        content.append(chunk.getDelta());
                        chunkCount.incrementAndGet();
                    }
                })
                .collect().last()
                .map(lastChunk -> {
                    long duration = System.currentTimeMillis() - startTime;

                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .content(content.toString())
                            .model(request.getModel())
                            .durationMs(duration)
                            .metadata("chunks", chunkCount.get())
                            .metadata("streaming", true)
                            .build();
                });
    }

    /**
     * Create stream handler
     */
    protected abstract StreamHandler createStreamHandler();

    /**
     * Create chunk processor
     */
    protected abstract ChunkProcessor createChunkProcessor();
}