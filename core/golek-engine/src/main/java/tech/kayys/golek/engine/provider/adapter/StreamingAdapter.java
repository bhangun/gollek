package tech.kayys.golek.engine.provider.adapter;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import java.util.Map;

import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.provider.ProviderRequest;
import tech.kayys.golek.spi.provider.StreamingProvider;
import tech.kayys.golek.spi.stream.StreamChunk;

import tech.kayys.golek.provider.core.streaming.ChunkProcessor;
import tech.kayys.golek.provider.core.streaming.StreamHandler;

import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Adapter that adds streaming capabilities to providers
 */
public abstract class StreamingAdapter extends AbstractProvider implements StreamingProvider {

    protected volatile StreamHandler streamHandler;
    protected volatile ChunkProcessor chunkProcessor;

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request) {
        if (!isInitialized()) {
            return Multi.createFrom().failure(
                    new IllegalStateException("Provider not initialized"));
        }

        String tenantId = resolveTenantId(request);

        return checkQuota(tenantId)
                .chain(() -> checkRateLimit(tenantId))
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
        AtomicReference<StringBuilder> content = new AtomicReference<>(new StringBuilder());
        AtomicInteger chunkCount = new AtomicInteger();
        AtomicLong startTime = new AtomicLong(System.currentTimeMillis());

        return doStream(request)
                .onItem().invoke(chunk -> {
                    if (!chunk.isFinal()) {
                        content.get().append(chunk.getDelta());
                        chunkCount.incrementAndGet();
                    }
                })
                .collect().last()
                .map(lastChunk -> {
                    long duration = System.currentTimeMillis() - startTime.get();

                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .content(content.get().toString())
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

    @Override
    protected Uni<Void> doInitialize(Map<String, Object> config) {
        this.streamHandler = createStreamHandler();
        this.chunkProcessor = createChunkProcessor();
        return Uni.createFrom().voidItem();
    }
}
