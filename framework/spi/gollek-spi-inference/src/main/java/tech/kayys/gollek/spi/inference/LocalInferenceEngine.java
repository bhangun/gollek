package tech.kayys.gollek.spi.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.alkhawarizm.spi.model.HealthStatus;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;

public interface LocalInferenceEngine extends InferenceEngine {
    
    @Override
    default Uni<InferenceResponse> executeAsync(String modelId, InferenceRequest request) {
        return infer(request);
    }

    @Override
    default InferenceResponse execute(String modelId, InferenceRequest request) {
        return infer(request).await().indefinitely();
    }

    @Override
    default Multi<StreamingInferenceChunk> streamExecute(String modelId, InferenceRequest request) {
        return stream(request);
    }

    @Override
    default Uni<EmbeddingResponse> executeEmbedding(String modelId, EmbeddingRequest request) {
        throw new UnsupportedOperationException("Embeddings not supported by this engine");
    }

    @Override
    default Uni<String> submitAsyncJob(InferenceRequest request) {
        throw new UnsupportedOperationException("Async jobs not supported");
    }

    @Override
    default boolean isHealthy() {
        return true;
    }

    @Override
    default HealthStatus health() {
        return HealthStatus.healthy("Local Engine OK");
    }

    @Override
    default void initialize() {
        // Default no-op
    }

    @Override
    default EngineStats getStats() {
        return null;
    }
}
