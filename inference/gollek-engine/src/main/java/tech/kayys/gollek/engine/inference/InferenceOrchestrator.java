package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

public interface InferenceOrchestrator {

    Uni<InferenceResponse> executeAsync(String modelId, InferenceRequest request);

    Uni<InferenceResponse> execute(String modelId, InferenceRequest request);

    Multi<StreamingInferenceChunk> streamExecute(String modelId, InferenceRequest request);

    Uni<EmbeddingResponse> executeEmbedding(String modelId,
            EmbeddingRequest request);

    Uni<Void> initialize();

    Uni<Void> shutdown();

}
