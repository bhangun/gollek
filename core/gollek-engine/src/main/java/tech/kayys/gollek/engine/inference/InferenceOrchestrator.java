package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.stream.StreamChunk;

public interface InferenceOrchestrator {

    Uni<InferenceResponse> executeAsync(String modelId, InferenceRequest request);

    InferenceResponse execute(String modelId, InferenceRequest request);

    Multi<StreamChunk> streamExecute(String modelId, InferenceRequest request);
}
