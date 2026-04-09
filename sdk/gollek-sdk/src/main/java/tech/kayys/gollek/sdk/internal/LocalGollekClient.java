package tech.kayys.gollek.sdk.internal;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.engine.inference.InferenceService;
import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.sdk.model.*;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/**
 * Local (in-process) implementation of {@link GollekClient}.
 *
 * <p>Directly invokes the {@link InferenceService} within the same JVM, bypassing
 * any network layer. This is the preferred mode when the Gollek engine is embedded
 * in the same Quarkus application.
 *
 * <p>The client does not own the engine lifecycle; calling {@link #close()} is a no-op.
 */
public class LocalGollekClient implements GollekClient {

    private final InferenceService inferenceService;
    /** Default model used when the request does not specify one. */
    private final String defaultModel;

    /**
     * Constructs a local client backed by the given inference service.
     *
     * @param inferenceService the in-process inference engine
     * @param defaultModel     fallback model identifier when none is specified in a request
     */
    public LocalGollekClient(InferenceService inferenceService, String defaultModel) {
        this.inferenceService = inferenceService;
        this.defaultModel = defaultModel;
    }

    @Override
    public GenerationResponse generate(GenerationRequest request) {
        return generateAsync(request).join();
    }

    @Override
    public CompletableFuture<GenerationResponse> generateAsync(GenerationRequest request) {
        InferenceRequest spiRequest = mapToSpiRequest(request);
        return inferenceService.inferAsync(spiRequest)
                .onItem().transform(this::mapToSdkResponse)
                .subscribeAsCompletionStage();
    }

    @Override
    public GenerationStream generateStream(GenerationRequest request) {
        InferenceRequest spiRequest = mapToSpiRequest(request);
        Multi<StreamingInferenceChunk> multi = inferenceService.inferStream(spiRequest);

        DefaultGenerationStream stream = new DefaultGenerationStream();
        multi.subscribe().with(
                chunk -> {
                    if (chunk.delta() != null) {
                        stream.emitToken(chunk.delta());
                    }
                },
                stream::emitError,
                stream::emitComplete);
        return stream;
    }

    @Override
    public EmbeddingResponse embed(EmbeddingRequest request) {
        var spiRequest = tech.kayys.gollek.spi.embedding.EmbeddingRequest.builder()
                .model(request.model() != null ? request.model() : defaultModel)
                .input(request.input())
                .build();

        var spiResponse = inferenceService.executeEmbedding(spiRequest).await().indefinitely();

        float[] vector = (spiResponse.embeddings() != null && !spiResponse.embeddings().isEmpty())
                ? spiResponse.embeddings().get(0)
                : new float[0];

        return EmbeddingResponse.builder()
                .vector(vector)
                .model(spiResponse.model())
                .build();
    }

    @Override
    public List<ModelInfo> listModels() {
        return List.of();
    }

    @Override
    public ModelInfo getModelInfo(String modelId) {
        return ModelInfo.builder().id(modelId).build();
    }

    /**
     * Maps an SDK {@link GenerationRequest} to the internal SPI {@link InferenceRequest}.
     * Falls back to {@link #defaultModel} when the request does not specify a model.
     *
     * @param request the SDK-level generation request
     * @return the corresponding SPI inference request
     */
    private InferenceRequest mapToSpiRequest(GenerationRequest request) {
        return InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(request.model() != null ? request.model() : defaultModel)
                .prompt(request.prompt())
                .temperature(request.temperature())
                .maxTokens(request.maxTokens())
                .streaming(request.stream())
                .build();
    }

    /**
     * Maps an SPI {@link InferenceResponse} to the SDK {@link GenerationResponse}.
     *
     * @param spiResponse the raw response from the inference engine
     * @return the SDK-level generation response
     */
    private GenerationResponse mapToSdkResponse(InferenceResponse spiResponse) {
        return GenerationResponse.builder()
                .text(spiResponse.getContent())
                .model(spiResponse.getModel())
                .usage(GenerationResponse.Usage.builder()
                        .promptTokens(spiResponse.getInputTokens())
                        .completionTokens(spiResponse.getOutputTokens())
                        .totalTokens(spiResponse.getTokensUsed())
                        .build())
                .build();
    }

    /**
     * No-op — the local client does not own the engine lifecycle.
     */
    @Override
    public void close() {
        // Local client doesn't own the engine lifecycle
    }
}
