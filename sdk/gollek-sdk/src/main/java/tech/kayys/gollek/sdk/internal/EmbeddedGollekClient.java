package tech.kayys.gollek.sdk.internal;

import tech.kayys.gollek.sdk.api.GollekClient;
import tech.kayys.gollek.spi.inference.InferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.ModelInfo;
import io.smallrye.mutiny.Multi;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

public class EmbeddedGollekClient implements GollekClient {

    private final InferenceEngine inferenceService;
    private final String defaultModel;
    private final String backend;
    private final int maxTokens;
    private final float temperature;

    public EmbeddedGollekClient(InferenceEngine inferenceService, String defaultModel, String backend, int maxTokens, float temperature) {
        this.inferenceService = inferenceService;
        this.defaultModel = defaultModel;
        this.backend = backend;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
    }

    @Override
    public GenerationResult generate(String prompt) {
        return generate(GenerationRequest.of(prompt));
    }

    @Override
    public GenerationResult generate(GenerationRequest request) {
        InferenceRequest spiRequest = mapToSpiRequest(request);
        InferenceResponse spiResponse = inferenceService.infer(spiRequest).await().indefinitely();
        return mapToSdkResponse(spiResponse, System.currentTimeMillis()); // simplified duration
    }

    @Override
    public GenerationStream generateStream(String prompt) {
        InferenceRequest spiRequest = mapToSpiRequest(GenerationRequest.of(prompt));
        Multi<StreamingInferenceChunk> multi = inferenceService.stream(spiRequest);
        
        EmbeddedGenerationStream stream = new EmbeddedGenerationStream();
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
    public List<GenerationResult> generateBatch(List<String> prompts) {
        return prompts.stream().map(this::generate).toList();
    }

    @Override
    public float[] embed(String text) {
        var spiRequest = tech.kayys.gollek.spi.embedding.EmbeddingRequest.builder()
                .model(defaultModel)
                .input(text)
                .build();
        var spiResponse = inferenceService.executeEmbedding(defaultModel, spiRequest).await().indefinitely();
        return (spiResponse.embeddings() != null && !spiResponse.embeddings().isEmpty()) ? spiResponse.embeddings().get(0) : new float[0];
    }

    @Override
    public List<float[]> embedBatch(List<String> texts) {
        return texts.stream().map(this::embed).toList();
    }

    @Override
    public ModelInfo modelInfo() {
        return ModelInfo.builder().modelId(defaultModel).format(backend).build();
    }

    @Override
    public boolean supports(Feature feature) {
        return true;
    }

    @Override
    public void close() {
        // No-op for embedded
    }

    private InferenceRequest mapToSpiRequest(GenerationRequest request) {
        return InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(defaultModel)
                .prompt(request.prompt())
                .temperature(request.temperature() > 0 ? request.temperature() : temperature)
                .maxTokens(request.maxTokens() > 0 ? request.maxTokens() : maxTokens)
                .streaming(false)
                .build();
    }

    private GenerationResult mapToSdkResponse(InferenceResponse spiResponse, long startTime) {
        return new GenerationResult(
                spiResponse.getContent(),
                spiResponse.getOutputTokens(),
                spiResponse.getInputTokens(),
                System.currentTimeMillis() - startTime
        );
    }
}
