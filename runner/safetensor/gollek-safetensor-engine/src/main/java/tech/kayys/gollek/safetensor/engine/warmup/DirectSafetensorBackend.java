/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.generation.TextInferenceEngine;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderResponse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.concurrent.Flow;

/**
 * Direct Safetensor Backend for SafetensorProvider.
 * Delegates requests to the internal native inference engines.
 */
@ApplicationScoped
public class DirectSafetensorBackend {

    private static final Logger log = Logger.getLogger(DirectSafetensorBackend.class);
    private static final String PROVIDER_ID = "safetensor-direct";

    @Inject
    DirectInferenceEngine engine;

    @Inject
    TextInferenceEngine textEngine;

    @Inject
    SafetensorProviderConfig config;

    public void initialize(ProviderConfig providerConfig) {
        log.infof("DirectSafetensorBackend: initialised with FFM memory mapped weights");
    }

    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.builder()
                .status(ProviderHealth.Status.HEALTHY)
                .message("Direct Safetensor backend active (zero-copy enabled)")
                .timestamp(Instant.now())
                .build());
    }

    public Uni<InferenceResponse> infer(ProviderRequest request) {
        // For non-streaming, collect everything from the stream until end
        return Multi.createFrom().publisher(streamToPublisher(request))
                .collect().asList()
                .map(chunks -> {
                    StringBuilder fullText = new StringBuilder();
                    for (ProviderResponse r : chunks) {
                        String delta = r.getContent();
                        if (delta != null)
                            fullText.append(delta);
                    }
                    return InferenceResponse.builder()
                            .requestId("chatcmpl-" + request.getRequestId())
                            .model(request.getModel())
                            .content(fullText.toString())
                            .tokensUsed(0)
                            .inputTokens(0)
                            .outputTokens(0)
                            .durationMs(0L)
                            .build();
                });
    }

    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        java.util.concurrent.atomic.AtomicInteger idx = new java.util.concurrent.atomic.AtomicInteger(0);
        return Multi.createFrom().publisher(streamToPublisher(request))
                .map(response -> {
                    boolean isFinal = "stop".equals(response.getFinishReason());
                    int index = idx.getAndIncrement();
                    if (isFinal) {
                        return StreamingInferenceChunk.finalChunk(response.getRequestId(), index, response.getContent());
                    } else {
                        return StreamingInferenceChunk.of(response.getRequestId(), index, response.getContent());
                    }
                });
    }

    private Flow.Publisher<ProviderResponse> streamToPublisher(ProviderRequest request) {
        Path modelPath = resolveModelPath(request.getModel());
        try {
            // Load the model zero-copy into native memory Arena
            String loadedKey = engine.loadModel(modelPath);

            // Extract prompt from messages
            String prompt = request.getMessages().stream()
                    .filter(m -> m.getRole() == Message.Role.USER)
                    .map(Message::getContent)
                    .reduce((first, second) -> second)
                    .orElse("");

            GenerationConfig genCfg = GenerationConfig.builder()
                    .maxNewTokens(request.getMaxTokens())
                    .temperature((float) request.getTemperature())
                    .topP((float) request.getTopP())
                    .build();

            // Generate response lazily
            return textEngine.generateStream(prompt, modelPath, genCfg)
                    .map(infResp -> ProviderResponse.builder()
                            .requestId(infResp.getRequestId())
                            .content(infResp.getContent())
                            .model(infResp.getModel())
                            .finishReason(infResp.getFinishReason() != null ? infResp.getFinishReason().name().toLowerCase() : "stop")
                            .promptTokens(infResp.getInputTokens())
                            .completionTokens(infResp.getOutputTokens())
                            .totalTokens(infResp.getTokensUsed())
                            .durationMs(infResp.getDurationMs())
                            .build());
        } catch (Exception e) {
            log.errorf(e, "Direct inference failed for model %s (resolved path: %s)",
                    request.getModel(), modelPath);
            throw new ProviderException("Direct inference failed for model "
                    + request.getModel() + ": " + e.getMessage(), e);
        }
    }

    /**
     * Resolve a model ID (e.g. "Qwen/Qwen2.5-0.5B-Instruct") to an absolute path
     * under the configured base directory (~/.gollek/models/safetensors/).
     */
    private Path resolveModelPath(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            throw new ProviderException("Model ID is null or blank");
        }

        Path asPath = Path.of(modelId);

        // If the model ID is already an absolute path, use it directly
        if (asPath.isAbsolute() && Files.exists(asPath)) {
            return asPath;
        }

        // Resolve relative to the configured base path
        Path resolved = Path.of(config.basePath(), modelId);
        if (Files.exists(resolved)) {
            log.debugf("Resolved model '%s' to path: %s", modelId, resolved);
            return resolved;
        }

        // Fall back to the as-given path (will likely fail but gives a clear error)
        log.warnf("Model path not found at %s, using as-is: %s", resolved, modelId);
        return asPath;
    }
}
