/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.runner.gguf;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * GGUF runner session implementation.
 * 
 * @since 2.1.0
 */
public class GGUFRunnerSession implements RunnerSession {

    private static final Logger LOG = Logger.getLogger(GGUFRunnerSession.class);

    private final String sessionId;
    private final String modelPath;
    private final Map<String, Object> config;
    private final RunnerPlugin runner;
    private final AtomicBoolean active = new AtomicBoolean(true);
    private final ModelInfo modelInfo;

    public GGUFRunnerSession(String modelPath, Map<String, Object> config) {
        this.sessionId = UUID.randomUUID().toString();
        this.modelPath = modelPath;
        this.config = config;
        this.runner = new GGUFRunnerPlugin();

        // Load model and extract info
        this.modelInfo = loadModelInfo(modelPath);

        LOG.infof("Created GGUF session %s for model: %s", sessionId, modelPath);
    }

    @Override
    public String getSessionId() {
        return sessionId;
    }

    @Override
    public String getModelPath() {
        return modelPath;
    }

    @Override
    public RunnerPlugin getRunner() {
        return runner;
    }

    @Override
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        if (!active.get()) {
            return Uni.createFrom().failure(new IllegalStateException("Session is closed"));
        }

        try {
            // Execute inference using llama.cpp binding
            return Uni.createFrom().item(executeInference(request));
        } catch (Exception e) {
            return Uni.createFrom().failure(e);
        }
    }

    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        if (!active.get()) {
            return Multi.createFrom().failure(new IllegalStateException("Session is closed"));
        }

        try {
            // Stream inference using llama.cpp binding
            return executeStreamingInference(request);
        } catch (Exception e) {
            return Multi.createFrom().failure(e);
        }
    }

    public Map<String, Object> getConfig() {
        return config;
    }

    @Override
    public boolean isActive() {
        return active.get();
    }

    @Override
    public void close() {
        if (active.compareAndSet(true, false)) {
            LOG.infof("Closing GGUF session %s", sessionId);
            // Release llama.cpp resources
            releaseModel();
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    private ModelInfo loadModelInfo(String modelPath) {
        // TODO: Load actual model info from GGUF file
        // For now, return placeholder
        return new ModelInfo(
                "placeholder",
                "llama",
                7_000_000_000L, // 7B placeholder
                4096,
                4096,
                Map.of("format", "GGUF", "path", modelPath));
    }

    private InferenceResponse executeInference(InferenceRequest request) {
        // TODO: Implement actual llama.cpp inference
        // This is a placeholder implementation

        return InferenceResponse.builder()
                .requestId(request.getRequestId())
                .content("GGUF inference response (placeholder)")
                .model(modelPath)
                .inputTokens(10)
                .outputTokens(20)
                .tokensUsed(30)
                .build();
    }

    private Multi<StreamingInferenceChunk> executeStreamingInference(InferenceRequest request) {
        // TODO: Implement actual llama.cpp streaming
        // This is a placeholder implementation

        return Multi.createFrom().items(
                StreamingInferenceChunk.of(request.getRequestId(), 0, "Hello "),
                StreamingInferenceChunk.of(request.getRequestId(), 1, "from "),
                StreamingInferenceChunk.of(request.getRequestId(), 2, "GGUF "),
                StreamingInferenceChunk.of(request.getRequestId(), 3, "runner!"),
                StreamingInferenceChunk.finalChunk(request.getRequestId(), 4, ""));
    }

    private void releaseModel() {
        // TODO: Release llama.cpp model resources
        // llama_free_model(ctx);
    }
}
