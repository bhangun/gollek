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

package tech.kayys.gollek.plugin.runner;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

import java.util.Map;

/**
 * Runner session for model inference.
 * 
 * @since 2.1.0
 */
public interface RunnerSession {

    /**
     * Get session ID.
     * 
     * @return Unique session identifier
     */
    String getSessionId();

    /**
     * Get model path.
     * 
     * @return Path to loaded model
     */
    String getModelPath();

    /**
     * Get runner that created this session.
     * 
     * @return Runner plugin
     */
    RunnerPlugin getRunner();

    /**
     * Execute inference request (non-streaming).
     * 
     * @param request Inference request
     * @return Inference response
     */
    Uni<InferenceResponse> infer(InferenceRequest request);

    /**
     * Execute streaming inference request.
     * 
     * @param request Inference request
     * @return Streaming response chunks
     */
    Multi<StreamingInferenceChunk> stream(InferenceRequest request);

    /**
     * Get session configuration.
     * 
     * @return Configuration map
     */
    Map<String, Object> getConfig();

    /**
     * Check if session is active.
     * 
     * @return true if session is active
     */
    boolean isActive();

    /**
     * Close session and release resources.
     */
    void close();

    /**
     * Get model information.
     * 
     * @return Model metadata
     */
    ModelInfo getModelInfo();
}
