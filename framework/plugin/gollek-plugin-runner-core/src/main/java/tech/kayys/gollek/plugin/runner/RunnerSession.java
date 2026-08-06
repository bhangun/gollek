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
import tech.kayys.alkhawarizm.spi.model.ModelInfo;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.KVCacheState;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

import java.util.Map;
import java.util.Optional;

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

    /**
     * Get the current state of the KV cache.
     * 
     * @return KV cache state
     */
    KVCacheState getKVCacheState();

    /**
     * Offload a portion of the KV cache to CPU/NVMe to free VRAM.
     */
    void offloadCache();

    // -----------------------------------------------------------------------
    // LoRA / Adapter Hot-Swapping
    // -----------------------------------------------------------------------

    /**
     * Loads a LoRA adapter into the currently running model session.
     *
     * <p>The adapter is identified by an {@code adapterId} that maps to a
     * weights file registered in the adapter registry.  If an adapter is
     * already loaded, it is automatically unloaded before loading the new one
     * (i.e., only one adapter can be active at a time per session).
     *
     * <p>Implementations must be thread-safe: concurrent inference requests
     * using the session should be drained or paused before swapping the adapter.
     *
     * @param adapterId  unique ID of the adapter (e.g. "finance-lora-v2")
     * @param adapterPath file-system path or URI to the adapter weights
     * @throws UnsupportedOperationException if the underlying runner does not
     *         support LoRA hot-swapping
     */
    void loadAdapter(String adapterId, String adapterPath);

    /**
     * Unloads the currently active LoRA adapter, reverting the session to
     * the base model weights.
     *
     * <p>A no-op if no adapter is currently loaded.
     */
    void unloadAdapter();

    /**
     * Returns the ID of the currently loaded LoRA adapter, if any.
     *
     * @return adapter ID, or {@link Optional#empty()} if the base model
     *         is active with no adapter
     */
    Optional<String> getLoadedAdapterId();

    /**
     * Returns {@code true} if this runner session supports LoRA adapter
     * hot-swapping without session restart.
     */
    default boolean supportsAdapterHotSwap() {
        return false;
    }
}
