/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelArchitecture.java
 * ───────────────────────
 * SPI interface for model architecture definitions.
 */
package tech.kayys.gollek.spi.model;

import java.util.List;

/**
 * Model architecture SPI — defines weight naming and architecture-specific
 * behavior.
 *
 * <p>
 * Implementations are auto-discovered via CDI
 * {@code Instance<ModelArchitecture>}.
 * </p>
 */
public interface ModelArchitecture {

    /**
     * Canonical architecture ID: "llama", "mistral", "gemma", "phi", "qwen2", etc.
     */
    String id();

    /**
     * Supported architecture class names (from config.json "architectures" array).
     * Example: ["LlamaForCausalLM", "MistralForCausalLM"].
     */
    List<String> supportedArchClassNames();

    /**
     * Supported model_type values (from config.json "model_type").
     * Example: ["llama", "mistral"].
     */
    List<String> supportedModelTypes();

    // ── Weight name resolution ────────────────────────────────────────────────

    /** Full tensor name of the token embedding table. */
    String embedTokensWeight();

    /** Full tensor name of the final layer norm weight. */
    String finalNormWeight();

    /** Full tensor name of the final layer norm bias (or null if none). */
    default String finalNormBias() {
        return null;
    }

    /** Full tensor name of the LM head weight (null if tied to embed_tokens). */
    default String lmHeadWeight() {
        return null;
    }

    /** Full tensor name of the query projection weight for layer {@code i}. */
    String layerQueryWeight(int i);

    /** Full tensor name of the key projection weight for layer {@code i}. */
    String layerKeyWeight(int i);

    /** Full tensor name of the value projection weight for layer {@code i}. */
    String layerValueWeight(int i);


    /** Full tensor name of the query projection bias (null if none). */
    default String layerQueryBias(int i) { return null; }

    /** Full tensor name of the key projection bias (null if none). */
    default String layerKeyBias(int i) { return null; }

    /** Full tensor name of the value projection bias (null if none). */
    default String layerValueBias(int i) { return null; }


    /**
     * Whether Q/K/V projections are fused into a single "qkv" tensor
     * (Falcon, old GPT-NeoX, some Phi variants).
     *
     * <p>
     * When {@code true}, use {@link #layerFusedQKVWeight(int)} instead of
     * the separate Q/K/V methods.
     */
    default boolean hasFusedQKV() {
        return false;
    }

    /**
     * Full tensor name of the fused QKV weight (only when {@link #hasFusedQKV()}).
     */
    default String layerFusedQKVWeight(int i) {
        return null;
    }

    /** Full tensor name of the output projection weight for layer {@code i}. */
    String layerOutputWeight(int i);

    /** Full tensor name of the output projection bias (null if none). */
    default String layerOutputBias(int i) { return null; }


    /** Full tensor name of the attention layer-norm weight for layer {@code i}. */
    String layerAttentionNormWeight(int i);

    /** Full tensor name of the attention layer-norm bias (null if absent). */
    default String layerAttentionNormBias(int i) {
        return null;
    }

    /** Full tensor name of the FFN gate projection weight (SwiGLU / GeGLU). */
    String layerFfnGateWeight(int i);

    /** Full tensor name of the FFN up projection weight. */
    String layerFfnUpWeight(int i);

    /** Full tensor name of the FFN down projection weight. */
    String layerFfnDownWeight(int i);


    /** Full tensor name of the FFN gate projection bias (null if none). */
    default String layerFfnGateBias(int i) { return null; }

    /** Full tensor name of the FFN up projection bias (null if none). */
    default String layerFfnUpBias(int i) { return null; }

    /** Full tensor name of the FFN down projection bias (null if none). */
    default String layerFfnDownBias(int i) { return null; }


    /** Full tensor name of the post-attention (FFN) layer-norm weight. */
    String layerFfnNormWeight(int i);

    /** Full tensor name of the post-attention layer-norm bias (null if absent). */
    default String layerFfnNormBias(int i) {
        return null;
    }

    /** Full tensor name of the Q-norm weight (null if none, e.g. Qwen 2.5). */
    default String layerQueryNormWeight(int i) { return null; }

    /** Full tensor name of the K-norm weight (null if none, e.g. Qwen 2.5). */
    default String layerKeyNormWeight(int i) { return null; }

    // ── Architecture properties ───────────────────────────────────────────────

    /** Whether this architecture uses RMSNorm (true) or LayerNorm (false). */
    default boolean usesRmsNorm() {
        return true;
    }

    /**
     * Whether this architecture has a separate gate projection in the FFN
     * (i.e. SwiGLU / GeGLU gate×up structure).
     */
    default boolean hasSeparateGateProjection() {
        return true;
    }

    /**
     * Check if embeddings are tied with LM head.
     */
    default boolean hasTiedEmbeddings() {
        return false;
    }

    /**
     * RMSNorm epsilon from config.
     */
    default double rmsNormEps() {
        return 1e-6;
    }

    /**
     * Check if this architecture has sliding window attention.
     */
    default boolean hasSlidingWindow() {
        return false;
    }

    /**
     * Get sliding window size (if applicable).
     */
    default int slidingWindowSize() {
        return Integer.MAX_VALUE;
    }
}
