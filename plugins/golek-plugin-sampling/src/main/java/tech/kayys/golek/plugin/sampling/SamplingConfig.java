/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.golek.plugin.sampling;

import java.util.List;

/**
 * Sampling configuration record.
 * <p>
 * Encapsulates all sampling parameters that Golek (policy layer) passes
 * to the inference backend (llama.cpp / LiteRT / remote LLM) for execution.
 *
 * @param temperature       controls randomness (0.0 = deterministic, 2.0 = very random)
 * @param topK              number of top tokens to consider (0 = disabled)
 * @param topP              cumulative probability threshold (nucleus sampling)
 * @param repetitionPenalty penalty for repeating tokens (1.0 = no penalty)
 * @param presencePenalty   penalty for tokens already in context (0.0 = no penalty)
 * @param maxTokens         maximum output tokens
 * @param stopTokens        list of stop sequences
 * @param grammarMode       grammar/JSON constraint mode (null = unconstrained)
 */
public record SamplingConfig(
        double temperature,
        int topK,
        double topP,
        double repetitionPenalty,
        double presencePenalty,
        int maxTokens,
        List<String> stopTokens,
        String grammarMode) {

    /**
     * Create a default sampling config.
     */
    public static SamplingConfig defaults() {
        return new SamplingConfig(0.7, 40, 0.95, 1.1, 0.0, 2048, List.of(), null);
    }

    /**
     * Create a deterministic config (temperature = 0).
     */
    public static SamplingConfig deterministic() {
        return new SamplingConfig(0.0, 1, 1.0, 1.0, 0.0, 2048, List.of(), null);
    }

    /**
     * Create a config optimized for creative text generation.
     */
    public static SamplingConfig creative() {
        return new SamplingConfig(1.2, 100, 0.9, 1.3, 0.5, 4096, List.of(), null);
    }

    /**
     * Create a config that forces JSON output.
     */
    public static SamplingConfig jsonMode() {
        return new SamplingConfig(0.0, 1, 1.0, 1.0, 0.0, 2048, List.of(), "json");
    }
}
