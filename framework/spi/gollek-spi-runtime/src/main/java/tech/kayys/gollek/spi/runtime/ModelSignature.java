/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.spi.runtime;

import java.util.Objects;

/**
 * Unique architectural signature of an LLM model.
 * Used to prevent "silent killer" bugs by ensuring KV caches are only shared
 * between compatible model configurations.
 * 
 * @param architecture e.g. "llama", "mistral", "qwen"
 * @param numHeads     number of attention heads
 * @param headDim      dimension per head
 * @param ropeTheta    RoPE base frequency (critical for context alignment)
 * @param vocabSize    vocabulary size (for validation)
 */
public record ModelSignature(
    String architecture,
    int numHeads,
    int headDim,
    float ropeTheta,
    int vocabSize
) {
    /**
     * Creates a stable fingerprint hash for the model architecture.
     */
    public String fingerprint() {
        return String.format("%s-h%d-d%d-t%.0f", 
            architecture.toLowerCase(), numHeads, headDim, ropeTheta);
    }

    /**
     * Validates if another signature is compatible for KV sharing.
     */
    public boolean isCompatible(ModelSignature other) {
        if (other == null) return false;
        return Objects.equals(architecture, other.architecture) &&
               numHeads == other.numHeads &&
               headDim == other.headDim &&
               Float.compare(ropeTheta, other.ropeTheta) == 0;
    }
}
