package tech.kayys.gollek.safetensor.engine.backend;

import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.safetensor.engine.planning.PreparedPrompt;

import java.util.Objects;

/**
 * Backend-prepared text generation handle.
 *
 * <p>This separates model residency from request execution so backends can
 * later attach prompt compilation, prefill caches, backend-native request
 * state, or reusable decode sessions without leaking those details to
 * provider or CLI orchestration.
 *
 * <p>That split is especially important on edge devices where prefill is expensive
 * and execution state may need to survive backend switches or be restored from a
 * persisted cache. See Shkolnikov (2026), arXiv:2603.04428.
 */
public record PreparedTextGeneration(
        String backendId,
        TextExecutionBackend backend,
        PreparedTextModel model,
        PreparedPrompt prompt,
        TextExecutionPreparationPlan preparationPlan,
        GenerationConfig generationConfig) {

    public PreparedTextGeneration {
        Objects.requireNonNull(backendId, "backendId");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(preparationPlan, "preparationPlan");
        Objects.requireNonNull(generationConfig, "generationConfig");
    }

    public String promptFingerprint() {
        return prompt.fingerprint();
    }

    public TextExecutionSessionPlan sessionPlan() {
        return preparationPlan.sessionPlan();
    }

    public TextExecutionArtifactPlan artifactPlan() {
        return preparationPlan.artifactPlan();
    }
}
