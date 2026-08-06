package tech.kayys.gollek.safetensor.engine.backend;

import java.util.Objects;

/**
 * Planned artifact strategy for one prepared generation.
 *
 * <p>This separates execution-state policy from backend implementation detail:
 * orchestration can decide whether a request should stay ephemeral, publish an
 * in-process prepared-generation handle, probe a prefix cache, or later target
 * a persisted KV snapshot. The strategy split follows the same edge-inference
 * rationale as recent prompt/prefix reuse work on Apple Silicon; see Barrios
 * (2026), <a href="https://arxiv.org/abs/2601.19139">arXiv:2601.19139</a>.
 */
public record TextExecutionArtifactPlan(
        Strategy strategy,
        boolean resumeTarget,
        String artifactKey,
        String rationale) {

    public enum Strategy {
        NONE,
        PROCESS_LOCAL_PRETOKENIZED_HANDLE,
        PREFIX_CACHE,
        KV_SNAPSHOT
    }

    public TextExecutionArtifactPlan {
        Objects.requireNonNull(strategy, "strategy");
        Objects.requireNonNull(rationale, "rationale");
    }

}
