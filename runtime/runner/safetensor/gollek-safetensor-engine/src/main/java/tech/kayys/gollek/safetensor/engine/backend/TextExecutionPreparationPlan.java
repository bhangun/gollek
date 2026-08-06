package tech.kayys.gollek.safetensor.engine.backend;

import tech.kayys.gollek.safetensor.engine.planning.TextExecutionPreparationContext;

import java.util.Objects;

/**
 * Engine-owned preparation policy for one text generation request.
 *
 * <p>This makes execution planning explicit before a backend is asked to prepare
 * request state. The planner decides how prefill/decode should behave and what
 * kind of resumable artifact is worth targeting; the backend then consumes that
 * policy instead of inventing session behavior internally.
 *
 * <p>This split matches the direction suggested by recent work on resumable
 * edge inference and prompt/prefix reuse, including Shkolnikov (2026),
 * <a href="https://arxiv.org/abs/2603.04428">arXiv:2603.04428</a> and
 * Barrios (2026), <a href="https://arxiv.org/abs/2601.19139">arXiv:2601.19139</a>.
 */
public record TextExecutionPreparationPlan(
        TextExecutionSessionPlan sessionPlan,
        TextExecutionArtifactPlan artifactPlan,
        TextExecutionPreparationContext context,
        String rationale) {

    public TextExecutionPreparationPlan {
        Objects.requireNonNull(sessionPlan, "sessionPlan");
        Objects.requireNonNull(artifactPlan, "artifactPlan");
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(rationale, "rationale");
    }

}
