package tech.kayys.gollek.safetensor.engine.planning;

import java.util.Objects;

/**
 * Planned prompt reuse policy for a prepared prompt.
 *
 * <p>Today Gollek only plans exact-prompt reuse, but making this explicit now
 * gives future backends a typed place to implement prefix caching rather than
 * smuggling reuse decisions through raw strings or ad hoc metadata.
 */
public record PromptReusePlan(
        Strategy strategy,
        String reuseKey,
        int preparedChars,
        String rationale) {

    public enum Strategy {
        NONE,
        EXACT_PROMPT
    }

    public PromptReusePlan {
        Objects.requireNonNull(strategy, "strategy");
        Objects.requireNonNull(rationale, "rationale");
    }

    public static PromptReusePlan none(String rationale) {
        return new PromptReusePlan(Strategy.NONE, null, 0, rationale);
    }

    public static PromptReusePlan exactPrompt(String reuseKey, int preparedChars) {
        Objects.requireNonNull(reuseKey, "reuseKey");
        return new PromptReusePlan(
                Strategy.EXACT_PROMPT,
                reuseKey,
                preparedChars,
                "Exact prepared-prompt reuse candidate");
    }

    public boolean cacheable() {
        return strategy != Strategy.NONE && reuseKey != null && !reuseKey.isBlank();
    }
}
