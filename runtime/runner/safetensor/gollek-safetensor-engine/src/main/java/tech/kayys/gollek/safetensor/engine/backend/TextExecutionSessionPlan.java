package tech.kayys.gollek.safetensor.engine.backend;

import java.util.Objects;

/**
 * Planned execution mode for one prepared text generation.
 *
 * <p>This keeps prefill, decode, and reuse policy explicit. Future backends can
 * use the same plan shape to decide whether they should run a full prefill, probe
 * a prefix cache, restore persistent KV state, or keep the session ephemeral.
 *
 * <p>The explicit prefill/reuse split is informed by recent Apple Silicon prefix
 * caching work such as Barrios (2026), arXiv:2601.19139.
 */
public record TextExecutionSessionPlan(
        PrefillStrategy prefillStrategy,
        DecodeStrategy decodeStrategy,
        ReusePolicy reusePolicy,
        Scope scope,
        String sessionKey,
        String rationale) {

    public enum PrefillStrategy {
        FULL_PREFILL,
        PREFIX_REUSE_CANDIDATE
    }

    public enum DecodeStrategy {
        TOKEN_ITERATIVE
    }

    public enum ReusePolicy {
        EPHEMERAL,
        EXACT_PROMPT_REUSABLE,
        CONVERSATION_STATEFUL
    }

    public enum Scope {
        REQUEST_LOCAL,
        PROCESS_LOCAL_EXACT_PROMPT,
        CONVERSATION_STATEFUL
    }

    public TextExecutionSessionPlan {
        Objects.requireNonNull(prefillStrategy, "prefillStrategy");
        Objects.requireNonNull(decodeStrategy, "decodeStrategy");
        Objects.requireNonNull(reusePolicy, "reusePolicy");
        Objects.requireNonNull(scope, "scope");
        Objects.requireNonNull(rationale, "rationale");
    }

    public boolean reusable() {
        return reusePolicy != ReusePolicy.EPHEMERAL && sessionKey != null && !sessionKey.isBlank();
    }

    public boolean conversationStateful() {
        return scope == Scope.CONVERSATION_STATEFUL;
    }
}
