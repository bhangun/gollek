package tech.kayys.gollek.safetensor.engine.backend;

import java.util.Objects;
import java.util.Optional;

/**
 * Typed reuse decision for a prepared text generation.
 *
 * <p>This keeps the session manager contract explicit: callers do not need to
 * know whether reuse was impossible, simply absent, or actually available.
 */
public record TextExecutionSessionReuseDecision(
        Status status,
        String sessionKey,
        ResumableSessionDescriptor candidate,
        String rationale) {

    public enum Status {
        NOT_REUSABLE,
        MISS,
        HIT
    }

    public TextExecutionSessionReuseDecision {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(rationale, "rationale");
    }

    public static TextExecutionSessionReuseDecision notReusable(String rationale) {
        return new TextExecutionSessionReuseDecision(Status.NOT_REUSABLE, null, null, rationale);
    }

    public static TextExecutionSessionReuseDecision miss(String sessionKey, String rationale) {
        Objects.requireNonNull(sessionKey, "sessionKey");
        return new TextExecutionSessionReuseDecision(Status.MISS, sessionKey, null, rationale);
    }

    public static TextExecutionSessionReuseDecision hit(ResumableSessionDescriptor candidate, String rationale) {
        Objects.requireNonNull(candidate, "candidate");
        return new TextExecutionSessionReuseDecision(Status.HIT, candidate.sessionKey(), candidate, rationale);
    }

    public boolean hasReusableCandidate() {
        return status == Status.HIT && candidate != null;
    }

    public Optional<ResumableSessionDescriptor> candidateOptional() {
        return Optional.ofNullable(candidate);
    }
}
