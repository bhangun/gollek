package tech.kayys.gollek.safetensor.engine.backend;

import java.time.Instant;
import java.util.Objects;

/**
 * Stable descriptor for a session that may be resumed or reused later.
 *
 * <p>This does not imply persistence exists yet. It makes the resumability
 * contract explicit so future backends can attach KV snapshots, prefix-cache
 * handles, or persisted session manifests without changing the planner and
 * metadata model again.
 */
public record ResumableSessionDescriptor(
        String backendId,
        String sessionKey,
        String promptFingerprint,
        TextExecutionSessionPlan.ReusePolicy reusePolicy,
        Instant createdAt,
        String rationale) {

    public ResumableSessionDescriptor {
        Objects.requireNonNull(backendId, "backendId");
        Objects.requireNonNull(sessionKey, "sessionKey");
        Objects.requireNonNull(promptFingerprint, "promptFingerprint");
        Objects.requireNonNull(reusePolicy, "reusePolicy");
        Objects.requireNonNull(createdAt, "createdAt");
        Objects.requireNonNull(rationale, "rationale");
    }

    public boolean reusable() {
        return reusePolicy != TextExecutionSessionPlan.ReusePolicy.EPHEMERAL;
    }
}
