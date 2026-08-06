package tech.kayys.gollek.safetensor.engine.session;

import tech.kayys.gollek.safetensor.engine.backend.ResumableSessionArtifact;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionPlan;

import java.time.Instant;
import java.util.Objects;

/**
 * Stable inspection view for a reusable text execution session.
 *
 * <p>This keeps higher-level tooling from depending directly on backend package
 * records while preserving the execution semantics needed for orchestration and
 * observability.
 */
public record TextExecutionSessionView(
        String backendId,
        String sessionKey,
        String promptFingerprint,
        TextExecutionSessionPlan.ReusePolicy reusePolicy,
        Instant createdAt,
        String rationale,
        String artifactKey,
        String artifactKind,
        Instant artifactCapturedAt,
        boolean resumeCapable) {

    public TextExecutionSessionView {
        Objects.requireNonNull(backendId, "backendId");
        Objects.requireNonNull(sessionKey, "sessionKey");
        Objects.requireNonNull(promptFingerprint, "promptFingerprint");
        Objects.requireNonNull(reusePolicy, "reusePolicy");
        Objects.requireNonNull(createdAt, "createdAt");
        Objects.requireNonNull(rationale, "rationale");
        Objects.requireNonNull(artifactKey, "artifactKey");
        Objects.requireNonNull(artifactKind, "artifactKind");
        Objects.requireNonNull(artifactCapturedAt, "artifactCapturedAt");
    }

    public static TextExecutionSessionView from(ResumableSessionArtifact artifact) {
        Objects.requireNonNull(artifact, "artifact");
        return new TextExecutionSessionView(
                artifact.backendId(),
                artifact.sessionKey(),
                artifact.promptFingerprint(),
                artifact.descriptor().reusePolicy(),
                artifact.descriptor().createdAt(),
                artifact.descriptor().rationale(),
                artifact.artifactKey(),
                artifact.kind().name().toLowerCase(),
                artifact.capturedAt(),
                artifact.resumeCapable());
    }

    public boolean reusable() {
        return reusePolicy != TextExecutionSessionPlan.ReusePolicy.EPHEMERAL;
    }
}
