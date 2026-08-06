package tech.kayys.gollek.safetensor.engine.backend;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * Backend-owned resumable execution artifact.
 *
 * <p>This extends a reusable session descriptor with the backend-specific state
 * needed for actual restore attempts, such as KV snapshots, prefix-cache
 * handles, or future backend-resident prompt/session identifiers. Making this
 * explicit separates "reusable in principle" from "restorable in practice";
 * see Shkolnikov (2026), <a href="https://arxiv.org/abs/2603.04428">arXiv:2603.04428</a>.
 */
public record ResumableSessionArtifact(
        ResumableSessionDescriptor descriptor,
        Kind kind,
        String artifactKey,
        Instant capturedAt,
        boolean resumeCapable,
        Map<String, Object> metadata) {

    public enum Kind {
        DESCRIPTOR_ONLY,
        PREFIX_CACHE_HANDLE,
        KV_SNAPSHOT,
        BACKEND_SESSION_HANDLE
    }

    public ResumableSessionArtifact {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(artifactKey, "artifactKey");
        Objects.requireNonNull(capturedAt, "capturedAt");
        metadata = metadata != null ? Map.copyOf(metadata) : Map.of();
    }

    public static ResumableSessionArtifact descriptorOnly(ResumableSessionDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        return new ResumableSessionArtifact(
                descriptor,
                Kind.DESCRIPTOR_ONLY,
                descriptor.sessionKey(),
                Instant.now(),
                false,
                Map.of());
    }

    public static ResumableSessionArtifact backendSessionHandle(
            ResumableSessionDescriptor descriptor,
            String artifactKey,
            Instant capturedAt,
            Map<String, Object> metadata) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(artifactKey, "artifactKey");
        Objects.requireNonNull(capturedAt, "capturedAt");
        return new ResumableSessionArtifact(
                descriptor,
                Kind.BACKEND_SESSION_HANDLE,
                artifactKey,
                capturedAt,
                true,
                metadata);
    }

    public static ResumableSessionArtifact kvSnapshot(
            ResumableSessionDescriptor descriptor,
            String artifactKey,
            Instant capturedAt,
            Map<String, Object> metadata) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(artifactKey, "artifactKey");
        Objects.requireNonNull(capturedAt, "capturedAt");
        return new ResumableSessionArtifact(
                descriptor,
                Kind.KV_SNAPSHOT,
                artifactKey,
                capturedAt,
                true,
                metadata);
    }

    public String backendId() {
        return descriptor.backendId();
    }

    public String sessionKey() {
        return descriptor.sessionKey();
    }

    public String promptFingerprint() {
        return descriptor.promptFingerprint();
    }

    public boolean reusable() {
        return descriptor.reusable();
    }
}
