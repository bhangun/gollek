package tech.kayys.gollek.safetensor.engine.backend;

import java.time.Instant;
import java.util.Objects;

/**
 * Process-local prepared-generation handle for the direct backend.
 *
 * <p>This is the first concrete resume-capable artifact in the new execution
 * architecture. It does not restore KV state yet, but it does preserve a
 * backend-owned prepared generation handle across requests so the direct
 * backend can reattach to existing prepared execution state.
 */
public record DirectPreparedGenerationHandle(
        String sessionKey,
        String promptFingerprint,
        long[] inputIds,
        PreparedTextGeneration generation,
        Instant capturedAt) {

    public DirectPreparedGenerationHandle {
        Objects.requireNonNull(sessionKey, "sessionKey");
        Objects.requireNonNull(promptFingerprint, "promptFingerprint");
        inputIds = inputIds != null ? inputIds.clone() : new long[0];
        Objects.requireNonNull(generation, "generation");
        Objects.requireNonNull(capturedAt, "capturedAt");
    }
}
