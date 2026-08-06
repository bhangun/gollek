package tech.kayys.gollek.safetensor.engine.backend;

import java.time.Instant;
import java.util.Objects;

/**
 * Snapshot of a reserved reusable session key.
 */
public record ReservedSessionKey(
        String sessionKey,
        Instant reservedAt) {

    public ReservedSessionKey {
        Objects.requireNonNull(sessionKey, "sessionKey");
        Objects.requireNonNull(reservedAt, "reservedAt");
    }
}
