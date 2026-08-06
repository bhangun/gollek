package tech.kayys.gollek.safetensor.engine.backend;

import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * Typed control-plane snapshot of reusable execution sessions.
 */
public record TextExecutionSessionSnapshot(
        Instant capturedAt,
        List<ResumableSessionDescriptor> reusableSessions,
        List<ReservedSessionKey> reservedSessionKeys) {

    public TextExecutionSessionSnapshot {
        Objects.requireNonNull(capturedAt, "capturedAt");
        reusableSessions = List.copyOf(reusableSessions);
        reservedSessionKeys = List.copyOf(reservedSessionKeys);
    }

    public int activeReusableSessions() {
        return reusableSessions.size();
    }

    public int activeReservations() {
        return reservedSessionKeys.size();
    }
}
