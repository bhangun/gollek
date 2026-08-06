package tech.kayys.gollek.safetensor.engine.session;

import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * Typed inspection snapshot for reusable text execution state.
 */
public record TextExecutionSessionInventory(
        Instant capturedAt,
        List<TextExecutionSessionView> reusableSessions,
        List<TextExecutionReservationView> reservedSessionKeys) {

    public TextExecutionSessionInventory {
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
