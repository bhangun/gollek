package tech.kayys.gollek.safetensor.engine.session;

import tech.kayys.gollek.safetensor.engine.backend.ReservedSessionKey;

import java.time.Instant;
import java.util.Objects;

/**
 * Stable inspection view over a reserved reusable-session key.
 */
public record TextExecutionReservationView(
        String sessionKey,
        Instant reservedAt) {

    public TextExecutionReservationView {
        Objects.requireNonNull(sessionKey, "sessionKey");
        Objects.requireNonNull(reservedAt, "reservedAt");
    }

    public static TextExecutionReservationView from(ReservedSessionKey reservedSessionKey) {
        Objects.requireNonNull(reservedSessionKey, "reservedSessionKey");
        return new TextExecutionReservationView(
                reservedSessionKey.sessionKey(),
                reservedSessionKey.reservedAt());
    }
}
