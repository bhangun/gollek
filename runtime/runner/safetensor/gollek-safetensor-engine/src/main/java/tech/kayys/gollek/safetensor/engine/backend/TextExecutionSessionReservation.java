package tech.kayys.gollek.safetensor.engine.backend;

import java.util.Objects;

/**
 * Reservation outcome for a prepared execution session key.
 *
 * <p>This models the lifecycle between a reuse decision and an active session
 * publish step. It lets Gollek distinguish ephemeral sessions, fresh reserved
 * sessions, and sessions that should attach to an existing reusable candidate.
 */
public record TextExecutionSessionReservation(
        Mode mode,
        String sessionKey,
        ResumableSessionDescriptor candidate,
        String rationale) {

    public enum Mode {
        EPHEMERAL,
        RESERVED_NEW,
        ATTACHED_EXISTING,
        CONTENDED
    }

    public TextExecutionSessionReservation {
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(rationale, "rationale");
    }

    public static TextExecutionSessionReservation ephemeral(String rationale) {
        return new TextExecutionSessionReservation(Mode.EPHEMERAL, null, null, rationale);
    }

    public static TextExecutionSessionReservation reservedNew(String sessionKey, String rationale) {
        Objects.requireNonNull(sessionKey, "sessionKey");
        return new TextExecutionSessionReservation(Mode.RESERVED_NEW, sessionKey, null, rationale);
    }

    public static TextExecutionSessionReservation attachedExisting(ResumableSessionDescriptor candidate, String rationale) {
        Objects.requireNonNull(candidate, "candidate");
        return new TextExecutionSessionReservation(Mode.ATTACHED_EXISTING, candidate.sessionKey(), candidate, rationale);
    }

    public static TextExecutionSessionReservation contended(String sessionKey, String rationale) {
        Objects.requireNonNull(sessionKey, "sessionKey");
        return new TextExecutionSessionReservation(Mode.CONTENDED, sessionKey, null, rationale);
    }

    public boolean hasCandidate() {
        return candidate != null;
    }
}
