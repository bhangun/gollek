package tech.kayys.gollek.safetensor.engine.backend;

import java.util.Objects;

/**
 * Result of acquiring an execution session for a prepared generation.
 *
 * <p>This makes the open/restore decision explicit: the engine can distinguish
 * a fresh session, a restored reusable session, and a fallback to fresh
 * execution when reuse metadata existed but the backend could not actually
 * resume state.
 */
public record TextExecutionSessionAcquisition(
        TextExecutionSession session,
        TextExecutionSessionReservation reservation,
        Mode mode,
        String rationale) {

    public enum Mode {
        FRESH,
        RESUMED,
        FALLBACK_FRESH
    }

    public TextExecutionSessionAcquisition {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(rationale, "rationale");
    }

    public static TextExecutionSessionAcquisition fresh(
            TextExecutionSession session,
            TextExecutionSessionReservation reservation,
            String rationale) {
        return new TextExecutionSessionAcquisition(session, reservation, Mode.FRESH, rationale);
    }

    public static TextExecutionSessionAcquisition resumed(
            TextExecutionSession session,
            TextExecutionSessionReservation reservation,
            String rationale) {
        return new TextExecutionSessionAcquisition(session, reservation, Mode.RESUMED, rationale);
    }

    public static TextExecutionSessionAcquisition fallbackFresh(
            TextExecutionSession session,
            TextExecutionSessionReservation reservation,
            String rationale) {
        return new TextExecutionSessionAcquisition(session, reservation, Mode.FALLBACK_FRESH, rationale);
    }

    public boolean resumed() {
        return mode == Mode.RESUMED;
    }
}
