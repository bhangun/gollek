package tech.kayys.gollek.safetensor.engine.session;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.safetensor.engine.backend.ResumableSessionArtifact;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionControl;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionManager;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionRegistry;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionSessionSnapshot;

import java.util.List;
import java.util.Optional;

/**
 * Typed inspection API for reusable text execution sessions.
 *
 * <p>This layer gives tooling, orchestration, and future admin surfaces a
 * stable engine-facing view over session state without importing backend
 * control records directly.
 */
@ApplicationScoped
public class TextExecutionSessionInspector {

    @Inject
    TextExecutionSessionControl sessionControl;

    public TextExecutionSessionInspector() {
    }

    public TextExecutionSessionInspector(TextExecutionSessionControl sessionControl) {
        this.sessionControl = sessionControl;
    }

    public TextExecutionSessionInventory inventory() {
        TextExecutionSessionSnapshot snapshot = control().snapshot();
        return new TextExecutionSessionInventory(
                snapshot.capturedAt(),
                control().activeArtifacts().stream()
                        .map(TextExecutionSessionView::from)
                        .toList(),
                snapshot.reservedSessionKeys().stream()
                        .map(TextExecutionReservationView::from)
                        .toList());
    }

    public List<TextExecutionSessionView> activeReusableSessions() {
        return inventory().reusableSessions();
    }

    public Optional<TextExecutionSessionView> inspect(String sessionKey) {
        return control().findArtifactByKey(sessionKey).map(TextExecutionSessionView::from);
    }

    public boolean evict(String sessionKey) {
        return control().evictByKey(sessionKey);
    }

    public int activeReusableSessionCount() {
        return control().activeReusableSessions();
    }

    public int activeReservationCount() {
        return control().activeReservations();
    }

    private TextExecutionSessionControl control() {
        return sessionControl != null
                ? sessionControl
                : new TextExecutionSessionManager(new TextExecutionSessionRegistry());
    }
}
