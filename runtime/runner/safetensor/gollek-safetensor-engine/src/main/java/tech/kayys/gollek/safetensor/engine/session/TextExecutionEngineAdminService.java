package tech.kayys.gollek.safetensor.engine.session;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Engine-facing admin surface for reusable text execution state.
 *
 * <p>This builds stable summaries on top of the lower-level inspector so
 * provider health checks, tooling, and future admin endpoints can consume
 * text-execution session state without duplicating grouping logic.
 */
@ApplicationScoped
public class TextExecutionEngineAdminService {

    @Inject
    TextExecutionSessionInspector sessionInspector;

    public TextExecutionEngineAdminService() {
    }

    public TextExecutionEngineAdminService(TextExecutionSessionInspector sessionInspector) {
        this.sessionInspector = sessionInspector;
    }

    public TextExecutionEngineStatus status() {
        TextExecutionSessionInventory inventory = inspector().inventory();
        Map<String, Integer> backendCounts = new LinkedHashMap<>();
        Map<String, Integer> reusePolicyCounts = new LinkedHashMap<>();
        Map<String, Integer> artifactKindCounts = new LinkedHashMap<>();
        int resumeCapableSessions = 0;
        for (TextExecutionSessionView session : inventory.reusableSessions()) {
            backendCounts.merge(session.backendId(), 1, Integer::sum);
            reusePolicyCounts.merge(session.reusePolicy().name().toLowerCase(), 1, Integer::sum);
            artifactKindCounts.merge(session.artifactKind(), 1, Integer::sum);
            if (session.resumeCapable()) {
                resumeCapableSessions++;
            }
        }
        return new TextExecutionEngineStatus(
                inventory.capturedAt(),
                inventory.activeReusableSessions(),
                inventory.activeReservations(),
                resumeCapableSessions,
                backendCounts,
                reusePolicyCounts,
                artifactKindCounts);
    }

    public TextExecutionSessionInventory inventory() {
        return inspector().inventory();
    }

    public Optional<TextExecutionSessionView> inspectSession(String sessionKey) {
        return inspector().inspect(sessionKey);
    }

    public boolean evictSession(String sessionKey) {
        return inspector().evict(sessionKey);
    }

    private TextExecutionSessionInspector inspector() {
        return sessionInspector != null ? sessionInspector : new TextExecutionSessionInspector();
    }
}
