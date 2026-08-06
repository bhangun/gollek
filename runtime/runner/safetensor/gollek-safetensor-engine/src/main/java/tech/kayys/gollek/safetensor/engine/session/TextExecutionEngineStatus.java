package tech.kayys.gollek.safetensor.engine.session;

import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * Admin/status summary for reusable text execution state.
 */
public record TextExecutionEngineStatus(
        Instant capturedAt,
        int activeReusableSessions,
        int activeReservations,
        int resumeCapableSessions,
        Map<String, Integer> backendSessionCounts,
        Map<String, Integer> reusePolicyCounts,
        Map<String, Integer> artifactKindCounts) {

    public TextExecutionEngineStatus {
        Objects.requireNonNull(capturedAt, "capturedAt");
        backendSessionCounts = Map.copyOf(backendSessionCounts);
        reusePolicyCounts = Map.copyOf(reusePolicyCounts);
        artifactKindCounts = Map.copyOf(artifactKindCounts);
    }
}
