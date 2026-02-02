package tech.kayys.golek.api.context;

import java.time.Duration;
import java.util.Optional;

import tech.kayys.golek.api.model.DeviceType;

public record RequestContext(
        String tenantId,
        String userId,
        String sessionId,
        String runId,
        String traceId,
        String originNodeId,
        String originRunId,
        int attempt,
        int maxAttempts,
        boolean dryRun,
        boolean debugMode,
        Optional<DeviceType> preferredDevice,
        Duration timeout,
        boolean costSensitive

) {

    public static RequestContext create(String tenantId, String userId, String sessionId) {
        return new RequestContext(
                tenantId,
                userId,
                sessionId,
                null,
                null,
                null,
                null,
                0,
                3,
                false,
                false,
                Optional.empty(),
                Duration.ofMinutes(1),
                false);
    }
}
