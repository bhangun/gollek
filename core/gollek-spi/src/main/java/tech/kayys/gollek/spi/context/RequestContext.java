package tech.kayys.gollek.spi.context;

import java.time.Duration;
import java.util.Optional;

import tech.kayys.gollek.spi.auth.ApiKeyConstants;
import tech.kayys.gollek.spi.model.DeviceType;

public record RequestContext(
        @Deprecated String apiKey,
        String userId,
        String sessionId,
        String tenantId,
        String requestId,
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

    public static final String COMMUNITY_TENANT_ID = "community";

    public String apiKey() {
        if (apiKey == null || apiKey.isBlank()) {
            return ApiKeyConstants.COMMUNITY_API_KEY;
        }
        return apiKey;
    }

    /**
     * Backward-compatible JavaBean getter aliases for legacy code paths that still
     * call getXxx() after migration from RequestContext.
     */
    public String getApiKey() {
        return apiKey();
    }

    public String getUserId() {
        return userId;
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getTenantId() {
        return tenantId;
    }

    public String getRequestId() {
        return resolveRequestId(requestId);
    }

    public String getRunId() {
        return runId;
    }

    public String getTraceId() {
        return traceId;
    }

    public int getAttempt() {
        return attempt;
    }

    public int getMaxAttempts() {
        return maxAttempts;
    }

    public boolean isDryRun() {
        return dryRun;
    }

    public boolean isDebugMode() {
        return debugMode;
    }

    public Optional<DeviceType> getPreferredDevice() {
        return preferredDevice;
    }

    public Duration getTimeout() {
        return timeout;
    }

    public boolean isCostSensitive() {
        return costSensitive;
    }

    public static RequestContext createApiKey(String apiKey, String userId, String sessionId) {
        return create(apiKey, userId, sessionId);
    }

    public static RequestContext create(String apiKey, String userId, String sessionId) {
        return new RequestContext(
                apiKey,
                userId,
                sessionId,
                COMMUNITY_TENANT_ID,
                COMMUNITY_TENANT_ID + "-" + System.currentTimeMillis(),
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

    /**
     * Community-mode context with explicit request id.
     * Tenant remains "community" unless set explicitly via forTenant().
     */
    public static RequestContext of(String requestId) {
        return new RequestContext(
                null,
                null,
                null,
                COMMUNITY_TENANT_ID,
                resolveRequestId(requestId),
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

    /**
     * Enterprise-friendly context where tenant is explicit and independent from
     * request id.
     */
    public static RequestContext forTenant(String tenantId, String requestId) {
        return new RequestContext(
                null,
                null,
                null,
                tenantId != null ? tenantId : COMMUNITY_TENANT_ID,
                resolveRequestId(requestId),
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

    private static String resolveRequestId(String requestId) {
        return (requestId == null || requestId.isBlank()) ? "req-" + System.currentTimeMillis() : requestId.trim();
    }
}
