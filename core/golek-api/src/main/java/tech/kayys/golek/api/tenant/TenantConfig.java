package tech.kayys.golek.api.tenant;

public record TenantConfig(
        TenantId tenantId,
        String userId,
        String sessionId,
        String runId,
        String traceId,
        String originNodeId,
        String originRunId,
        int attempt,
        int maxAttempts,
        boolean dryRun,
        boolean debugMode) {

    public static TenantConfig create(String tenantId, String userId, String sessionId) {
        return new TenantConfig(
                new TenantId(tenantId),
                userId,
                sessionId,
                null,
                null,
                null,
                null,
                0,
                3,
                false,
                false);
    }
}
