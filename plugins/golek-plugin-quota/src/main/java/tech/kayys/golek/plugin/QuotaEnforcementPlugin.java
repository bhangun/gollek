package tech.kayys.golek.plugin;

import tech.kayys.golek.inference.kernel.plugin.InferencePhasePlugin;
import tech.kayys.golek.inference.kernel.plugin.InferencePhasePlugin.PluginException;
import tech.kayys.golek.inference.kernel.engine.EngineContext;
import tech.kayys.golek.inference.kernel.execution.ExecutionContext;
import tech.kayys.golek.inference.kernel.pipeline.InferencePhase;
import tech.kayys.golek.inference.TenantId;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

/**
 * Enforces tenant-level quotas for inference requests.
 */
@ApplicationScoped
public class QuotaEnforcementPlugin implements InferencePhasePlugin {

    @Inject
    TenantQuotaService quotaService;

    @Override
    public String id() {
        return "tech.kayys.golek.policy.quota";
    }

    @Override
    public int order() {
        return 10; // Execute early in the authorization phase
    }

    @Override
    public void initialize(EngineContext context) {
        // Initialization logic if needed
        System.out.println("Quota Enforcement Plugin initialized");
    }

    @Override
    public void shutdown() {
        // Cleanup logic if needed
        System.out.println("Quota Enforcement Plugin shut down");
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.AUTHORIZE; // Quota enforcement is part of authorization
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        String tenantIdStr = context.tenantContext().getTenantId();
        if (tenantIdStr == null || tenantIdStr.trim().isEmpty()) {
            throw new PluginException("Tenant ID is required for quota enforcement");
        }

        TenantId tenantId = new TenantId(tenantIdStr);

        // Check if tenant has capacity
        QuotaInfo quota = quotaService.checkQuota(tenantId);
        if (!quota.hasCapacity()) {
            throw new PluginException(
                "Tenant " + tenantId.value() + " has exceeded quota: " + quota.getLimit());
        }

        // Reserve quota for this request
        quotaService.reserve(tenantId, 1);

        // Store reservation info for cleanup in later phases
        context.putVariable("quotaReserved", true);
        context.putVariable("reservedQuotaId", quota.getId());
        context.putVariable("reservedTenantId", tenantId.value());
    }
}