package tech.kayys.golek.plugin;

import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.core.plugin.GolekConfigurablePlugin;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginException;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.core.execution.ExecutionContext;
import tech.kayys.golek.spi.inference.InferencePhase;
import tech.kayys.wayang.tenant.RequestId;

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
    public void initialize(PluginContext context) {
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
        RequestId requestId = context.requestContext().getRequestId();
        if (requestId == null) {
            throw new PluginException("Tenant ID is required for quota enforcement");
        }

        // Check if tenant has capacity
        QuotaInfo quota = quotaService.checkQuota(requestId);
        if (!quota.hasCapacity()) {
            throw new PluginException(
                    "Tenant " + requestId.value() + " has exceeded quota: " + quota.getLimit());
        }

        // Reserve quota for this request
        quotaService.reserve(requestId, 1);

        // Store reservation info for cleanup in later phases
        context.putVariable("quotaReserved", true);
        context.putVariable("reservedQuotaId", quota.getId());
        context.putVariable("reservedRequestId", requestId.value());
    }

    @Override
    public void onConfigUpdate(java.util.Map<String, Object> newConfig)
            throws GolekConfigurablePlugin.ConfigurationException {
        // No dynamic config for now
    }

    @Override
    public java.util.Map<String, Object> currentConfig() {
        return java.util.Collections.emptyMap();
    }
}