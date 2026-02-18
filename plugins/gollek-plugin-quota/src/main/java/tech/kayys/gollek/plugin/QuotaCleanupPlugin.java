package tech.kayys.gollek.plugin;

import tech.kayys.gollek.core.plugin.InferencePhasePlugin;
import tech.kayys.gollek.spi.plugin.PluginException;
import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.core.execution.ExecutionContext;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.wayang.tenant.RequestId;
import tech.kayys.gollek.core.plugin.GollekConfigurablePlugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.util.Optional;

/**
 * Cleanup plugin to release reserved quotas when inference requests complete.
 * This plugin runs in the CLEANUP phase to ensure quotas are properly released.
 */
@ApplicationScoped
public class QuotaCleanupPlugin implements InferencePhasePlugin {

    @Inject
    TenantQuotaService quotaService;

    @Override
    public String id() {
        return "tech.kayys.gollek.policy.quota-cleanup";
    }

    @Override
    public int order() {
        return 90; // Run late in the cleanup phase
    }

    @Override
    public void initialize(PluginContext context) {
        // Initialization logic if needed
        System.out.println("Quota Cleanup Plugin initialized");
    }

    @Override
    public void shutdown() {
        // Cleanup logic if needed
        System.out.println("Quota Cleanup Plugin shut down");
    }

    @Override
    public InferencePhase phase() {
        return InferencePhase.CLEANUP; // Run in cleanup phase to release resources
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        // Check if quota was reserved during the request
        Boolean quotaReserved = context.getVariable("quotaReserved", Boolean.class).orElse(false);

        if (Boolean.TRUE.equals(quotaReserved)) {
            // Retrieve the tenant ID and quota ID that were stored during reservation
            Optional<String> requestIdOpt = context.getVariable("reservedRequestId", String.class);

            if (requestIdOpt.isPresent()) {
                try {
                    RequestId requestId = RequestId.of(requestIdOpt.get());

                    // Release the quota that was reserved for this request
                    quotaService.release(requestId, 1);

                    // Clean up the variables we stored
                    // Using null to effectively remove if removeVariable is not available
                    context.putVariable("quotaReserved", null);
                    context.putVariable("reservedQuotaId", null);
                    context.putVariable("reservedRequestId", null);
                } catch (Exception e) {
                    // Log the error but don't fail the request
                    // Releasing quota is important but shouldn't break the pipeline
                    System.err.println("Failed to release quota for tenant: " + e.getMessage());
                }
            }
        }
    }

    @Override
    public void onConfigUpdate(java.util.Map<String, Object> newConfig)
            throws GollekConfigurablePlugin.ConfigurationException {
        // No dynamic config for now
    }

    @Override
    public java.util.Map<String, Object> currentConfig() {
        return java.util.Collections.emptyMap();
    }
}