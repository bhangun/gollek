package tech.kayys.golek.plugin;

import tech.kayys.golek.inference.kernel.plugin.InferencePhasePlugin;
import tech.kayys.golek.inference.kernel.plugin.InferencePhasePlugin.PluginException;
import tech.kayys.golek.inference.kernel.engine.EngineContext;
import tech.kayys.golek.inference.kernel.execution.ExecutionContext;
import tech.kayys.golek.inference.kernel.pipeline.InferencePhase;
import tech.kayys.golek.inference.TenantId;

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
        return "tech.kayys.golek.policy.quota-cleanup";
    }

    @Override
    public int order() {
        return 90; // Run late in the cleanup phase
    }

    @Override
    public void initialize(EngineContext context) {
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
            Optional<String> tenantIdOpt = context.getVariable("reservedTenantId", String.class);
            Optional<String> quotaIdOpt = context.getVariable("reservedQuotaId", String.class);
            
            if (tenantIdOpt.isPresent()) {
                try {
                    TenantId tenantId = new TenantId(tenantIdOpt.get());
                    
                    // Release the quota that was reserved for this request
                    quotaService.release(tenantId, 1);
                    
                    // Clean up the variables we stored
                    context.removeVariable("quotaReserved");
                    context.removeVariable("reservedQuotaId");
                    context.removeVariable("reservedTenantId");
                } catch (Exception e) {
                    // Log the error but don't fail the request
                    // Releasing quota is important but shouldn't break the pipeline
                    System.err.println("Failed to release quota for tenant: " + e.getMessage());
                }
            }
        }
    }
}