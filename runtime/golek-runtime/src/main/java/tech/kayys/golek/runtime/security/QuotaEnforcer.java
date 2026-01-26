package tech.kayys.golek.runtime.security;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.tenant.TenantContext;

import org.jboss.logging.Logger;

/**
 * Enforces tenant quotas and rate limits
 */
@ApplicationScoped
public class QuotaEnforcer {

        private static final Logger LOG = Logger.getLogger(QuotaEnforcer.class);

        @Inject
        QuotaRepository quotaRepository;

        public Uni<Boolean> checkQuota(
                        TenantContext tenantContext,
                        InferenceRequest request) {
                return Uni.createFrom().item(() -> {
                        // Check monthly token quota
                        long currentUsage = quotaRepository.getCurrentMonthlyUsage(
                                        tenantContext.getTenantId());
                        long monthlyLimit = quotaRepository.getMonthlyLimit(
                                        tenantContext.getTenantId());

                        if (currentUsage >= monthlyLimit) {
                                LOG.warnf("Quota exceeded for tenant: %s (%d/%d)",
                                                tenantContext.getTenantId(), currentUsage, monthlyLimit);
                                return false;
                        }

                        // Check rate limit
                        int requestsInWindow = quotaRepository.getRequestsInWindow(
                                        tenantContext.getTenantId(),
                                        60 // 60 seconds window
                        );
                        int rateLimit = quotaRepository.getRateLimit(
                                        tenantContext.getTenantId());

                        if (requestsInWindow >= rateLimit) {
                                LOG.warnf("Rate limit exceeded for tenant: %s (%d/%d)",
                                                tenantContext.getTenantId(), requestsInWindow, rateLimit);
                                return false;
                        }

                        return true;
                });
        }

        public void consumeQuota(TenantContext tenantContext, int tokens) {
                quotaRepository.incrementUsage(tenantContext.getTenantId(), tokens);
                quotaRepository.recordRequest(tenantContext.getTenantId());
        }
}