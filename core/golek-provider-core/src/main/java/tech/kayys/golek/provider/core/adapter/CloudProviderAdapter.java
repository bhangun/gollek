package tech.kayys.golek.provider.core.adapter;

import io.smallrye.mutiny.Uni;

import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.ProviderCapabilities;
import tech.kayys.golek.provider.core.ratelimit.RateLimiter;
import tech.kayys.golek.provider.core.ratelimit.SlidingWindowRateLimiter;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base adapter for cloud-based LLM providers (OpenAI, Anthropic, etc.)
 * Handles API authentication, rate limiting, and retry logic.
 */
public abstract class CloudProviderAdapter extends AbstractProvider {

    protected String apiKey;
    protected String baseUrl;
    protected Duration requestTimeout;
    protected final AtomicLong totalRequests = new AtomicLong();
    protected final AtomicLong totalErrors = new AtomicLong();

    @jakarta.inject.Inject
    tech.kayys.golek.provider.core.quota.ProviderQuotaService quotaService;

    @Override
    protected Throwable handleFailure(Throwable ex) {
        if (ex instanceof jakarta.ws.rs.WebApplicationException) {
            jakarta.ws.rs.WebApplicationException webEx = (jakarta.ws.rs.WebApplicationException) ex;
            if (webEx.getResponse().getStatus() == 429) {
                // Report exhaustion
                long retryAfter = 60; // Default
                String retryAfterHeader = webEx.getResponse().getHeaderString("Retry-After");
                if (retryAfterHeader != null) {
                    try {
                        retryAfter = Long.parseLong(retryAfterHeader);
                    } catch (NumberFormatException e) {
                        // ignore
                    }
                }
                quotaService.reportExhaustion(id(), retryAfter);
                return new tech.kayys.golek.api.routing.QuotaExhaustedException(id(), "Provider quota exhausted (429)");
            }
        }
        return super.handleFailure(ex);
    }

    @Override
    protected Uni<Void> doInitialize(Map<String, Object> config, TenantContext tenant) {
        return Uni.createFrom().item(() -> {
            this.apiKey = extractApiKey(config, tenant);
            this.baseUrl = getConfigValue("base-url", getDefaultBaseUrl());
            this.requestTimeout = Duration.parse(
                    getConfigValue("request-timeout", "PT30S"));

            if (apiKey == null || apiKey.isBlank()) {
                throw new IllegalStateException(
                        "API key not configured for provider " + id());
            }

            log.infof("Cloud provider %s initialized with base URL: %s",
                    id(), baseUrl);

            return null;
        });
    }

    @Override
    protected Uni<ProviderHealth> doHealthCheck() {
        return performHealthCheckRequest()
                .map(success -> {
                    if (success) {
                        return ProviderHealth.builder()
                                .status(ProviderHealth.Status.HEALTHY)
                                .message("Cloud provider API reachable")
                                .detail("base_url", baseUrl)
                                .detail("total_requests", totalRequests.get())
                                .detail("total_errors", totalErrors.get())
                                .build();
                    } else {
                        return ProviderHealth.degraded("API health check failed");
                    }
                })
                .onFailure().recoverWithItem(ex -> ProviderHealth.unhealthy("API unreachable: " + ex.getMessage()));
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(supportsFunctionCalling())
                .multimodal(supportsMultimodal())
                .embeddings(supportsEmbeddings())
                .maxContextTokens(getMaxContextTokens())
                .maxOutputTokens(getMaxOutputTokens())
                .build();
    }

    @Override
    protected RateLimiter createRateLimiter(String tenantId) {
        // Cloud providers typically have stricter rate limits
        int requestsPerMinute = getConfigValue("rate-limit.requests-per-minute", 60);

        log.debugf("Creating sliding window rate limiter for tenant %s: %d req/min",
                tenantId, requestsPerMinute);

        return new SlidingWindowRateLimiter(requestsPerMinute, Duration.ofMinutes(1));
    }

    /**
     * Extract API key from config or tenant context
     */
    protected String extractApiKey(Map<String, Object> config, TenantContext tenant) {
        // Try config first
        String key = (String) config.get("api-key");

        // Try environment variable
        if (key == null || key.isBlank()) {
            String envVar = getApiKeyEnvironmentVariable();
            key = System.getenv(envVar);
        }

        // Try tenant-specific key
        if (key == null || key.isBlank()) {
            key = tenant.getAttribute("api-key-" + id()).orElse(null);
        }

        return key;
    }

    /**
     * Perform health check request to provider API
     */
    protected abstract Uni<Boolean> performHealthCheckRequest();

    /**
     * Get default base URL for the provider
     */
    protected abstract String getDefaultBaseUrl();

    /**
     * Get environment variable name for API key
     */
    protected abstract String getApiKeyEnvironmentVariable();

    /**
     * Check if provider supports function calling
     */
    protected boolean supportsFunctionCalling() {
        return false;
    }

    /**
     * Check if provider supports multimodal inputs
     */
    protected boolean supportsMultimodal() {
        return false;
    }

    /**
     * Check if provider supports embeddings
     */
    protected boolean supportsEmbeddings() {
        return false;
    }

    /**
     * Get max context tokens (cloud provider specific)
     */
    protected int getMaxContextTokens() {
        return getConfigValue("max-context-tokens", 8192);
    }

    /**
     * Get max output tokens (cloud provider specific)
     */
    protected int getMaxOutputTokens() {
        return getConfigValue("max-output-tokens", 4096);
    }

    /**
     * Track request
     */
    protected void trackRequest() {
        totalRequests.incrementAndGet();
    }

    /**
     * Track error
     */
    protected void trackError() {
        totalErrors.incrementAndGet();
    }

    /**
     * Get API key (for internal use)
     */
    protected String getApiKey() {
        return apiKey;
    }

    /**
     * Get base URL
     */
    protected String getBaseUrl() {
        return baseUrl;
    }

    /**
     * Get request timeout
     */
    protected Duration getRequestTimeout() {
        return requestTimeout;
    }
}