package tech.kayys.wayang.tool.dto;

import java.util.Set;

/**
 * Retry configuration
 */
public class RetryConfig {
    private int maxAttempts = 3;
    private long initialDelayMs = 1000;
    private double backoffMultiplier = 2.0;
    private long maxDelayMs = 30000;
    private Set<Integer> retryableStatusCodes = Set.of(408, 429, 500, 502, 503, 504);

    public int getMaxAttempts() {
        return maxAttempts;
    }

    public void setMaxAttempts(int maxAttempts) {
        this.maxAttempts = maxAttempts;
    }

    public long getInitialDelayMs() {
        return initialDelayMs;
    }

    public void setInitialDelayMs(long initialDelayMs) {
        this.initialDelayMs = initialDelayMs;
    }

    public double getBackoffMultiplier() {
        return backoffMultiplier;
    }

    public void setBackoffMultiplier(double backoffMultiplier) {
        this.backoffMultiplier = backoffMultiplier;
    }

    public long getMaxDelayMs() {
        return maxDelayMs;
    }

    public void setMaxDelayMs(long maxDelayMs) {
        this.maxDelayMs = maxDelayMs;
    }

    public Set<Integer> getRetryableStatusCodes() {
        return retryableStatusCodes;
    }

    public void setRetryableStatusCodes(Set<Integer> retryableStatusCodes) {
        this.retryableStatusCodes = retryableStatusCodes;
    }

}