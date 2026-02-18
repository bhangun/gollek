package tech.kayys.gollek.provider.core.ratelimit;

/**
 * Rate limiter contract for controlling request throughput.
 * 
 * Implementations must be thread-safe.
 */
public interface RateLimiter {

    /**
     * Try to acquire a single permit.
     * 
     * @return true if permit acquired, false if rate limit exceeded
     */
    boolean tryAcquire();

    /**
     * Try to acquire multiple permits.
     * 
     * @param permits number of permits to acquire
     * @return true if all permits acquired, false otherwise
     */
    boolean tryAcquire(int permits);

    /**
     * Get number of available permits.
     * 
     * @return available permits (may be approximate)
     */
    int availablePermits();

    /**
     * Reset the rate limiter state.
     * Useful for testing or administrative operations.
     */
    void reset();
}