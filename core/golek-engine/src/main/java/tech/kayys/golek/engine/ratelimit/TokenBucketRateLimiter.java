package tech.kayys.golek.engine.ratelimit;

import org.jboss.logging.Logger;

import tech.kayys.golek.provider.core.ratelimit.RateLimiter;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Token bucket rate limiter implementation.
 * 
 * Allows bursts up to capacity while maintaining steady refill rate.
 * More memory-efficient than sliding window for high-throughput scenarios.
 * 
 * Performance characteristics:
 * - tryAcquire: O(1)
 * - availablePermits: O(1)
 * - Memory: O(1)
 */
public class TokenBucketRateLimiter implements RateLimiter {

    private static final Logger LOG = Logger.getLogger(TokenBucketRateLimiter.class);

    private final int capacity;
    private final double refillRate; // tokens per nanosecond
    private final AtomicLong tokens; // stored as nanos * refillRate
    private final AtomicLong lastRefillTime;
    private final Lock lock;

    /**
     * Create token bucket rate limiter.
     * 
     * @param capacity     maximum tokens (burst size)
     * @param refillPeriod time to fully refill bucket
     */
    public TokenBucketRateLimiter(int capacity, Duration refillPeriod) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive");
        }
        if (refillPeriod == null || refillPeriod.isNegative() || refillPeriod.isZero()) {
            throw new IllegalArgumentException("refillPeriod must be positive");
        }

        this.capacity = capacity;
        this.refillRate = (double) capacity / refillPeriod.toNanos();
        this.tokens = new AtomicLong((long) (capacity * 1e9)); // Store as nanos
        this.lastRefillTime = new AtomicLong(System.nanoTime());
        this.lock = new ReentrantLock();

        LOG.infof("Created TokenBucketRateLimiter: capacity=%d, refillPeriod=%s, rate=%.2e tokens/ns",
                capacity, refillPeriod, refillRate);
    }

    @Override
    public boolean tryAcquire() {
        return tryAcquire(1);
    }

    @Override
    public boolean tryAcquire(int permits) {
        if (permits <= 0) {
            throw new IllegalArgumentException("permits must be positive");
        }
        if (permits > capacity) {
            return false; // Cannot acquire more than capacity
        }

        lock.lock();
        try {
            refillTokens();

            long required = (long) (permits * 1e9);
            long available = tokens.get();

            if (available >= required) {
                tokens.addAndGet(-required);
                LOG.trace("Acquired %d permits, remaining: %.2f");
                return true;
            }

            LOG.debugf("Insufficient tokens: required=%.2f, available=%.2f",
                    required / 1e9, available / 1e9);
            return false;

        } finally {
            lock.unlock();
        }
    }

    @Override
    public int availablePermits() {
        lock.lock();
        try {
            refillTokens();
            return (int) Math.min(capacity, tokens.get() / 1e9);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void reset() {
        lock.lock();
        try {
            tokens.set((long) (capacity * 1e9));
            lastRefillTime.set(System.nanoTime());
            LOG.infof("Token bucket reset: capacity=%d", capacity);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Refill tokens based on elapsed time.
     * Must be called while holding lock.
     */
    private void refillTokens() {
        long now = System.nanoTime();
        long last = lastRefillTime.get();
        long elapsed = now - last;

        if (elapsed > 0) {
            long toAdd = (long) (elapsed * refillRate * 1e9);
            if (toAdd > 0) {
                long newTokens = Math.min(
                        (long) (capacity * 1e9),
                        tokens.get() + toAdd);
                tokens.set(newTokens);
                lastRefillTime.set(now);

                LOG.tracef("Refilled tokens: added=%.2f, total=%.2f",
                        toAdd / 1e9, newTokens / 1e9);
            }
        }
    }

    /**
     * Get time until specified permits are available
     */
    public Duration getTimeUntilAvailable(int permits) {
        lock.lock();
        try {
            refillTokens();

            long required = (long) (permits * 1e9);
            long available = tokens.get();

            if (available >= required) {
                return Duration.ZERO;
            }

            long deficit = required - available;
            long nanosNeeded = (long) (deficit / refillRate);

            return Duration.ofNanos(nanosNeeded);

        } finally {
            lock.unlock();
        }
    }

    @Override
    public String toString() {
        return String.format(
                "TokenBucketRateLimiter{capacity=%d, available=%.2f, rate=%.2e/ns}",
                capacity, tokens.get() / 1e9, refillRate);
    }
}