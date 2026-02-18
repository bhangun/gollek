package tech.kayys.gollek.engine.ratelimit;

import org.jboss.logging.Logger;

import tech.kayys.gollek.provider.core.ratelimit.RateLimiter;

import java.time.Duration;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe sliding window rate limiter implementation.
 * 
 * Uses a deque to track timestamps of requests within the sliding window.
 * Automatically evicts expired timestamps to maintain accurate counts.
 * 
 * Performance characteristics:
 * - tryAcquire: O(n) worst case where n = expired timestamps
 * - availablePermits: O(n) worst case
 * - Memory: O(maxRequests) in worst case
 * 
 * Thread-safety: All operations are atomic and thread-safe.
 */
public class SlidingWindowRateLimiter implements RateLimiter {

    private static final Logger LOG = Logger.getLogger(SlidingWindowRateLimiter.class);

    private final int maxRequests;
    private final Duration window;
    private final ConcurrentLinkedDeque<Long> timestamps;
    private final AtomicInteger count;
    private final ReadWriteLock lock;

    // Metrics
    private final AtomicInteger rejectedCount;
    private final AtomicInteger acceptedCount;

    public SlidingWindowRateLimiter(int maxRequests, Duration window) {
        if (maxRequests <= 0) {
            throw new IllegalArgumentException("maxRequests must be positive");
        }
        if (window == null || window.isNegative() || window.isZero()) {
            throw new IllegalArgumentException("window must be positive duration");
        }

        this.maxRequests = maxRequests;
        this.window = window;
        this.timestamps = new ConcurrentLinkedDeque<>();
        this.count = new AtomicInteger(0);
        this.lock = new ReentrantReadWriteLock();
        this.rejectedCount = new AtomicInteger(0);
        this.acceptedCount = new AtomicInteger(0);

        LOG.infof("Created SlidingWindowRateLimiter: maxRequests=%d, window=%s",
                maxRequests, window);
    }

    @Override
    public boolean tryAcquire() {
        return tryAcquire(1);
    }

    @Override
    public boolean tryAcquire(int permits) {
        if (permits != 1) {
            throw new UnsupportedOperationException(
                    "Sliding window only supports single permits, got: " + permits);
        }

        lock.writeLock().lock();
        try {
            long now = System.nanoTime();
            long windowStart = now - window.toNanos();

            // Remove old timestamps outside the window
            cleanupOldTimestamps(windowStart);

            // Check if we can accept new request
            if (count.get() < maxRequests) {
                timestamps.addLast(now);
                count.incrementAndGet();
                acceptedCount.incrementAndGet();

                LOG.tracef("Rate limit acquired: current=%d, max=%d",
                        count.get(), maxRequests);
                return true;
            }

            rejectedCount.incrementAndGet();
            LOG.debugf("Rate limit exceeded: current=%d, max=%d",
                    count.get(), maxRequests);
            return false;

        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public int availablePermits() {
        lock.readLock().lock();
        try {
            long now = System.nanoTime();
            long windowStart = now - window.toNanos();

            // Cleanup in read-only check
            cleanupOldTimestamps(windowStart);

            return Math.max(0, maxRequests - count.get());
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public void reset() {
        lock.writeLock().lock();
        try {
            timestamps.clear();
            count.set(0);
            rejectedCount.set(0);
            acceptedCount.set(0);

            LOG.infof("Rate limiter reset: maxRequests=%d, window=%s",
                    maxRequests, window);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Remove timestamps that fall outside the current window.
     * Should be called while holding write lock.
     * 
     * @param windowStart Nanosecond timestamp of window start
     */
    private void cleanupOldTimestamps(long windowStart) {
        int removed = 0;

        // Remove from head while timestamps are outside window
        while (!timestamps.isEmpty()) {
            Long oldest = timestamps.peekFirst();
            if (oldest != null && oldest < windowStart) {
                timestamps.pollFirst();
                count.decrementAndGet();
                removed++;
            } else {
                break;
            }
        }

        if (removed > 0) {
            LOG.tracef("Cleaned up %d expired timestamps", removed);
        }
    }

    /**
     * Get metrics for monitoring
     */
    public RateLimiterMetrics getMetrics() {
        lock.readLock().lock();
        try {
            return new RateLimiterMetrics(
                    maxRequests,
                    count.get(),
                    acceptedCount.get(),
                    rejectedCount.get(),
                    availablePermits());
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Get time until next permit is available
     */
    public Duration getTimeUntilNextPermit() {
        lock.readLock().lock();
        try {
            if (count.get() < maxRequests) {
                return Duration.ZERO;
            }

            Long oldest = timestamps.peekFirst();
            if (oldest == null) {
                return Duration.ZERO;
            }

            long now = System.nanoTime();
            long oldestExpiry = oldest + window.toNanos();
            long nanosUntilAvailable = oldestExpiry - now;

            return nanosUntilAvailable > 0
                    ? Duration.ofNanos(nanosUntilAvailable)
                    : Duration.ZERO;

        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public String toString() {
        return String.format(
                "SlidingWindowRateLimiter{max=%d, window=%s, current=%d, available=%d}",
                maxRequests, window, count.get(), availablePermits());
    }

    /**
     * Metrics snapshot for monitoring
     */
    public record RateLimiterMetrics(
            int maxRequests,
            int currentRequests,
            int totalAccepted,
            int totalRejected,
            int availablePermits) {
        public double rejectionRate() {
            int total = totalAccepted + totalRejected;
            return total > 0 ? (double) totalRejected / total : 0.0;
        }

        public double utilizationRate() {
            return maxRequests > 0 ? (double) currentRequests / maxRequests : 0.0;
        }
    }
}