package tech.kayys.golek.api.provider;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Thread-safe metrics collector for providers.
 */
public final class ProviderMetrics {

    private final LongAdder totalRequests = new LongAdder();
    private final LongAdder successfulRequests = new LongAdder();
    private final LongAdder failedRequests = new LongAdder();
    private final LongAdder totalTokens = new LongAdder();
    private final LongAdder totalDurationMs = new LongAdder();
    private final AtomicLong lastRequestTimestamp = new AtomicLong(0);

    public void recordRequest() {
        totalRequests.increment();
    }

    public void recordSuccess() {
        successfulRequests.increment();
    }

    public void recordFailure() {
        failedRequests.increment();
    }

    public void recordTokens(int tokens) {
        totalTokens.add(tokens);
    }

    public void recordDuration(long durationMs) {
        totalDurationMs.add(durationMs);
        lastRequestTimestamp.set(System.currentTimeMillis());
    }

    public long getTotalRequests() {
        return totalRequests.sum();
    }

    public long getSuccessfulRequests() {
        return successfulRequests.sum();
    }

    public long getFailedRequests() {
        return failedRequests.sum();
    }

    public long getTotalTokens() {
        return totalTokens.sum();
    }

    public long getTotalDurationMs() {
        return totalDurationMs.sum();
    }

    public long getLastRequestTimestamp() {
        return lastRequestTimestamp.get();
    }

    public double getSuccessRate() {
        long total = getTotalRequests();
        if (total == 0)
            return 0.0;
        return (double) getSuccessfulRequests() / total;
    }

    public double getAverageDurationMs() {
        long total = getTotalRequests();
        if (total == 0)
            return 0.0;
        return (double) getTotalDurationMs() / total;
    }

    public Duration getTimeSinceLastRequest() {
        long last = getLastRequestTimestamp();
        if (last == 0)
            return Duration.ZERO;
        return Duration.ofMillis(System.currentTimeMillis() - last);
    }

    public void reset() {
        totalRequests.reset();
        successfulRequests.reset();
        failedRequests.reset();
        totalTokens.reset();
        totalDurationMs.reset();
    }

    @Override
    public String toString() {
        return "ProviderMetrics{" +
                "totalRequests=" + getTotalRequests() +
                ", successRate=" + String.format("%.2f%%", getSuccessRate() * 100) +
                ", avgDuration=" + getAverageDurationMs() + "ms" +
                '}';
    }
}
