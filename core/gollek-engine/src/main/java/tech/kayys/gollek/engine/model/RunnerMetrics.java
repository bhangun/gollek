package tech.kayys.gollek.engine.model;

import java.util.Objects;

/**
 * Runtime metrics for a model runner.
 */
public class RunnerMetrics {

    /**
     * Total requests processed.
     */
    private long totalRequests;

    /**
     * Failed requests count.
     */
    private long failedRequests;

    /**
     * Average latency in milliseconds.
     */
    private long averageLatencyMs;

    /**
     * P95 latency in milliseconds.
     */
    private long p95LatencyMs;

    /**
     * P99 latency in milliseconds.
     */
    private long p99LatencyMs;

    public RunnerMetrics() {}

    public RunnerMetrics(long totalRequests, long failedRequests, long averageLatencyMs, 
                        long p95LatencyMs, long p99LatencyMs) {
        this.totalRequests = totalRequests;
        this.failedRequests = failedRequests;
        this.averageLatencyMs = averageLatencyMs;
        this.p95LatencyMs = p95LatencyMs;
        this.p99LatencyMs = p99LatencyMs;
    }

    // Getters
    public long getTotalRequests() {
        return totalRequests;
    }

    public long getFailedRequests() {
        return failedRequests;
    }

    public long getAverageLatencyMs() {
        return averageLatencyMs;
    }

    public long getP95LatencyMs() {
        return p95LatencyMs;
    }

    public long getP99LatencyMs() {
        return p99LatencyMs;
    }

    // Setters
    public void setTotalRequests(long totalRequests) {
        this.totalRequests = totalRequests;
    }

    public void setFailedRequests(long failedRequests) {
        this.failedRequests = failedRequests;
    }

    public void setAverageLatencyMs(long averageLatencyMs) {
        this.averageLatencyMs = averageLatencyMs;
    }

    public void setP95LatencyMs(long p95LatencyMs) {
        this.p95LatencyMs = p95LatencyMs;
    }

    public void setP99LatencyMs(long p99LatencyMs) {
        this.p99LatencyMs = p99LatencyMs;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RunnerMetrics that = (RunnerMetrics) o;
        return totalRequests == that.totalRequests &&
               failedRequests == that.failedRequests &&
               averageLatencyMs == that.averageLatencyMs &&
               p95LatencyMs == that.p95LatencyMs &&
               p99LatencyMs == that.p99LatencyMs;
    }

    @Override
    public int hashCode() {
        return Objects.hash(totalRequests, failedRequests, averageLatencyMs, p95LatencyMs, p99LatencyMs);
    }
    
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private long totalRequests;
        private long failedRequests;
        private long averageLatencyMs;
        private long p95LatencyMs;
        private long p99LatencyMs;

        public Builder totalRequests(long totalRequests) {
            this.totalRequests = totalRequests;
            return this;
        }

        public Builder failedRequests(long failedRequests) {
            this.failedRequests = failedRequests;
            return this;
        }

        public Builder averageLatencyMs(long averageLatencyMs) {
            this.averageLatencyMs = averageLatencyMs;
            return this;
        }

        public Builder p95LatencyMs(long p95LatencyMs) {
            this.p95LatencyMs = p95LatencyMs;
            return this;
        }

        public Builder p99LatencyMs(long p99LatencyMs) {
            this.p99LatencyMs = p99LatencyMs;
            return this;
        }

        public RunnerMetrics build() {
            return new RunnerMetrics(totalRequests, failedRequests, averageLatencyMs, p95LatencyMs, p99LatencyMs);
        }
    }
}
