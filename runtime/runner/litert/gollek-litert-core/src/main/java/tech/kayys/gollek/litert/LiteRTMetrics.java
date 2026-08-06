package tech.kayys.gollek.litert;

import java.util.Objects;

/**
 * LiteRT performance metrics.
 */
public class LiteRTMetrics {

    /**
     * Total number of inferences.
     */
    private long totalInferences = 0;

    /**
     * Number of failed inferences.
     */
    private long failedInferences = 0;

    /**
     * Average latency in milliseconds.
     */
    private double avgLatencyMs = 0.0;

    /**
     * P50 latency in milliseconds.
     */
    private double p50LatencyMs = 0.0;

    /**
     * P95 latency in milliseconds.
     */
    private double p95LatencyMs = 0.0;

    /**
     * P99 latency in milliseconds.
     */
    private double p99LatencyMs = 0.0;

    /**
     * Peak memory usage in bytes.
     */
    private long peakMemoryBytes = 0;

    /**
     * Current memory usage in bytes.
     */
    private long currentMemoryBytes = 0;

    /**
     * Active delegate being used.
     */
    private String activeDelegate;

    public LiteRTMetrics() {
    }

    public LiteRTMetrics(long totalInferences, long failedInferences, double avgLatencyMs, double p50LatencyMs,
            double p95LatencyMs, double p99LatencyMs, long peakMemoryBytes, long currentMemoryBytes,
            String activeDelegate) {
        this.totalInferences = totalInferences;
        this.failedInferences = failedInferences;
        this.avgLatencyMs = avgLatencyMs;
        this.p50LatencyMs = p50LatencyMs;
        this.p95LatencyMs = p95LatencyMs;
        this.p99LatencyMs = p99LatencyMs;
        this.peakMemoryBytes = peakMemoryBytes;
        this.currentMemoryBytes = currentMemoryBytes;
        this.activeDelegate = activeDelegate;
    }

    public long getTotalInferences() {
        return totalInferences;
    }

    public void setTotalInferences(long totalInferences) {
        this.totalInferences = totalInferences;
    }

    public long getFailedInferences() {
        return failedInferences;
    }

    public void setFailedInferences(long failedInferences) {
        this.failedInferences = failedInferences;
    }

    public double getAvgLatencyMs() {
        return avgLatencyMs;
    }

    public void setAvgLatencyMs(double avgLatencyMs) {
        this.avgLatencyMs = avgLatencyMs;
    }

    public double getP50LatencyMs() {
        return p50LatencyMs;
    }

    public void setP50LatencyMs(double p50LatencyMs) {
        this.p50LatencyMs = p50LatencyMs;
    }

    public double getP95LatencyMs() {
        return p95LatencyMs;
    }

    public void setP95LatencyMs(double p95LatencyMs) {
        this.p95LatencyMs = p95LatencyMs;
    }

    public double getP99LatencyMs() {
        return p99LatencyMs;
    }

    public void setP99LatencyMs(double p99LatencyMs) {
        this.p99LatencyMs = p99LatencyMs;
    }

    public long getPeakMemoryBytes() {
        return peakMemoryBytes;
    }

    public void setPeakMemoryBytes(long peakMemoryBytes) {
        this.peakMemoryBytes = peakMemoryBytes;
    }

    public long getCurrentMemoryBytes() {
        return currentMemoryBytes;
    }

    public void setCurrentMemoryBytes(long currentMemoryBytes) {
        this.currentMemoryBytes = currentMemoryBytes;
    }

    public String getActiveDelegate() {
        return activeDelegate;
    }

    public void setActiveDelegate(String activeDelegate) {
        this.activeDelegate = activeDelegate;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        LiteRTMetrics that = (LiteRTMetrics) o;
        return totalInferences == that.totalInferences && failedInferences == that.failedInferences
                && Double.compare(avgLatencyMs, that.avgLatencyMs) == 0
                && Double.compare(p50LatencyMs, that.p50LatencyMs) == 0
                && Double.compare(p95LatencyMs, that.p95LatencyMs) == 0
                && Double.compare(p99LatencyMs, that.p99LatencyMs) == 0
                && peakMemoryBytes == that.peakMemoryBytes && currentMemoryBytes == that.currentMemoryBytes
                && Objects.equals(activeDelegate, that.activeDelegate);
    }

    @Override
    public int hashCode() {
        return Objects.hash(totalInferences, failedInferences, avgLatencyMs, p50LatencyMs, p95LatencyMs, p99LatencyMs,
                peakMemoryBytes, currentMemoryBytes, activeDelegate);
    }

    @Override
    public String toString() {
        return "LiteRTMetrics{" +
                "totalInferences=" + totalInferences +
                ", failedInferences=" + failedInferences +
                ", avgLatencyMs=" + avgLatencyMs +
                ", p50LatencyMs=" + p50LatencyMs +
                ", p95LatencyMs=" + p95LatencyMs +
                ", p99LatencyMs=" + p99LatencyMs +
                ", peakMemoryBytes=" + peakMemoryBytes +
                ", currentMemoryBytes=" + currentMemoryBytes +
                ", activeDelegate='" + activeDelegate + '\'' +
                '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private long totalInferences = 0;
        private long failedInferences = 0;
        private double avgLatencyMs = 0.0;
        private double p50LatencyMs = 0.0;
        private double p95LatencyMs = 0.0;
        private double p99LatencyMs = 0.0;
        private long peakMemoryBytes = 0;
        private long currentMemoryBytes = 0;
        private String activeDelegate;

        public Builder totalInferences(long totalInferences) {
            this.totalInferences = totalInferences;
            return this;
        }

        public Builder failedInferences(long failedInferences) {
            this.failedInferences = failedInferences;
            return this;
        }

        public Builder avgLatencyMs(double avgLatencyMs) {
            this.avgLatencyMs = avgLatencyMs;
            return this;
        }

        public Builder p50LatencyMs(double p50LatencyMs) {
            this.p50LatencyMs = p50LatencyMs;
            return this;
        }

        public Builder p95LatencyMs(double p95LatencyMs) {
            this.p95LatencyMs = p95LatencyMs;
            return this;
        }

        public Builder p99LatencyMs(double p99LatencyMs) {
            this.p99LatencyMs = p99LatencyMs;
            return this;
        }

        public Builder peakMemoryBytes(long peakMemoryBytes) {
            this.peakMemoryBytes = peakMemoryBytes;
            return this;
        }

        public Builder currentMemoryBytes(long currentMemoryBytes) {
            this.currentMemoryBytes = currentMemoryBytes;
            return this;
        }

        public Builder activeDelegate(String activeDelegate) {
            this.activeDelegate = activeDelegate;
            return this;
        }

        public LiteRTMetrics build() {
            return new LiteRTMetrics(totalInferences, failedInferences, avgLatencyMs, p50LatencyMs, p95LatencyMs,
                    p99LatencyMs, peakMemoryBytes, currentMemoryBytes, activeDelegate);
        }
    }
}
