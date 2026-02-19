package tech.kayys.gollek.spi.inference;

import java.time.Duration;

/**
 * Configuration for the inference batching scheduler.
 */
public record BatchConfig(
        BatchStrategy strategy,
        int maxBatchSize,
        Duration maxWaitTime,
        int maxConcurrentBatches,
        int prefillBatchSize,
        int smallPromptThreshold,
        boolean enableDisaggregation) {

    public static BatchConfig defaultDynamic() {
        return new BatchConfig(
                BatchStrategy.DYNAMIC,
                8,
                Duration.ofMillis(50),
                4,
                4,
                128,
                false);
    }

    public static BatchConfig defaultContinuous() {
        return new BatchConfig(
                BatchStrategy.CONTINUOUS,
                32,
                Duration.ofMillis(10),
                8,
                8,
                128,
                true);
    }

    public void validate() {
        if (maxBatchSize <= 0) {
            throw new IllegalArgumentException("maxBatchSize must be > 0");
        }
        if (maxWaitTime == null || maxWaitTime.isNegative()) {
            throw new IllegalArgumentException("maxWaitTime must be >= 0");
        }
        if (maxConcurrentBatches <= 0) {
            throw new IllegalArgumentException("maxConcurrentBatches must be > 0");
        }
    }
}
