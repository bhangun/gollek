/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.plugin.kernel;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Metrics collector for kernel plugin operations with support for
 * operation timing, error tracking, and performance monitoring.
 *
 * @since 2.0.0
 */
public final class KernelMetrics {

    private final Map<String, LongAdder> counters = new ConcurrentHashMap<>();
    private final Map<String, OperationStats> operationStats = new ConcurrentHashMap<>();
    private final Map<String, Long> errorCounts = new ConcurrentHashMap<>();
    private final Map<String, Long> lastEventTime = new ConcurrentHashMap<>();
    private final Map<String, Object> metadata = new ConcurrentHashMap<>();
    private final AtomicLong startTime;

    public KernelMetrics() {
        this.startTime = new AtomicLong(System.currentTimeMillis());
    }

    /**
     * Record counter increment.
     *
     * @param name counter name
     * @param amount amount to increment
     */
    public void incrementCounter(String name, long amount) {
        counters.computeIfAbsent(name, k -> new LongAdder()).add(amount);
    }

    /**
     * Record counter increment.
     *
     * @param name counter name
     */
    public void incrementCounter(String name) {
        incrementCounter(name, 1);
    }

    /**
     * Record operation execution.
     *
     * @param operation operation name
     * @param durationMs execution duration in milliseconds
     * @param success whether operation succeeded
     */
    public void recordOperation(String operation, long durationMs, boolean success) {
        operationStats.computeIfAbsent(operation, k -> new OperationStats())
                .record(durationMs, success);
    }

    /**
     * Record error.
     *
     * @param errorType error type
     * @param exception exception
     */
    public void recordError(String errorType, Exception exception) {
        errorCounts.compute(errorType, (k, v) -> v == null ? 1 : v + 1);
        incrementCounter("total_errors");
    }

    /**
     * Record validation error.
     *
     * @param platform platform
     * @param errors error messages
     */
    public void recordValidationError(String platform, List<String> errors) {
        incrementCounter("validation_errors");
        metadata.put("validation_error_" + platform, errors);
    }

    /**
     * Record event.
     *
     * @param eventName event name
     */
    public void recordEvent(String eventName) {
        lastEventTime.put(eventName, System.currentTimeMillis());
        incrementCounter("event_" + eventName);
    }

    /**
     * Record kernel registered.
     *
     * @param platform platform
     */
    public void recordKernelRegistered(String platform) {
        incrementCounter("kernels_registered");
        metadata.put("kernel_registered_" + platform, System.currentTimeMillis());
    }

    /**
     * Record kernel activated.
     *
     * @param platform platform
     */
    public void recordKernelActivated(String platform) {
        incrementCounter("kernels_activated");
        metadata.put("kernel_activated_" + platform, System.currentTimeMillis());
    }

    /**
     * Get counter value.
     *
     * @param name counter name
     * @return counter value
     */
    public long getCounter(String name) {
        LongAdder adder = counters.get(name);
        return adder != null ? adder.sum() : 0;
    }

    /**
     * Get error count.
     *
     * @param errorType error type
     * @return error count
     */
    public long getErrorCount(String errorType) {
        return errorCounts.getOrDefault(errorType, 0L);
    }

    /**
     * Get operation statistics.
     *
     * @param operation operation name
     * @return operation stats
     */
    public OperationStats getOperationStats(String operation) {
        return operationStats.getOrDefault(operation, OperationStats.EMPTY);
    }

    /**
     * Get all counters.
     *
     * @return immutable counter map
     */
    public Map<String, Long> getAllCounters() {
        Map<String, Long> result = new HashMap<>();
        counters.forEach((k, v) -> result.put(k, v.sum()));
        return result;
    }

    /**
     * Get all error counts.
     *
     * @return immutable error map
     */
    public Map<String, Long> getAllErrors() {
        return Map.copyOf(errorCounts);
    }

    /**
     * Get uptime in milliseconds.
     *
     * @return uptime
     */
    public long getUptime() {
        return System.currentTimeMillis() - startTime.get();
    }

    /**
     * Get start time.
     *
     * @return start time
     */
    public long getStartTime() {
        return startTime.get();
    }

    /**
     * Get last event time.
     *
     * @param eventName event name
     * @return last event time or null
     */
    public Long getLastEventTime(String eventName) {
        return lastEventTime.get(eventName);
    }

    /**
     * Get metadata.
     *
     * @return immutable metadata map
     */
    public Map<String, Object> getMetadata() {
        return Map.copyOf(metadata);
    }

    /**
     * Convert metrics to map.
     *
     * @return metrics map
     */
    public Map<String, Object> toMap() {
        Map<String, Object> result = new HashMap<>();
        result.put("uptime_ms", getUptime());
        result.put("start_time", startTime.get());
        result.put("counters", getAllCounters());
        result.put("errors", getAllErrors());

        Map<String, Object> opsStats = new HashMap<>();
        operationStats.forEach((op, stats) ->
            opsStats.put(op, stats.toMap()));
        result.put("operations", opsStats);

        result.put("metadata", metadata);

        return result;
    }

    /**
     * Reset all metrics.
     */
    public void reset() {
        counters.clear();
        operationStats.clear();
        errorCounts.clear();
        lastEventTime.clear();
        metadata.clear();
        startTime.set(System.currentTimeMillis());
    }

    /**
     * Operation statistics.
     */
    public static final class OperationStats {
        private static final OperationStats EMPTY = new OperationStats();

        private final LongAdder count = new LongAdder();
        private final LongAdder successCount = new LongAdder();
        private final LongAdder failureCount = new LongAdder();
        private final LongAdder totalDuration = new LongAdder();
        private volatile long minDuration = Long.MAX_VALUE;
        private volatile long maxDuration = 0;

        private OperationStats() {
            // Private constructor for EMPTY
        }

        /**
         * Record operation execution.
         *
         * @param durationMs duration in milliseconds
         * @param success whether operation succeeded
         */
        public void record(long durationMs, boolean success) {
            count.increment();
            totalDuration.add(durationMs);

            if (success) {
                successCount.increment();
            } else {
                failureCount.increment();
            }

            // Update min/max
            long currentMin = minDuration;
            while (durationMs < currentMin) {
                if (compareAndSetMin(currentMin, durationMs)) {
                    break;
                }
                currentMin = minDuration;
            }

            long currentMax = maxDuration;
            while (durationMs > currentMax) {
                if (compareAndSetMax(currentMax, durationMs)) {
                    break;
                }
                currentMax = maxDuration;
            }
        }

        private synchronized boolean compareAndSetMin(long expect, long update) {
            if (minDuration == expect) {
                minDuration = update;
                return true;
            }
            return false;
        }

        private synchronized boolean compareAndSetMax(long expect, long update) {
            if (maxDuration == expect) {
                maxDuration = update;
                return true;
            }
            return false;
        }

        /**
         * Get total count.
         *
         * @return count
         */
        public long getCount() {
            return count.sum();
        }

        /**
         * Get success count.
         *
         * @return success count
         */
        public long getSuccessCount() {
            return successCount.sum();
        }

        /**
         * Get failure count.
         *
         * @return failure count
         */
        public long getFailureCount() {
            return failureCount.sum();
        }

        /**
         * Get success rate.
         *
         * @return success rate (0.0-1.0)
         */
        public double getSuccessRate() {
            long total = count.sum();
            return total > 0 ? (double) successCount.sum() / total : 0.0;
        }

        /**
         * Get average duration.
         *
         * @return average duration in ms
         */
        public double getAverageDuration() {
            long total = count.sum();
            return total > 0 ? (double) totalDuration.sum() / total : 0.0;
        }

        /**
         * Get minimum duration.
         *
         * @return minimum duration in ms
         */
        public long getMinDuration() {
            return minDuration == Long.MAX_VALUE ? 0 : minDuration;
        }

        /**
         * Get maximum duration.
         *
         * @return maximum duration in ms
         */
        public long getMaxDuration() {
            return maxDuration;
        }

        /**
         * Convert to map.
         *
         * @return stats map
         */
        public Map<String, Object> toMap() {
            Map<String, Object> map = new HashMap<>();
            map.put("count", getCount());
            map.put("success_count", getSuccessCount());
            map.put("failure_count", getFailureCount());
            map.put("success_rate", getSuccessRate());
            map.put("avg_duration_ms", getAverageDuration());
            map.put("min_duration_ms", getMinDuration());
            map.put("max_duration_ms", getMaxDuration());
            return map;
        }
    }
}
