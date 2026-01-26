package tech.kayys.golek.memory;

import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Default circuit breaker implementation with configurable thresholds
 */
public class DefaultCircuitBreaker implements CircuitBreaker {

    private static final Logger LOG = Logger.getLogger(DefaultCircuitBreaker.class);

    private final String name;
    private final CircuitBreakerConfig config;
    private final AtomicReference<State> state = new AtomicReference<>(State.CLOSED);
    private final AtomicLong successCount = new AtomicLong(0);
    private final AtomicLong failureCount = new AtomicLong(0);
    private final AtomicLong rejectedCount = new AtomicLong(0);
    private final AtomicInteger halfOpenAttempts = new AtomicInteger(0);
    private volatile Instant openedAt;
    private volatile Instant lastFailureAt;

    public DefaultCircuitBreaker(String name, CircuitBreakerConfig config) {
        this.name = name;
        this.config = config;
    }

    @Override
    public <T> T call(Callable<T> callable) throws Exception {
        State currentState = state.get();

        // Check if circuit is open
        if (currentState == State.OPEN) {
            if (shouldAttemptReset()) {
                LOG.debugf("[%s] Transitioning to HALF_OPEN", name);
                state.compareAndSet(State.OPEN, State.HALF_OPEN);
                halfOpenAttempts.set(0);
            } else {
                rejectedCount.incrementAndGet();
                throw new CircuitBreakerOpenException(
                        "Circuit breaker is OPEN for: " + name);
            }
        }

        // Execute call
        try {
            T result = callable.call();
            onSuccess();
            return result;

        } catch (Exception e) {
            onFailure(e);
            throw e;
        }
    }

    @Override
    public <T> Uni<T> callAsync(Callable<Uni<T>> callable) {
        return Uni.createFrom().deferred(() -> {
            try {
                State currentState = state.get();

                if (currentState == State.OPEN) {
                    if (shouldAttemptReset()) {
                        state.compareAndSet(State.OPEN, State.HALF_OPEN);
                        halfOpenAttempts.set(0);
                    } else {
                        rejectedCount.incrementAndGet();
                        return Uni.createFrom().failure(
                                new CircuitBreakerOpenException(
                                        "Circuit breaker is OPEN for: " + name));
                    }
                }

                return callable.call()
                        .onItem().invoke(item -> onSuccess())
                        .onFailure().invoke(this::onFailure);

            } catch (Exception e) {
                return Uni.createFrom().failure(e);
            }
        });
    }

    @Override
    public State getState() {
        State currentState = state.get();

        // Auto-transition from OPEN to HALF_OPEN if timeout elapsed
        if (currentState == State.OPEN && shouldAttemptReset()) {
            state.compareAndSet(State.OPEN, State.HALF_OPEN);
            return State.HALF_OPEN;
        }

        return currentState;
    }

    @Override
    public void tripOpen() {
        State previous = state.getAndSet(State.OPEN);
        openedAt = Instant.now();

        if (previous != State.OPEN) {
            LOG.warnf("[%s] Circuit breaker tripped OPEN (failures: %d, rate: %.2f%%)",
                    name, failureCount.get(), calculateFailureRate() * 100);
        }
    }

    @Override
    public void reset() {
        state.set(State.CLOSED);
        successCount.set(0);
        failureCount.set(0);
        rejectedCount.set(0);
        halfOpenAttempts.set(0);
        openedAt = null;
        lastFailureAt = null;

        LOG.infof("[%s] Circuit breaker reset to CLOSED", name);
    }

    @Override
    public CircuitBreakerMetrics getMetrics() {
        return new MetricsSnapshot(
                successCount.get(),
                failureCount.get(),
                rejectedCount.get(),
                state.get());
    }

    private void onSuccess() {
        successCount.incrementAndGet();
        lastFailureAt = null;

        State currentState = state.get();

        if (currentState == State.HALF_OPEN) {
            int attempts = halfOpenAttempts.incrementAndGet();

            if (attempts >= config.halfOpenSuccessThreshold) {
                state.compareAndSet(State.HALF_OPEN, State.CLOSED);
                halfOpenAttempts.set(0);

                LOG.infof("[%s] Circuit breaker closed after successful test calls", name);
            }
        }
    }

    private void onFailure(Throwable error) {
        failureCount.incrementAndGet();
        lastFailureAt = Instant.now();

        State currentState = state.get();

        if (currentState == State.HALF_OPEN) {
            // Immediately reopen on failure during half-open
            tripOpen();
            return;
        }

        if (currentState == State.CLOSED) {
            // Check if should open
            if (shouldOpen()) {
                tripOpen();
            }
        }
    }

    private boolean shouldOpen() {
        long total = successCount.get() + failureCount.get();

        // Need minimum calls before opening
        if (total < config.minimumCalls) {
            return false;
        }

        // Check failure threshold
        if (failureCount.get() >= config.failureThreshold) {
            return true;
        }

        // Check failure rate
        double failureRate = calculateFailureRate();
        return failureRate >= config.failureRateThreshold;
    }

    private boolean shouldAttemptReset() {
        if (openedAt == null) {
            return false;
        }

        Duration elapsed = Duration.between(openedAt, Instant.now());
        return elapsed.compareTo(config.openDuration) >= 0;
    }

    private double calculateFailureRate() {
        long total = successCount.get() + failureCount.get();
        if (total == 0) {
            return 0.0;
        }
        return (double) failureCount.get() / total;
    }

    private static class MetricsSnapshot implements CircuitBreakerMetrics {
        private final long successCount;
        private final long failureCount;
        private final long rejectedCount;
        private final State state;

        MetricsSnapshot(long successCount, long failureCount, long rejectedCount, State state) {
            this.successCount = successCount;
            this.failureCount = failureCount;
            this.rejectedCount = rejectedCount;
            this.state = state;
        }

        @Override
        public long successCount() {
            return successCount;
        }

        @Override
        public long failureCount() {
            return failureCount;
        }

        @Override
        public long totalCalls() {
            return successCount + failureCount;
        }

        @Override
        public double failureRate() {
            long total = totalCalls();
            return total > 0 ? (double) failureCount / total : 0.0;
        }

        @Override
        public State currentState() {
            return state;
        }

        @Override
        public long rejectedCalls() {
            return rejectedCount;
        }

        @Override
        public String toString() {
            return String.format(
                    "CircuitBreakerMetrics{state=%s, success=%d, failure=%d, rejected=%d, rate=%.2f%%}",
                    state, successCount, failureCount, rejectedCount, failureRate() * 100);
        }
    }

    /**
     * Circuit breaker configuration
     */
    public static class CircuitBreakerConfig {
        private final int failureThreshold;
        private final double failureRateThreshold;
        private final int minimumCalls;
        private final Duration openDuration;
        private final int halfOpenPermits;
        private final int halfOpenSuccessThreshold;

        private CircuitBreakerConfig(Builder builder) {
            this.failureThreshold = builder.failureThreshold;
            this.failureRateThreshold = builder.failureRateThreshold;
            this.minimumCalls = builder.minimumCalls;
            this.openDuration = builder.openDuration;
            this.halfOpenPermits = builder.halfOpenPermits;
            this.halfOpenSuccessThreshold = builder.halfOpenSuccessThreshold;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private int failureThreshold = 5;
            private double failureRateThreshold = 0.5;
            private int minimumCalls = 10;
            private Duration openDuration = Duration.ofSeconds(60);
            private int halfOpenPermits = 3;
            private int halfOpenSuccessThreshold = 2;

            public Builder failureThreshold(int threshold) {
                this.failureThreshold = threshold;
                return this;
            }

            public Builder failureRateThreshold(double threshold) {
                this.failureRateThreshold = threshold;
                return this;
            }

            public Builder minimumCalls(int calls) {
                this.minimumCalls = calls;
                return this;
            }

            public Builder openDuration(Duration duration) {
                this.openDuration = duration;
                return this;
            }

            public Builder halfOpenPermits(int permits) {
                this.halfOpenPermits = permits;
                return this;
            }

            public Builder halfOpenSuccessThreshold(int threshold) {
                this.halfOpenSuccessThreshold = threshold;
                return this;
            }

            public CircuitBreakerConfig build() {
                return new CircuitBreakerConfig(this);
            }
        }
    }
}