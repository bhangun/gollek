package tech.kayys.golek.engine.reliability;

import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Default circuit breaker implementation with configurable thresholds.
 * 
 * Thread-safe and non-blocking for state queries.
 * Uses atomic operations for counters and CAS for state transitions.
 */
public class DefaultCircuitBreaker implements CircuitBreaker {

    private static final Logger LOG = Logger.getLogger(DefaultCircuitBreaker.class);

    private final String name;
    private final CircuitBreakerConfig config;
    private final AtomicReference<State> state;
    private final AtomicInteger failureCount;
    private final AtomicInteger successCount;
    private final AtomicLong stateChangeTime;
    private final Lock stateLock;
    private final Predicate<Throwable> failurePredicate;

    public DefaultCircuitBreaker(String name, CircuitBreakerConfig config) {
        this.name = name;
        this.config = config;
        this.state = new AtomicReference<>(State.CLOSED);
        this.failureCount = new AtomicInteger(0);
        this.successCount = new AtomicInteger(0);
        this.stateChangeTime = new AtomicLong(System.currentTimeMillis());
        this.stateLock = new ReentrantLock();
        this.failurePredicate = config.failurePredicate();

        LOG.infof("Created circuit breaker '%s' with config: %s", name, config);
    }

    @Override
    public <T> io.smallrye.mutiny.Uni<T> call(io.smallrye.mutiny.Uni<T> uni) {
        if (!permitCall()) {
            return io.smallrye.mutiny.Uni.createFrom().failure(
                    new CircuitBreakerOpenException(name, getEstimatedRecoveryTime()));
        }

        long startTime = System.nanoTime();
        return uni
                .invoke(() -> onSuccess(System.nanoTime() - startTime))
                .onFailure().invoke(th -> onFailure(th, System.nanoTime() - startTime));
    }

    @Override
    public <T> T call(Callable<T> callable) throws Exception {
        if (!permitCall()) {
            throw new CircuitBreakerOpenException(
                    name,
                    getEstimatedRecoveryTime());
        }

        long startTime = System.nanoTime();
        try {
            T result = callable.call();
            onSuccess(System.nanoTime() - startTime);
            return result;

        } catch (Exception e) {
            onFailure(e, System.nanoTime() - startTime);
            throw e;
        }
    }

    @Override
    public <T> T get(Supplier<T> supplier) {
        try {
            return call(supplier::get);
        } catch (Exception e) {
            if (e instanceof RuntimeException) {
                throw (RuntimeException) e;
            }
            throw new RuntimeException("Circuit breaker execution failed", e);
        }
    }

    @Override
    public void run(Runnable runnable) {
        get(() -> {
            runnable.run();
            return null;
        });
    }

    @Override
    public State getState() {
        // Check if OPEN circuit should transition to HALF_OPEN
        if (state.get() == State.OPEN) {
            long timeSinceOpen = System.currentTimeMillis() - stateChangeTime.get();
            if (timeSinceOpen >= config.openDuration().toMillis()) {
                transitionToHalfOpen();
            }
        }
        return state.get();
    }

    @Override
    public CircuitBreakerMetrics getMetrics() {
        State currentState = getState();
        int failures = failureCount.get();
        int successes = successCount.get();
        int total = failures + successes;
        double failureRate = total > 0 ? (double) failures / total : 0.0;

        return new tech.kayys.golek.engine.observability.CircuitBreakerMetrics(
                currentState,
                failures,
                successes,
                total,
                failureRate,
                Duration.ofMillis(System.currentTimeMillis() - stateChangeTime.get()),
                permitCall(),
                getEstimatedRecoveryTime());
    }

    @Override
    public void tripOpen() {
        stateLock.lock();
        try {
            if (state.get() != State.OPEN) {
                LOG.warnf("Circuit breaker '%s' manually tripped OPEN", name);
                transitionTo(State.OPEN);
            }
        } finally {
            stateLock.unlock();
        }
    }

    @Override
    public void reset() {
        stateLock.lock();
        try {
            LOG.infof("Circuit breaker '%s' manually reset", name);
            failureCount.set(0);
            successCount.set(0);
            transitionTo(State.CLOSED);
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Check if call is permitted
     */
    private boolean permitCall() {
        State currentState = getState();

        return switch (currentState) {
            case CLOSED -> true;
            case OPEN -> false;
            case HALF_OPEN -> {
                // In HALF_OPEN, allow limited number of test calls
                int permitted = config.halfOpenPermits();
                int total = successCount.get() + failureCount.get();
                yield total < permitted;
            }
        };
    }

    /**
     * Handle successful call
     */
    private void onSuccess(long durationNanos) {
        successCount.incrementAndGet();

        State currentState = state.get();
        if (currentState == State.HALF_OPEN) {
            // Check if enough successful calls to close circuit
            if (successCount.get() >= config.halfOpenSuccessThreshold()) {
                transitionToClosed();
            }
        }

        LOG.tracef("Circuit '%s' success: duration=%.2fms, state=%s, successes=%d",
                name, durationNanos / 1e6, currentState, successCount.get());
    }

    /**
     * Handle failed call
     */
    private void onFailure(Throwable throwable, long durationNanos) {
        // Check if this failure type should count
        if (!failurePredicate.test(throwable)) {
            LOG.tracef("Circuit '%s' ignoring failure type: %s",
                    name, throwable.getClass().getSimpleName());
            return;
        }

        int failures = failureCount.incrementAndGet();
        State currentState = state.get();

        LOG.debugf("Circuit '%s' failure: duration=%.2fms, state=%s, failures=%d, error=%s",
                name, durationNanos / 1e6, currentState, failures,
                throwable.getClass().getSimpleName());

        if (currentState == State.HALF_OPEN) {
            // Any failure in HALF_OPEN reopens circuit
            transitionToOpen();
        } else if (currentState == State.CLOSED) {
            // Check if failure threshold exceeded
            int total = successCount.get() + failures;
            if (total >= config.slidingWindowSize()) {
                double failureRate = (double) failures / total;
                if (failureRate >= config.failureRateThreshold()) {
                    transitionToOpen();
                }
            } else if (failures >= config.failureThreshold()) {
                // Absolute threshold check
                transitionToOpen();
            }
        }
    }

    /**
     * Transition to CLOSED state
     */
    private void transitionToClosed() {
        stateLock.lock();
        try {
            if (state.get() != State.CLOSED) {
                LOG.infof("Circuit breaker '%s' transitioning HALF_OPEN -> CLOSED", name);
                failureCount.set(0);
                successCount.set(0);
                transitionTo(State.CLOSED);
            }
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Transition to OPEN state
     */
    private void transitionToOpen() {
        stateLock.lock();
        try {
            if (state.get() != State.OPEN) {
                LOG.warnf("Circuit breaker '%s' transitioning %s -> OPEN (failures=%d, rate=%.2f%%)",
                        name, state.get(), failureCount.get(),
                        getMetrics().failureRate() * 100);
                transitionTo(State.OPEN);
            }
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Transition to HALF_OPEN state
     */
    private void transitionToHalfOpen() {
        stateLock.lock();
        try {
            if (state.compareAndSet(State.OPEN, State.HALF_OPEN)) {
                LOG.infof("Circuit breaker '%s' transitioning OPEN -> HALF_OPEN (testing recovery)",
                        name);
                failureCount.set(0);
                successCount.set(0);
                stateChangeTime.set(System.currentTimeMillis());
            }
        } finally {
            stateLock.unlock();
        }
    }

    /**
     * Transition to new state
     */
    private void transitionTo(State newState) {
        state.set(newState);
        stateChangeTime.set(System.currentTimeMillis());
    }

    /**
     * Get estimated recovery time in milliseconds
     */
    private long getEstimatedRecoveryTime() {
        if (state.get() != State.OPEN) {
            return 0;
        }

        long timeSinceOpen = System.currentTimeMillis() - stateChangeTime.get();
        long openDurationMs = config.openDuration().toMillis();

        return Math.max(0, openDurationMs - timeSinceOpen);
    }

    @Override
    public String toString() {
        CircuitBreaker.CircuitBreakerMetrics metrics = getMetrics();
        return String.format(
                "CircuitBreaker{name='%s', state=%s, failures=%d, successes=%d, rate=%.2f%%}",
                name, metrics.state(), metrics.getFailureCount(), metrics.getSuccessCount(),
                metrics.failureRate() * 100);
    }

    /**
     * Circuit breaker configuration
     */
    public record CircuitBreakerConfig(
            int failureThreshold,
            double failureRateThreshold,
            int slidingWindowSize,
            Duration openDuration,
            int halfOpenPermits,
            int halfOpenSuccessThreshold,
            Predicate<Throwable> failurePredicate) {
        public CircuitBreakerConfig {
            if (failureThreshold <= 0) {
                throw new IllegalArgumentException("failureThreshold must be positive");
            }
            if (failureRateThreshold <= 0 || failureRateThreshold > 1) {
                throw new IllegalArgumentException("failureRateThreshold must be in (0,1]");
            }
            if (slidingWindowSize < failureThreshold) {
                throw new IllegalArgumentException(
                        "slidingWindowSize must be >= failureThreshold");
            }
            if (openDuration == null || openDuration.isNegative()) {
                throw new IllegalArgumentException("openDuration must be positive");
            }
            if (halfOpenPermits <= 0) {
                throw new IllegalArgumentException("halfOpenPermits must be positive");
            }
            if (halfOpenSuccessThreshold <= 0 ||
                    halfOpenSuccessThreshold > halfOpenPermits) {
                throw new IllegalArgumentException(
                        "halfOpenSuccessThreshold must be in (0, halfOpenPermits]");
            }
            if (failurePredicate == null) {
                throw new IllegalArgumentException("failurePredicate cannot be null");
            }
        }

        /**
         * Default configuration
         */
        public static CircuitBreakerConfig defaults() {
            return new CircuitBreakerConfig(
                    5, // 5 failures
                    0.5, // 50% failure rate
                    10, // 10 call window
                    Duration.ofSeconds(60), // 60s open duration
                    3, // 3 test calls in half-open
                    2, // 2 successes to close
                    throwable -> true // Count all exceptions
            );
        }

        /**
         * Create builder
         */
        public static Builder builder() {
            return new Builder();
        }

        public static class Builder {
            private int failureThreshold = 5;
            private double failureRateThreshold = 0.5;
            private int slidingWindowSize = 10;
            private Duration openDuration = Duration.ofSeconds(60);
            private int halfOpenPermits = 3;
            private int halfOpenSuccessThreshold = 2;
            private Predicate<Throwable> failurePredicate = throwable -> true;

            public Builder failureThreshold(int threshold) {
                this.failureThreshold = threshold;
                return this;
            }

            public Builder failureRateThreshold(double threshold) {
                this.failureRateThreshold = threshold;
                return this;
            }

            public Builder slidingWindowSize(int size) {
                this.slidingWindowSize = size;
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

            public Builder failurePredicate(Predicate<Throwable> predicate) {
                this.failurePredicate = predicate;
                return this;
            }

            public CircuitBreakerConfig build() {
                return new CircuitBreakerConfig(
                        failureThreshold,
                        failureRateThreshold,
                        slidingWindowSize,
                        openDuration,
                        halfOpenPermits,
                        halfOpenSuccessThreshold,
                        failurePredicate);
            }
        }
    }
}