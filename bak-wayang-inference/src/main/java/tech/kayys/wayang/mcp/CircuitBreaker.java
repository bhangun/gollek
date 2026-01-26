package tech.kayys.golek.mcp;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class CircuitBreaker {
    
    public enum State {
        CLOSED,      // Normal operation
        OPEN,        // Failure threshold exceeded, rejecting requests
        HALF_OPEN    // Testing if system recovered
    }
    
    private final int failureThreshold;
    private final Duration timeout;
    private final Duration resetTimeout;
    
    private final AtomicInteger failureCount = new AtomicInteger(0);
    private final AtomicInteger successCount = new AtomicInteger(0);
    private final AtomicLong lastFailureTime = new AtomicLong(0);
    private final AtomicReference<State> state = new AtomicReference<>(State.CLOSED);
    
    public CircuitBreaker(int failureThreshold, Duration timeout, Duration resetTimeout) {
        this.failureThreshold = failureThreshold;
        this.timeout = timeout;
        this.resetTimeout = resetTimeout;
    }
    
    public <T> T execute(Operation<T> operation) throws Exception {
        State currentState = getState();
        
        if (currentState == State.OPEN) {
            throw new CircuitBreakerOpenException("Circuit breaker is OPEN");
        }
        
        try {
            T result = operation.execute();
            onSuccess();
            return result;
        } catch (Exception e) {
            onFailure();
            throw e;
        }
    }
    
    private State getState() {
        State currentState = state.get();
        
        if (currentState == State.OPEN) {
            long timeSinceFailure = System.currentTimeMillis() - lastFailureTime.get();
            if (timeSinceFailure >= resetTimeout.toMillis()) {
                state.compareAndSet(State.OPEN, State.HALF_OPEN);
                return State.HALF_OPEN;
            }
        }
        
        return currentState;
    }
    
    private void onSuccess() {
        State currentState = state.get();
        
        if (currentState == State.HALF_OPEN) {
            successCount.incrementAndGet();
            if (successCount.get() >= 3) {
                state.set(State.CLOSED);
                failureCount.set(0);
                successCount.set(0);
            }
        } else if (currentState == State.CLOSED) {
            failureCount.set(0);
        }
    }
    
    private void onFailure() {
        lastFailureTime.set(System.currentTimeMillis());
        
        if (state.get() == State.HALF_OPEN) {
            state.set(State.OPEN);
            successCount.set(0);
        } else if (failureCount.incrementAndGet() >= failureThreshold) {
            state.set(State.OPEN);
        }
    }
    
    public State getCircuitState() {
        return getState();
    }
    
    public void reset() {
        state.set(State.CLOSED);
        failureCount.set(0);
        successCount.set(0);
    }
    
    @FunctionalInterface
    public interface Operation<T> {
        T execute() throws Exception;
    }
    
    public static class CircuitBreakerOpenException extends RuntimeException {
        public CircuitBreakerOpenException(String message) {
            super(message);
        }
    }
}
