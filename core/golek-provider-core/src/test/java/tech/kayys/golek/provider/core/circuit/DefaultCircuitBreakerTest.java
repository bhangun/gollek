package tech.kayys.golek.provider.core.circuit;

import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import tech.kayys.golek.provider.core.circuit.DefaultCircuitBreaker;
import tech.kayys.golek.provider.core.circuit.CircuitBreaker;
import tech.kayys.golek.provider.core.circuit.CircuitBreakerOpenException;

import java.time.Duration;
import static org.assertj.core.api.Assertions.*;

@QuarkusTest
class DefaultCircuitBreakerTest {

    private DefaultCircuitBreaker circuitBreaker;

    @BeforeEach
    void setUp() {
        var config = DefaultCircuitBreaker.CircuitBreakerConfig.builder()
                .failureThreshold(3)
                .openDuration(Duration.ofMillis(500))
                .halfOpenPermits(2)
                .halfOpenSuccessThreshold(2)
                .build();

        circuitBreaker = new DefaultCircuitBreaker("test", config);
    }

    @Test
    void shouldStartInClosedState() {
        assertThat(circuitBreaker.getState()).isEqualTo(CircuitBreaker.State.CLOSED);
    }

    @Test
    void shouldOpenAfterFailureThreshold() {
        // Trigger 3 failures
        for (int i = 0; i < 3; i++) {
            assertThatThrownBy(() -> circuitBreaker
                    .call(io.smallrye.mutiny.Uni.createFrom().failure(new RuntimeException("Test failure"))).await()
                    .indefinitely())
                    .isInstanceOf(RuntimeException.class);
        }

        // Should be open
        assertThat(circuitBreaker.getState()).isEqualTo(CircuitBreaker.State.OPEN);
    }

    @Test
    void shouldRejectCallsWhenOpen() {
        // Open the circuit
        circuitBreaker.tripOpen();

        // Calls should fail immediately
        assertThatThrownBy(
                () -> circuitBreaker.call(io.smallrye.mutiny.Uni.createFrom().item("success")).await().indefinitely())
                .isInstanceOf(CircuitBreakerOpenException.class);
    }

    @Test
    void shouldTransitionToHalfOpenAfterTimeout() throws InterruptedException {
        // Open the circuit
        circuitBreaker.tripOpen();
        assertThat(circuitBreaker.getState()).isEqualTo(CircuitBreaker.State.OPEN);

        // Wait for open duration
        Thread.sleep(600);

        // Should transition to half-open
        assertThat(circuitBreaker.getState()).isEqualTo(CircuitBreaker.State.HALF_OPEN);
    }

    @Test
    void shouldCloseAfterSuccessfulTestCalls() throws Exception {
        // Open the circuit
        circuitBreaker.tripOpen();
        Thread.sleep(600); // Wait for half-open

        // Make successful test calls
        circuitBreaker.call(io.smallrye.mutiny.Uni.createFrom().item("success")).await().indefinitely();
        circuitBreaker.call(io.smallrye.mutiny.Uni.createFrom().item("success")).await().indefinitely();

        // Should be closed
        assertThat(circuitBreaker.getState()).isEqualTo(CircuitBreaker.State.CLOSED);
    }

    @Test
    void shouldReopenOnFailureInHalfOpen() throws Exception {
        // Open and transition to half-open
        circuitBreaker.tripOpen();
        Thread.sleep(600);

        assertThat(circuitBreaker.getState()).isEqualTo(CircuitBreaker.State.HALF_OPEN);

        // Fail a test call
        assertThatThrownBy(() -> circuitBreaker
                .call(io.smallrye.mutiny.Uni.createFrom().failure(new RuntimeException("Test failure"))).await()
                .indefinitely())
                .isInstanceOf(RuntimeException.class);

        // Should reopen
        assertThat(circuitBreaker.getState()).isEqualTo(CircuitBreaker.State.OPEN);
    }

    @Test
    void shouldTrackMetrics() throws Exception {
        circuitBreaker.call(io.smallrye.mutiny.Uni.createFrom().item("success")).await().indefinitely();

        assertThatThrownBy(
                () -> circuitBreaker.call(io.smallrye.mutiny.Uni.createFrom().failure(new RuntimeException("failure")))
                        .await().indefinitely());

        var metrics = circuitBreaker.getMetrics();
        assertThat(metrics.successCount()).isEqualTo(1);
        assertThat(metrics.failureCount()).isEqualTo(1);
        assertThat(metrics.totalRequests()).isEqualTo(2);
    }
}
