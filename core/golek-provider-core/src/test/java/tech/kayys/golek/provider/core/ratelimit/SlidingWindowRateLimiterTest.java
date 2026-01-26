package tech.kayys.golek.provider.core.ratelimit;

import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import tech.kayys.golek.provider.core.ratelimit.SlidingWindowRateLimiter;

import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.*;

@QuarkusTest
class SlidingWindowRateLimiterTest {

    private SlidingWindowRateLimiter rateLimiter;

    @BeforeEach
    void setUp() {
        rateLimiter = new SlidingWindowRateLimiter(10, Duration.ofSeconds(1));
    }

    @Test
    void shouldAllowRequestsUpToLimit() {
        // Should allow 10 requests
        for (int i = 0; i < 10; i++) {
            assertThat(rateLimiter.tryAcquire()).isTrue();
        }

        // 11th request should be denied
        assertThat(rateLimiter.tryAcquire()).isFalse();
    }

    @Test
    void shouldReleasePermitsAfterWindow() throws InterruptedException {
        // Fill up the limiter
        for (int i = 0; i < 10; i++) {
            rateLimiter.tryAcquire();
        }

        // Should be full
        assertThat(rateLimiter.availablePermits()).isZero();

        // Wait for window to expire
        Thread.sleep(1100);

        // Should have permits again
        assertThat(rateLimiter.availablePermits()).isEqualTo(10);
        assertThat(rateLimiter.tryAcquire()).isTrue();
    }

    @Test
    void shouldHandleConcurrentRequests() throws InterruptedException {
        int threads = 20;
        int requestsPerThread = 10;
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch latch = new CountDownLatch(threads);
        AtomicInteger accepted = new AtomicInteger();
        AtomicInteger rejected = new AtomicInteger();

        for (int i = 0; i < threads; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < requestsPerThread; j++) {
                        if (rateLimiter.tryAcquire()) {
                            accepted.incrementAndGet();
                        } else {
                            rejected.incrementAndGet();
                        }
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await(5, TimeUnit.SECONDS);
        executor.shutdown();

        // Exactly 10 should be accepted
        assertThat(accepted.get()).isEqualTo(10);
        assertThat(rejected.get()).isEqualTo(threads * requestsPerThread - 10);
    }

    @Test
    void shouldResetCorrectly() {
        // Fill up
        for (int i = 0; i < 10; i++) {
            rateLimiter.tryAcquire();
        }

        assertThat(rateLimiter.availablePermits()).isZero();

        // Reset
        rateLimiter.reset();

        // Should be empty
        assertThat(rateLimiter.availablePermits()).isEqualTo(10);
    }

    @Test
    void shouldTrackMetrics() {
        rateLimiter.tryAcquire();
        rateLimiter.tryAcquire();

        for (int i = 0; i < 10; i++) {
            rateLimiter.tryAcquire(); // Fill up
        }

        var metrics = rateLimiter.getMetrics();
        assertThat(metrics.totalAccepted()).isEqualTo(10);
        assertThat(metrics.totalRejected()).isEqualTo(2);
    }
}
