package tech.kayys.gollek.inference.llamacpp;

import tech.kayys.gollek.gguf.tokenizer.GGUFChatTemplateService;
import tech.kayys.gollek.model.repo.local.ManifestStore;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Unit tests for LlamaCppSessionManager
 */
@QuarkusTest
class LlamaCppSessionManagerTest {

    @Inject
    LlamaCppProviderConfig config;

    @Mock
    LlamaCppBinding binding;

    @Mock
    GGUFChatTemplateService templateService;
    
    @Mock
    ManifestStore manifestStore;

    private LlamaCppSessionManager sessionManager;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        sessionManager = new LlamaCppSessionManager(binding, templateService, manifestStore);

        // Initialize the manager
        sessionManager.initialize();
    }

    @Test
    @DisplayName("SessionManager should initialize successfully")
    void testInitialization() {
        // Given/When: Manager is initialized in setUp

        // Then: Should be healthy
        assertThat(sessionManager.isHealthy()).isTrue();
        assertThat(sessionManager.getActiveSessionCount()).isEqualTo(0);
    }

    @Test
    @DisplayName("SessionManager should not double-initialize")
    void testDoubleInitialization() {
        // Given: Manager is already initialized

        // When: Trying to initialize again
        sessionManager.initialize();

        // Then: Should be idempotent
        assertThat(sessionManager.isHealthy()).isTrue();
    }

    @Test
    @DisplayName("SessionContext should track last used time")
    void testSessionContextTouch() {
        // Given: A session context
        var runner = mock(LlamaCppRunner.class);
        var context = new LlamaCppSessionManager.SessionContext(
                "session-1",
                runner,
                null,
                Instant.now(),
                Instant.now());

        // Wait a bit
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // When: Touching the session
        var touched = context.touch();

        // Then: Last used time should be updated
        assertThat(touched.lastUsedAt()).isAfter(context.lastUsedAt());
        assertThat(touched.sessionId()).isEqualTo(context.sessionId());
        assertThat(touched.runner()).isEqualTo(context.runner());
    }

    @Test
    @DisplayName("SessionContext should detect idle sessions")
    void testSessionIdleDetection() {
        // Given: A session created 10 seconds ago
        var runner = mock(LlamaCppRunner.class);
        var oldTime = Instant.now().minus(Duration.ofSeconds(10));
        var context = new LlamaCppSessionManager.SessionContext(
                "session-1",
                runner,
                null,
                oldTime,
                oldTime);

        // When/Then: Should be idle with 5 second timeout
        assertThat(context.isIdle(Duration.ofSeconds(5))).isTrue();

        // Should not be idle with 15 second timeout
        assertThat(context.isIdle(Duration.ofSeconds(15))).isFalse();
    }

    @Test
    @DisplayName("SessionManager should track active sessions")
    void testActiveSessionCount() {
        // Given: Manager is initialized
        int initialCount = sessionManager.getActiveSessionCount();

        // Note: We can't easily test actual session creation without
        // setting up full mocks for model loading, but we can verify
        // the counter starts at 0

        // Then: Initial count should be 0
        assertThat(initialCount).isEqualTo(0);
    }

    @Test
    @DisplayName("SessionManager should be healthy after initialization")
    void testHealthStatus() {
        // Given: Manager is initialized

        // When: Checking health
        boolean healthy = sessionManager.isHealthy();

        // Then: Should be healthy
        assertThat(healthy).isTrue();
    }

    @Test
    @DisplayName("SessionManager should be unhealthy after shutdown")
    void testHealthAfterShutdown() {
        // Given: Manager is initialized
        assertThat(sessionManager.isHealthy()).isTrue();

        // When: Shutting down
        sessionManager.shutdown();

        // Then: Should be unhealthy
        assertThat(sessionManager.isHealthy()).isFalse();
    }

    @Test
    @DisplayName("SessionManager should cleanup on shutdown")
    void testShutdown() {
        // Given: Manager is initialized

        // When: Shutting down
        sessionManager.shutdown();

        // Then: Should not be healthy
        assertThat(sessionManager.isHealthy()).isFalse();

        // And: Active session count should be 0
        assertThat(sessionManager.getActiveSessionCount()).isEqualTo(0);
    }

    @Test
    @DisplayName("SessionManager should not accept operations after shutdown")
    void testOperationsAfterShutdown() {
        // Given: Manager is shutdown
        sessionManager.shutdown();

        // When/Then: Operations should throw
        assertThatThrownBy(() -> sessionManager.getSession("tenant1", "model.gguf", config))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("shutdown");
    }

    @Test
    @DisplayName("SessionManager should handle concurrent session requests")
    void testConcurrentSessionRequests() throws InterruptedException {
        // Given: Manager is initialized
        int threadCount = 5;
        CountDownLatch latch = new CountDownLatch(threadCount);
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);

        // When: Multiple threads request sessions concurrently
        for (int i = 0; i < threadCount; i++) {
            final int threadNum = i;
            executor.submit(() -> {
                try {
                    // Note: This will likely fail without full model setup
                    // but it tests thread safety
                    sessionManager.getSession(
                            "tenant" + threadNum,
                            "model.gguf",
                            config);
                } catch (Exception e) {
                    // Expected - no model file exists
                } finally {
                    latch.countDown();
                }
            });
        }

        // Then: All threads should complete without deadlock
        boolean completed = latch.await(5, TimeUnit.SECONDS);
        assertThat(completed).isTrue();

        executor.shutdown();
    }

    @Test
    @DisplayName("SessionManager cleanup should be safe to call multiple times")
    void testMultipleCleanupCalls() {
        // Given: Manager is initialized

        // When: Calling cleanup multiple times
        sessionManager.cleanupIdleSessions();
        sessionManager.cleanupIdleSessions();
        sessionManager.cleanupIdleSessions();

        // Then: Should not throw
        assertThat(sessionManager.isHealthy()).isTrue();
    }

    @Test
    @DisplayName("SessionManager should not cleanup after shutdown")
    void testCleanupAfterShutdown() {
        // Given: Manager is shutdown
        sessionManager.shutdown();

        // When: Calling cleanup
        sessionManager.cleanupIdleSessions();

        // Then: Should be no-op (not throw)
        assertThat(sessionManager.isHealthy()).isFalse();
    }

    @Test
    @DisplayName("Adaptive idle timeout should tighten under sustained pressure and recover after decay")
    void testAdaptiveIdleTimeoutFeedbackLoop() {
        Duration baseline = sessionManager.resolveAdaptiveIdleTimeoutForTest(config);

        for (int i = 0; i < 8; i++) {
            sessionManager.recordAdaptiveTelemetryForTest(true, 0);
        }

        Duration tightened = sessionManager.resolveAdaptiveIdleTimeoutForTest(config);
        assertThat(tightened).isLessThan(baseline);
        assertThat(sessionManager.adaptivePressureScoreForTest()).isGreaterThan(0.60d);

        for (int i = 0; i < 12; i++) {
            sessionManager.recordAdaptiveTelemetryForTest(false, 0);
        }

        Duration recovered = sessionManager.resolveAdaptiveIdleTimeoutForTest(config);
        assertThat(recovered).isGreaterThan(tightened);
        assertThat(sessionManager.adaptivePressureScoreForTest()).isLessThan(0.35d);
    }

}
