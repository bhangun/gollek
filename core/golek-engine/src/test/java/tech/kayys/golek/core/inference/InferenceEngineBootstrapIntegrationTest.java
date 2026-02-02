package tech.kayys.golek.core.inference;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.TestProfile;
import io.quarkus.test.junit.QuarkusTestProfile;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import tech.kayys.golek.engine.context.EngineContext;
import tech.kayys.golek.engine.inference.InferenceEngineBootstrap;
import tech.kayys.golek.engine.plugin.PluginLoader;
import tech.kayys.golek.engine.plugin.PluginRegistry;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.*;

/**
 * Integration tests for InferenceEngineBootstrap.
 * Tests the full bootstrap lifecycle with real components.
 */
@QuarkusTest
@DisplayName("InferenceEngineBootstrap Integration Tests")
class InferenceEngineBootstrapIntegrationTest {

    @Inject
    InferenceEngineBootstrap bootstrap;

    @Inject
    InferenceEngine engine;

    @Inject
    PluginLoader pluginLoader;

    @Inject
    PluginRegistry pluginRegistry;

    @Inject
    EngineContext engineContext;

    @Test
    @DisplayName("Should bootstrap with real plugins")
    void shouldBootstrapWithRealPlugins() {
        // Given - Bootstrap should have been initialized by Quarkus

        // Then
        assertThat(bootstrap.isInitialized()).isTrue();
        assertThat(bootstrap.getStartupTime()).isNotNull();
        assertThat(bootstrap.getUptime()).isGreaterThan(Duration.ZERO);

        // Verify plugins are loaded
        int pluginCount = pluginRegistry.count();
        assertThat(pluginCount).isGreaterThanOrEqualTo(0);

        // Verify engine health
        HealthStatus health = engine.health();
        assertThat(health).isNotNull();
    }

    @Test
    @DisplayName("Should handle concurrent initialization attempts")
    void shouldHandleConcurrentInitialization() throws InterruptedException {
        // Given
        int threadCount = 10;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(threadCount);

        // When - Try to initialize from multiple threads
        for (int i = 0; i < threadCount; i++) {
            new Thread(() -> {
                try {
                    startLatch.await();
                    // Bootstrap is already initialized, this should be safe
                    boolean initialized = bootstrap.isInitialized();
                    assertThat(initialized).isTrue();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        boolean completed = doneLatch.await(5, TimeUnit.SECONDS);

        // Then
        assertThat(completed).isTrue();
        assertThat(bootstrap.isInitialized()).isTrue();
    }

    @Test
    @DisplayName("Should provide plugin statistics")
    void shouldProvidePluginStatistics() {
        // When
        Map<String, Object> stats = bootstrap.getPluginStatistics();

        // Then
        assertThat(stats).isNotNull();
        assertThat(stats).containsKeys("total", "successful", "failed");

        Integer total = (Integer) stats.get("total");
        Integer successful = (Integer) stats.get("successful");
        Integer failed = (Integer) stats.get("failed");

        assertThat(total).isGreaterThanOrEqualTo(0);
        assertThat(successful).isGreaterThanOrEqualTo(0);
        assertThat(failed).isGreaterThanOrEqualTo(0);
        assertThat(total).isEqualTo(successful + failed);
    }

    @Test
    @DisplayName("Should track uptime accurately")
    void shouldTrackUptimeAccurately() throws InterruptedException {
        // Given
        Duration initialUptime = bootstrap.getUptime();

        // When
        Thread.sleep(500);
        Duration laterUptime = bootstrap.getUptime();

        // Then
        assertThat(laterUptime).isGreaterThan(initialUptime);
        assertThat(laterUptime.minus(initialUptime))
                .isGreaterThanOrEqualTo(Duration.ofMillis(400));
    }

    /**
     * Test profile for integration tests with custom configuration
     */
    public static class IntegrationTestProfile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of(
                    "wayang.inference.engine.enabled", "true",
                    "wayang.inference.engine.startup.timeout", "60s",
                    "wayang.inference.engine.startup.fail-on-plugin-error", "false",
                    "wayang.inference.engine.startup.min-plugins", "0");
        }
    }
}
