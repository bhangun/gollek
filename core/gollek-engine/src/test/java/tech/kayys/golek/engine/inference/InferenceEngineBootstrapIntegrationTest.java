package tech.kayys.gollek.engine.inference;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.QuarkusTestProfile;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import tech.kayys.gollek.core.inference.InferenceEngine;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.engine.plugin.PluginLoader;
import tech.kayys.gollek.spi.plugin.PluginRegistry;
import tech.kayys.gollek.spi.model.HealthStatus;

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
@io.quarkus.test.junit.TestProfile(InferenceEngineBootstrapIntegrationTest.IntegrationTestProfile.class)
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
        // Changed from count() to all().size()
        int pluginCount = pluginRegistry.all().size();
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
            return Map.ofEntries(
                    Map.entry("wayang.inference.engine.enabled", "true"),
                    Map.entry("wayang.inference.engine.startup.timeout", "60s"),
                    Map.entry("wayang.inference.engine.startup.fail-on-plugin-error", "false"),
                    Map.entry("wayang.inference.engine.startup.min-plugins", "0"),
                    Map.entry("inference.model-storage.gcs.project-id", "test-project"),
                    Map.entry("inference.model-storage.s3.access-key-id", "test-key"),
                    Map.entry("inference.model-storage.s3.endpoint", "http://localhost:9000"),
                    Map.entry("inference.model-storage.s3.secret-access-key", "test-secret"),
                    Map.entry("inference.model-storage.azure.connection-string",
                            "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"),
                    Map.entry("quarkus.redis.hosts", "redis://localhost:6379"),
                    // Disable Keycloak DevService to prevent container startup issues
                    Map.entry("quarkus.keycloak.devservices.enabled", "false"),
                    // Provide a dummy auth server URL to satisfy OIDC extension
                    Map.entry("quarkus.oidc.auth-server-url", "http://localhost:8080/realms/test"));
        }
    }
}
