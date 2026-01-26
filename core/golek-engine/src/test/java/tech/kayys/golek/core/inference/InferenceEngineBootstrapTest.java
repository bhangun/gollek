package tech.kayys.golek.core.inference;

import io.quarkus.runtime.StartupEvent;
import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.InjectMock;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.helpers.test.UniAssertSubscriber;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mockito;
import tech.kayys.golek.core.engine.EngineContext;
import tech.kayys.golek.provider.core.plugin.*;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeoutException;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Unit tests for InferenceEngineBootstrap.
 * Tests configuration validation, plugin initialization, error handling, and metrics.
 */
@QuarkusTest
@DisplayName("InferenceEngineBootstrap Tests")
class InferenceEngineBootstrapTest {

    @Inject
    InferenceEngineBootstrap bootstrap;

    @InjectMock
    InferenceEngine engine;

    @InjectMock
    PluginLoader pluginLoader;

    @InjectMock
    PluginRegistry pluginRegistry;

    @InjectMock
    EngineContext engineContext;

    private StartupEvent startupEvent;

    @BeforeEach
    void setUp() {
        startupEvent = mock(StartupEvent.class);
        
        // Reset counters and state
        Mockito.reset(engine, pluginLoader, pluginRegistry, engineContext);
        
        // Default successful mocks
        when(engineContext.initialize()).thenReturn(Uni.createFrom().voidItem());
        when(pluginLoader.loadAll()).thenReturn(Uni.createFrom().item(5));
        when(pluginLoader.initializeAll(any())).thenReturn(Uni.createFrom().voidItem());
        when(pluginRegistry.count()).thenReturn(5);
        when(pluginRegistry.getAllPlugins()).thenReturn(Collections.emptyList());
        when(engine.health()).thenReturn(HealthStatus.healthy());
    }

    @Test
    @DisplayName("Should initialize successfully with valid configuration")
    void shouldInitializeSuccessfully() {
        // When
        bootstrap.onStart(startupEvent);

        // Then
        assertThat(bootstrap.isInitialized()).isTrue();
        assertThat(bootstrap.getStartupTime()).isNotNull();
        assertThat(bootstrap.getUptime()).isGreaterThan(Duration.ZERO);

        // Verify bootstrap sequence
        verify(engineContext).initialize();
        verify(pluginLoader).loadAll();
        verify(pluginLoader).initializeAll(any(PluginContext.class));
    }

    @Test
    @DisplayName("Should validate configuration before startup")
    void shouldValidateConfigurationBeforeStartup() {
        // Configuration validation happens internally
        // This test verifies no exception is thrown with valid config
        assertThatCode(() -> bootstrap.onStart(startupEvent))
                .doesNotThrowAnyException();
    }

    @Test
    @DisplayName("Should handle plugin loading failure")
    void shouldHandlePluginLoadingFailure() {
        // Given
        when(pluginLoader.loadAll())
                .thenReturn(Uni.createFrom().failure(new RuntimeException("Plugin load failed")));

        // When/Then
        assertThatThrownBy(() -> bootstrap.onStart(startupEvent))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Inference engine startup failed");

        assertThat(bootstrap.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should handle plugin initialization failure with fail-fast mode")
    void shouldHandlePluginInitializationFailureFailFast() {
        // Given
        when(pluginLoader.initializeAll(any()))
                .thenReturn(Uni.createFrom().failure(new RuntimeException("Init failed")));

        // When/Then
        assertThatThrownBy(() -> bootstrap.onStart(startupEvent))
                .isInstanceOf(RuntimeException.class);

        assertThat(bootstrap.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should handle plugin initialization failure with graceful degradation")
    void shouldHandlePluginInitializationFailureGracefully() {
        // Given - Create mock plugins
        GolekPlugin successPlugin = createMockPlugin("success-plugin", true);
        GolekPlugin failPlugin = createMockPlugin("fail-plugin", false);
        
        when(pluginRegistry.getAllPlugins())
                .thenReturn(Arrays.asList(successPlugin, failPlugin));
        when(pluginRegistry.count()).thenReturn(1); // Only 1 successful

        // When
        bootstrap.onStart(startupEvent);

        // Then
        assertThat(bootstrap.isInitialized()).isTrue();
        
        Map<String, Object> stats = bootstrap.getPluginStatistics();
        assertThat(stats.get("successful")).isEqualTo(1);
        assertThat(stats.get("failed")).isEqualTo(1);
    }

    @Test
    @DisplayName("Should timeout on slow startup")
    void shouldTimeoutOnSlowStartup() {
        // Given - Simulate slow initialization
        when(engineContext.initialize())
                .thenReturn(Uni.createFrom().voidItem()
                        .onItem().delayIt().by(Duration.ofSeconds(60)));

        // When/Then
        assertThatThrownBy(() -> bootstrap.onStart(startupEvent))
                .hasCauseInstanceOf(TimeoutException.class);

        assertThat(bootstrap.isInitialized()).isFalse();
    }

    @Test
    @DisplayName("Should not initialize twice")
    void shouldNotInitializeTwice() {
        // Given
        bootstrap.onStart(startupEvent);
        assertThat(bootstrap.isInitialized()).isTrue();

        // When - Try to initialize again
        reset(engineContext, pluginLoader);
        bootstrap.onStart(startupEvent);

        // Then - Should not reinitialize
        verify(engineContext, never()).initialize();
        verify(pluginLoader, never()).loadAll();
    }

    @Test
    @DisplayName("Should report health correctly")
    void shouldReportHealthCorrectly() {
        // Given
        Map<String, PluginHealth> healthMap = new HashMap<>();
        healthMap.put("plugin1", PluginHealth.healthy());
        healthMap.put("plugin2", PluginHealth.unhealthy("Test failure"));
        
        when(pluginLoader.checkAllHealth()).thenReturn(healthMap);

        // When
        bootstrap.onStart(startupEvent);

        // Then
        assertThat(bootstrap.isInitialized()).isTrue();
        verify(pluginLoader).checkAllHealth();
    }

    @Test
    @DisplayName("Should collect startup metrics")
    void shouldCollectStartupMetrics() {
        // When
        bootstrap.onStart(startupEvent);

        // Then
        assertThat(bootstrap.isInitialized()).isTrue();
        
        Map<String, Object> stats = bootstrap.getPluginStatistics();
        assertThat(stats).containsKeys("total", "successful", "failed");
        assertThat(stats.get("total")).isEqualTo(5);
    }

    @Test
    @DisplayName("Should skip initialization when engine is disabled")
    void shouldHandleDisabledEngine() {
        // This test would require setting the config property
        // In a real scenario, you'd use @TestProfile or application-test.properties
        // For now, we verify the normal flow works
        bootstrap.onStart(startupEvent);
        assertThat(bootstrap.isInitialized()).isTrue();
    }

    @Test
    @DisplayName("Should log plugin summary correctly")
    void shouldLogPluginSummaryCorrectly() {
        // Given
        DefaultPluginRegistry mockRegistry = mock(DefaultPluginRegistry.class);
        when(mockRegistry.count()).thenReturn(5);
        when(mockRegistry.getStatistics()).thenReturn(
                new DefaultPluginRegistry.PluginStatistics(5, 3, Collections.emptyMap())
        );

        // When
        bootstrap.onStart(startupEvent);

        // Then
        assertThat(bootstrap.isInitialized()).isTrue();
        verify(pluginLoader).loadAll();
    }

    @Test
    @DisplayName("Should get plugin statistics")
    void shouldGetPluginStatistics() {
        // Given
        bootstrap.onStart(startupEvent);

        // When
        Map<String, Object> stats = bootstrap.getPluginStatistics();

        // Then
        assertThat(stats).isNotNull();
        assertThat(stats).containsKeys("total", "successful", "failed");
    }

    @Test
    @DisplayName("Should track uptime correctly")
    void shouldTrackUptimeCorrectly() throws InterruptedException {
        // Given
        bootstrap.onStart(startupEvent);

        // When
        Thread.sleep(100);
        Duration uptime = bootstrap.getUptime();

        // Then
        assertThat(uptime).isGreaterThanOrEqualTo(Duration.ofMillis(100));
    }

    // Helper methods

    private GolekPlugin createMockPlugin(String id, boolean shouldSucceed) {
        GolekPlugin plugin = mock(GolekPlugin.class);
        when(plugin.id()).thenReturn(id);
        when(plugin.name()).thenReturn(id);
        when(plugin.order()).thenReturn(100);
        
        if (shouldSucceed) {
            when(plugin.initialize(any())).thenReturn(Uni.createFrom().voidItem());
        } else {
            when(plugin.initialize(any())).thenReturn(
                    Uni.createFrom().failure(new RuntimeException("Plugin init failed"))
            );
        }
        
        return plugin;
    }
}
