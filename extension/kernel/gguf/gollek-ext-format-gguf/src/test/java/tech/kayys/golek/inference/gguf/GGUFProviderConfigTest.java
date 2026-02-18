package tech.kayys.gollek.inference.gguf;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.time.Duration;

import static org.assertj.core.api.Assertions.*;

/**
 * Unit tests for GGUFProviderConfig
 * 
 * These tests verify that configuration is properly loaded from
 * application.properties
 */
@QuarkusTest
class GGUFProviderConfigTest {

    @Inject
    GGUFProviderConfig config;

    @Test
    @DisplayName("Config should load default values")
    void testDefaultValues() {
        // When: Config is injected

        // Then: Default values should be present
        assertThat(config.modelBasePath()).isEqualTo("/var/lib/gollek/models/gguf");
        assertThat(config.maxContextTokens()).isEqualTo(4096);
        assertThat(config.gpuEnabled()).isTrue();
        assertThat(config.gpuLayers()).isEqualTo(32);
        assertThat(config.threads()).isEqualTo(4);
        assertThat(config.batchSize()).isEqualTo(512);
    }

    @Test
    @DisplayName("Config should have memory settings")
    void testMemorySettings() {
        // When: Getting memory settings

        // Then: Should have expected values
        assertThat(config.mmapEnabled()).isTrue();
        assertThat(config.mlockEnabled()).isFalse();
        assertThat(config.maxMemoryBytes()).isEqualTo(0); // 0 = unlimited
    }

    @Test
    @DisplayName("Config should have session pool settings")
    void testSessionPoolSettings() {
        // When: Getting session pool settings

        // Then: Should have expected values
        assertThat(config.sessionPoolMinSize()).isEqualTo(1);
        assertThat(config.sessionPoolMaxSize()).isEqualTo(4);
        assertThat(config.sessionPoolIdleTimeout()).isEqualTo(Duration.ofMinutes(5));
        assertThat(config.maxConcurrentRequests()).isEqualTo(10);
    }

    @Test
    @DisplayName("Config should have timeout settings")
    void testTimeoutSettings() {
        // When: Getting timeout settings

        // Then: Should have expected values
        assertThat(config.defaultTimeout()).isEqualTo(Duration.ofSeconds(30));
    }

    @Test
    @DisplayName("Config should have circuit breaker settings")
    void testCircuitBreakerSettings() {
        // When: Getting circuit breaker settings

        // Then: Should have expected values
        assertThat(config.circuitBreakerFailureThreshold()).isEqualTo(5);
        assertThat(config.circuitBreakerOpenDuration()).isEqualTo(Duration.ofSeconds(30));
        assertThat(config.circuitBreakerHalfOpenPermits()).isEqualTo(3);
        assertThat(config.circuitBreakerHalfOpenSuccessThreshold()).isEqualTo(2);
    }

    @Test
    @DisplayName("Config should have generation defaults")
    void testGenerationDefaults() {
        // When: Getting generation defaults

        // Then: Should have expected values
        assertThat(config.defaultTemperature()).isEqualTo(0.8f);
        assertThat(config.defaultTopP()).isEqualTo(0.95f);
        assertThat(config.defaultTopK()).isEqualTo(40);
        assertThat(config.defaultRepeatPenalty()).isEqualTo(1.1f);
        assertThat(config.defaultRepeatLastN()).isEqualTo(64);
    }

    @Test
    @DisplayName("Config should have health check settings")
    void testHealthCheckSettings() {
        // When: Getting health check settings

        // Then: Should have expected values
        assertThat(config.healthEnabled()).isTrue();
        assertThat(config.healthCheckInterval()).isEqualTo(Duration.ofSeconds(30));
    }

    @Test
    @DisplayName("Config should have metrics settings")
    void testMetricsSettings() {
        // When: Getting metrics settings

        // Then: Should have expected values
        assertThat(config.metricsEnabled()).isTrue();
    }

    @Test
    @DisplayName("Config should have prewarm settings")
    void testPrewarmSettings() {
        // When: Getting prewarm settings

        // Then: Should have expected values
        assertThat(config.prewarmEnabled()).isFalse();
        assertThat(config.prewarmModels()).isEmpty(); // Not configured by default
    }

    @Test
    @DisplayName("Config should have GPU settings")
    void testGpuSettings() {
        // When: Getting GPU settings

        // Then: Should have expected values
        assertThat(config.gpuEnabled()).isTrue();
        assertThat(config.gpuLayers()).isEqualTo(32);
        assertThat(config.gpuDeviceId()).isEqualTo(0);
    }

    @Test
    @DisplayName("All config values should be non-null")
    void testNoNullValues() {
        // When: Accessing all config methods

        // Then: None should return null
        assertThat(config.modelBasePath()).isNotNull();
        assertThat(config.sessionPoolIdleTimeout()).isNotNull();
        assertThat(config.defaultTimeout()).isNotNull();
        assertThat(config.circuitBreakerOpenDuration()).isNotNull();
        assertThat(config.healthCheckInterval()).isNotNull();
    }

    @Test
    @DisplayName("Duration values should be positive")
    void testPositiveDurations() {
        // When: Getting duration values

        // Then: All should be positive
        assertThat(config.sessionPoolIdleTimeout().toMillis()).isGreaterThan(0);
        assertThat(config.defaultTimeout().toMillis()).isGreaterThan(0);
        assertThat(config.circuitBreakerOpenDuration().toMillis()).isGreaterThan(0);
        assertThat(config.healthCheckInterval().toMillis()).isGreaterThan(0);
    }

    @Test
    @DisplayName("Integer values should be positive")
    void testPositiveIntegers() {
        // When: Getting integer config values

        // Then: All should be positive
        assertThat(config.maxContextTokens()).isGreaterThan(0);
        assertThat(config.threads()).isGreaterThan(0);
        assertThat(config.batchSize()).isGreaterThan(0);
        assertThat(config.sessionPoolMinSize()).isGreaterThan(0);
        assertThat(config.sessionPoolMaxSize()).isGreaterThanOrEqualTo(config.sessionPoolMinSize());
        assertThat(config.maxConcurrentRequests()).isGreaterThan(0);
    }

    @Test
    @DisplayName("Float values should be in valid ranges")
    void testFloatRanges() {
        // When: Getting float config values

        // Then: Should be in valid ranges
        assertThat(config.defaultTemperature()).isBetween(0.0f, 2.0f);
        assertThat(config.defaultTopP()).isBetween(0.0f, 1.0f);
        assertThat(config.defaultRepeatPenalty()).isGreaterThan(0.0f);
    }
}
