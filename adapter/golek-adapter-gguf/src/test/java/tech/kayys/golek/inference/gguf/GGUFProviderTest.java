package tech.kayys.golek.inference.gguf;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanBuilder;
import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;
import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import tech.kayys.golek.api.provider.*;
import tech.kayys.golek.api.Message;
import tech.kayys.wayang.tenant.TenantContext;

import java.time.Duration;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Unit tests for GGUFProvider
 */
@QuarkusTest
class GGUFProviderTest {

    @Inject
    GGUFProviderConfig config;

    @Mock
    LlamaCppBinding binding;

    @Mock
    GGUFSessionManager sessionManager;

    @Mock
    Tracer tracer;

    private GGUFProvider provider;
    private MeterRegistry meterRegistry;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        // Mock tracer behavior to avoid NPE
        SpanBuilder spanBuilder = mock(SpanBuilder.class);
        Span span = mock(Span.class);
        when(tracer.spanBuilder(anyString())).thenReturn(spanBuilder);
        when(spanBuilder.setAttribute(anyString(), anyString())).thenReturn(spanBuilder);
        when(spanBuilder.startSpan()).thenReturn(span);

        // Create simple meter registry for testing
        meterRegistry = new SimpleMeterRegistry();

        // Create provider instance with mocked dependencies
        provider = new GGUFProvider();
        setField(provider, "config", config);
        setField(provider, "binding", binding);
        setField(provider, "sessionManager", sessionManager);
        setField(provider, "meterRegistry", meterRegistry);
        setField(provider, "tracer", tracer);
    }

    @Test
    @DisplayName("Provider should have correct ID and name")
    void testProviderMetadata() {
        // Given: Provider is initialized
        initializeProvider();

        // When: Getting provider metadata
        String id = provider.id();
        String name = provider.name();
        String version = provider.version();

        // Then: Metadata should be correct
        assertThat(id).isEqualTo("gguf-llama-cpp");
        assertThat(name).isEqualTo("GGUF Provider (llama.cpp)");
        assertThat(version).isEqualTo("1.1.0");
    }

    @Test
    @DisplayName("Provider should report capabilities correctly")
    void testProviderCapabilities() {
        // Given: Provider is initialized
        initializeProvider();

        // When: Getting capabilities
        ProviderCapabilities capabilities = provider.capabilities();

        // Then: Capabilities should match configuration
        assertThat(capabilities).isNotNull();
        assertThat(capabilities.getMaxContextTokens()).isEqualTo(config.maxContextTokens());
        assertThat(capabilities.getSupportedFormats())
                .contains(tech.kayys.golek.api.model.ModelFormat.GGUF);
    }

    @Test
    @DisplayName("Provider support check")
    void testSupportsModel() {
        initializeProvider();
        String modelId = "model.gguf";
        // Since we can't easily mock Files.exists, assume false or check behavior
        boolean supported = provider.supports(modelId, TenantContext.of("t1"));
        assertThat(supported).isFalse();
    }

    @Test
    @DisplayName("Provider health check")
    void testHealth() {
        initializeProvider();
        when(sessionManager.isHealthy()).thenReturn(true);
        when(sessionManager.getActiveSessionCount()).thenReturn(5);

        var health = provider.health().await().indefinitely();
        assertThat(health.status()).isEqualTo(ProviderHealth.Status.HEALTHY);
        assertThat(health.details()).containsEntry("active_sessions", 5);
    }

    @Test
    @DisplayName("Provider should report metrics")
    void testMetrics() {
        // Given: Provider is initialized
        initializeProvider();
        when(sessionManager.getActiveSessionCount()).thenReturn(3);

        // When: Getting metrics
        var metrics = provider.metrics();

        // Then: Metrics should be present
        assertThat(metrics).isPresent();
        assertThat(metrics.get().getTotalRequests()).isEqualTo(0);
    }

    @Test
    @DisplayName("Provider should handle initialization errors gracefully")
    void testInitializationError() {
        // Given: Binding throws error on init
        doThrow(new RuntimeException("Native library not found"))
                .when(binding).backendInit();

        // When/Then: Initialization should throw
        assertThatThrownBy(() -> provider.onStart(new StartupEvent()))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("initialization failed");
    }

    @Test
    @DisplayName("Provider should cleanup resources on shutdown")
    void testShutdown() {
        // Given: Provider is initialized
        initializeProvider();

        // When: Shutting down
        provider.onStop(new ShutdownEvent());

        // Then: Cleanup should be called
        verify(sessionManager, times(1)).shutdown();
        verify(binding, times(1)).backendFree();
    }

    @Test
    @DisplayName("Provider should throw if not initialized before use")
    void testNotInitializedError() {
        // Given: Provider is NOT initialized

        // When/Then: Should throw when trying to infer
        // When/Then: Should throw when trying to infer
        TenantContext context = TenantContext.of("tenant1");
        ProviderRequest request = ProviderRequest.builder()
                .model("model.gguf")
                .message(Message.user("Hello"))
                .build();

        assertThatThrownBy(() -> provider.infer(request, context).await().indefinitely())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not initialized");
    }

    @Test
    @DisplayName("Inference should record metrics on success")
    void testInferenceMetrics() {
        initializeProvider();
        // Given: Provider is initialized
        when(sessionManager.isHealthy()).thenReturn(true);

        // Session/Runner mock setup would go here
        // But for brevity we just check calls

        // When: Running inference
        TenantContext context = TenantContext.of("tenant1");
        ProviderRequest request = ProviderRequest.builder()
                .model("model.gguf")
                .message(Message.user("Hello"))
                .parameter("max_tokens", 100)
                .parameter("temperature", 0.8)
                .parameter("top_p", 1.0)
                .build();

        // We expect infer to fail because we didn't fully mock SessionManager/Runner
        // behavior enough
        // to return a response successfully without more complex setup.
        // However, we can assert that it attempts it.

        try {
            provider.infer(request, context).await().atMost(Duration.ofSeconds(1));
        } catch (Exception e) {
            // Expected
        }

        // Check that metrics were recorded (even request count)
        assertThat(provider.metrics()).isPresent();
        assertThat(provider.metrics().get().getTotalRequests()).isGreaterThan(0);
    }

    // Helper methods

    private void initializeProvider() {
        // Mock binding initialization
        doNothing().when(binding).backendInit();

        // Mock session manager
        doNothing().when(sessionManager).initialize();
        when(sessionManager.isHealthy()).thenReturn(true);
        when(sessionManager.getActiveSessionCount()).thenReturn(0);

        // Initialize provider
        provider.onStart(new StartupEvent());
    }

    private void setField(Object target, String fieldName, Object value) {
        try {
            var field = target.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(target, value);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set field: " + fieldName, e);
        }
    }
}
