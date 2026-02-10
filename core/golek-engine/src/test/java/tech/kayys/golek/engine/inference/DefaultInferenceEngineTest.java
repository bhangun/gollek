package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.context.EngineContext;
import tech.kayys.golek.core.engine.EngineMetadata;
import tech.kayys.golek.spi.model.HealthStatus;
import tech.kayys.golek.engine.execution.ExecutionStateMachine;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.wayang.tenant.TenantId;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DefaultInferenceEngineTest {

    @Mock
    private InferenceOrchestrator orchestrator;

    @Mock
    private ExecutionStateMachine stateMachine;

    @Mock
    private EngineContext engineContext;

    @Mock
    private InferenceMetrics metrics;

    @Mock
    private InferenceRequest request;

    @Mock
    private TenantContext tenantContext;

    @Mock
    private TenantId tenantIdMock;

    private DefaultInferenceEngine engine;

    @BeforeEach
    void setUp() {
        engine = new DefaultInferenceEngine();
        // Inject mocks via reflection if fields are private/protected/package-private
        // @Inject
        // Assuming DefaultInferenceEngine has standard CDI injection points.
        // We can try to use setters if available, or just use QuarkusComponentTest if
        // possible.
        // But this is a unit test with Mockito.
        // Let's assume field injection.
        try {
            injectField(engine, "orchestrator", orchestrator);
            injectField(engine, "metrics", metrics);
        } catch (Exception e) {
            fail("Failed to inject mocks: " + e.getMessage());
        }

        // Initialize engine
        engine.initialize();
    }

    private void injectField(Object target, String fieldName, Object value) throws Exception {
        java.lang.reflect.Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    @Test
    void testInfer_SuccessfulExecution() {
        // Arrange
        lenient().when(request.getRequestId()).thenReturn("test-request-id");
        lenient().when(tenantContext.getTenantId()).thenReturn(tenantIdMock);
        lenient().when(tenantIdMock.value()).thenReturn("test-tenant");
        when(request.getModel()).thenReturn("test-model");

        doReturn(Uni.createFrom().item(InferenceResponse.builder()
                .requestId("test-req")
                .content("test-content")
                .build()))
                .when(orchestrator).executeAsync(any(), any(), any());

        // Act
        Uni<InferenceResponse> result = engine.infer(request, tenantContext);

        // Assert
        assertNotNull(result);
        result.await().indefinitely();
        verify(orchestrator).executeAsync(any(), any(), any());
    }

    @Test
    void testMetadata_ReturnsCorrectMetadata() {
        // Act
        EngineMetadata metadata = engine.metadata();

        // Assert
        assertNotNull(metadata);
        assertEquals("1.0.0", metadata.version());
    }

    @Test
    void testHealth_ReturnsHealthy() {
        // Act
        HealthStatus status = engine.health();

        // Assert
        assertEquals("HEALTHY", status.status().name());
        assertEquals("Engine is operational", status.message());
    }
}