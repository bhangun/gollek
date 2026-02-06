package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.engine.observability.MetricsPublisher;
import tech.kayys.golek.engine.routing.ModelRouterService;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.wayang.tenant.TenantId;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class InferenceOrchestratorTest {

        @Mock
        private ModelRouterService router;

        @Mock
        private MetricsPublisher metrics;

        @Mock
        private InferenceRequest request;

        @Mock
        private TenantContext tenantContext;

        @Mock
        private TenantId tenantId;

        private InferenceOrchestrator orchestrator;

        @BeforeEach
        void setUp() {
                orchestrator = new InferenceOrchestrator(router, metrics);

                when(tenantContext.getTenantId()).thenReturn(tenantId);
                when(tenantId.value()).thenReturn("test-tenant");
        }

        @Test
        void testExecuteAsync_SuccessfulExecution() {
                // Arrange
                InferenceResponse expectedResponse = mock(InferenceResponse.class);
                when(router.route(anyString(), any(InferenceRequest.class), any(TenantContext.class)))
                                .thenReturn(Uni.createFrom().item(expectedResponse));

                // Act
                var result = orchestrator.executeAsync("test-model", request, tenantContext);

                // Assert
                InferenceResponse actualResponse = result.await().indefinitely();
                assertNotNull(actualResponse);
                assertEquals(expectedResponse, actualResponse);
                verify(metrics).recordSuccess(eq("unified"), eq("test-model"), anyLong());
        }

        @Test
        void testExecuteAsync_Failure() {
                // Arrange
                when(router.route(anyString(), any(InferenceRequest.class), any(TenantContext.class)))
                                .thenReturn(Uni.createFrom().failure(new RuntimeException("Test failure")));

                // Act & Assert
                assertThrows(RuntimeException.class, () -> {
                        orchestrator.executeAsync("test-model", request, tenantContext).await().indefinitely();
                });

                verify(metrics).recordFailure(eq("unified"), eq("test-model"), anyString());
        }
}