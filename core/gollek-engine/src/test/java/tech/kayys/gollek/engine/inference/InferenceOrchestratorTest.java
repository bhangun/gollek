package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.engine.observability.MetricsPublisher;
import tech.kayys.gollek.engine.routing.ModelRouterService;

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
        private RequestContext requestContext;

        private final String requestId = "test-tenant";

        private DefaultInferenceOrchestrator orchestrator;

        @BeforeEach
        void setUp() {
                orchestrator = new DefaultInferenceOrchestrator(router, metrics);

                when(requestContext.requestId()).thenReturn(requestId);
        }

        @Test
        void testExecuteAsync_SuccessfulExecution() {
                // Arrange
                InferenceResponse expectedResponse = mock(InferenceResponse.class);
                when(router.route(anyString(), any(InferenceRequest.class)))
                                .thenReturn(Uni.createFrom().item(expectedResponse));

                // Act
                var result = orchestrator.executeAsync("test-model", request);

                // Assert
                InferenceResponse actualResponse = result.await().indefinitely();
                assertNotNull(actualResponse);
                assertEquals(expectedResponse, actualResponse);
                verify(metrics).recordSuccess(eq("unified"), eq("test-model"), anyLong());
        }

        @Test
        void testExecuteAsync_Failure() {
                // Arrange
                when(router.route(anyString(), any(InferenceRequest.class)))
                                .thenReturn(Uni.createFrom().failure(new RuntimeException("Test failure")));

                // Act & Assert
                assertThrows(RuntimeException.class, () -> {
                        orchestrator.executeAsync("test-model", request).await().indefinitely();
                });

                verify(metrics).recordFailure(eq("unified"), eq("test-model"), anyString());
        }
}