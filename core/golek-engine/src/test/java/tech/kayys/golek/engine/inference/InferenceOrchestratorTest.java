package tech.kayys.golek.engine.inference;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.tenant.TenantContext;
import tech.kayys.golek.api.tenant.TenantId;
import tech.kayys.golek.api.model.ModelRunner;
import tech.kayys.golek.model.core.ModelRunnerFactory;
import tech.kayys.golek.engine.model.ModelRepository;
import tech.kayys.golek.engine.model.RunnerCandidate;
import tech.kayys.golek.engine.model.RequestContext;
import tech.kayys.golek.engine.observability.MetricsPublisher;
import tech.kayys.golek.engine.resource.CircuitBreaker;
import tech.kayys.golek.model.ModelManifest;
import tech.kayys.golek.engine.model.ModelRouterService;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class InferenceOrchestratorTest {

    @Mock
    private ModelRouterService router;

    @Mock
    private ModelRunnerFactory factory;

    @Mock
    private ModelRepository repository;

    @Mock
    private MetricsPublisher metrics;

    @Mock
    private CircuitBreaker circuitBreaker;

    @Mock
    private InferenceRequest request;

    @Mock
    private TenantContext tenantContext;

    @Mock
    private TenantId tenantId;

    @Mock
    private ModelManifest manifest;

    @Mock
    private ModelRunner runner;

    private InferenceOrchestrator orchestrator;

    @BeforeEach
    void setUp() {
        orchestrator = new InferenceOrchestrator(router, factory, repository, metrics, circuitBreaker);
        
        // Setup common mocks
        when(tenantContext.tenantId()).thenReturn(tenantId);
        when(tenantId.value()).thenReturn("test-tenant");
        when(request.priority()).thenReturn(1);
        when(request.deviceHint()).thenReturn("cpu");
        
        when(repository.findById(anyString(), any(TenantId.class)))
                .thenReturn(Optional.of(manifest));
        when(manifest.modelId()).thenReturn("test-model");
        
        when(factory.getRunner(any(ModelManifest.class), anyString(), any(TenantContext.class)))
                .thenReturn(runner);
    }

    @Test
    void testExecuteAsync_SuccessfulExecution() {
        // Arrange
        List<RunnerCandidate> candidates = Arrays.asList(
                new RunnerCandidate("runner1", "provider1", 90, true)
        );
        
        when(router.selectRunners(any(ModelManifest.class), any(RequestContext.class)))
                .thenReturn(candidates);
        
        when(circuitBreaker.call(any()))
                .thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));
        
        // Act
        var result = orchestrator.executeAsync("test-model", request, tenantContext);

        // Assert
        assertNotNull(result);
    }

    @Test
    void testExecute_ModelNotFound() {
        // Arrange
        when(repository.findById(anyString(), any(TenantId.class)))
                .thenReturn(Optional.empty());

        // Act & Assert
        assertThrows(RuntimeException.class, () -> {
            orchestrator.execute("non-existent-model", request, tenantContext);
        });
    }
}