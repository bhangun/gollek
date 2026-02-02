package tech.kayys.golek.engine.model;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import jakarta.enterprise.inject.Instance;
import tech.kayys.golek.api.model.ModelFormat;
import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.api.model.RunnerMetadata;
import tech.kayys.golek.core.model.ModelRunner;
import tech.kayys.golek.engine.model.ModelRepository;
import tech.kayys.wayang.tenant.TenantConfigurationService;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.wayang.tenant.TenantId;

public class ModelRunnerFactoryTest {

    private ModelRunnerFactory factory;
    private ModelRepository repository;
    private TenantConfigurationService tenantConfigService;
    private Instance<ModelRunnerProvider> runnerProviders;

    @BeforeEach
    @SuppressWarnings("unchecked")
    void setUp() {
        repository = mock(ModelRepository.class);
        tenantConfigService = mock(TenantConfigurationService.class);
        runnerProviders = (Instance<ModelRunnerProvider>) mock(Instance.class);

        factory = new ModelRunnerFactory();
        factory.repository = repository;
        factory.tenantConfigService = tenantConfigService;
        factory.runnerProviders = runnerProviders;
    }

    @Test
    void testGetRunner_CachingBehavior() {
        TenantId t1 = new TenantId("t1");
        TenantContext ctx = TenantContext.of(t1);
        ModelManifest manifest = createManifest(t1, "m1");
        String runnerName = "test-runner";

        ModelRunner mockRunner = mock(ModelRunner.class);
        ModelRunnerProvider mockProvider = mock(ModelRunnerProvider.class);

        RunnerMetadata meta = new RunnerMetadata(runnerName, "1.0", List.of(ModelFormat.GGUF), List.of(),
                Collections.emptyMap());
        when(mockProvider.metadata()).thenReturn(meta);
        when(mockProvider.create(any(), any(), any())).thenReturn(mockRunner);

        when(runnerProviders.stream()).thenReturn(Stream.of(mockProvider));
        when(repository.findById("m1", t1)).thenReturn(Optional.of(manifest));
        when(tenantConfigService.getRunnerConfig(anyString(), anyString())).thenReturn(Collections.emptyMap());

        // First call - should create
        ModelRunner runner1 = factory.getRunner(manifest, runnerName, ctx);
        assertNotNull(runner1);

        // Second call - should return cached instance
        ModelRunner runner2 = factory.getRunner(manifest, runnerName, ctx);
        assertSame(runner1, runner2);

        // Verify create logic happened once via provider
        verify(mockProvider, times(1)).create(any(), any(), any());
    }

    @Test
    void testPrewarm() {
        TenantId t1 = new TenantId("t1");
        TenantContext ctx = TenantContext.of(t1);
        ModelManifest manifest = createManifest(t1, "m1");

        ModelRunner mockRunner = mock(ModelRunner.class);
        ModelRunnerProvider mockProvider = mock(ModelRunnerProvider.class);

        // Updated constructor
        RunnerMetadata meta = new RunnerMetadata("r1", "1.0", List.of(ModelFormat.GGUF), List.of(),
                Collections.emptyMap());
        when(mockProvider.metadata()).thenReturn(meta);
        when(mockProvider.create(any(), any(), any())).thenReturn(mockRunner);

        when(runnerProviders.stream()).thenReturn(Stream.of(mockProvider));
        when(repository.findById("m1", t1)).thenReturn(Optional.of(manifest));
        when(tenantConfigService.getRunnerConfig(anyString(), anyString())).thenReturn(Collections.emptyMap());

        factory.prewarm(manifest, List.of("r1"), ctx);

        // Check if cached
        ModelRunner cached = factory.getRunner(manifest, "r1", ctx);
        assertSame(mockRunner, cached);
    }

    private ModelManifest createManifest(TenantId tenantId, String modelId) {
        return new ModelManifest(
                modelId, "model", "1.0", tenantId,
                Collections.emptyMap(), Collections.emptyList(),
                null, Collections.emptyMap(), null, null);
    }
}
