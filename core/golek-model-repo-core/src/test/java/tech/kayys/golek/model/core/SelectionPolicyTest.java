package tech.kayys.golek.model.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import io.quarkus.runtime.configuration.MemorySize;
import jakarta.enterprise.inject.Instance;
import tech.kayys.golek.api.context.RequestContext;
import tech.kayys.golek.api.model.DeviceType;
import tech.kayys.golek.api.model.ModelFormat;
import tech.kayys.golek.api.model.ModelRunner;
import tech.kayys.golek.api.model.RunnerMetadata;
import tech.kayys.golek.api.tenant.TenantId;

public class SelectionPolicyTest {

    private SelectionPolicy selectionPolicy;
    private RunnerMetrics runnerMetrics;
    private HardwareDetector hardwareDetector;
    private Instance<ModelRunnerProvider> runnerProviders;

    @BeforeEach
    @SuppressWarnings("unchecked")
    void setUp() {
        runnerMetrics = mock(RunnerMetrics.class);
        hardwareDetector = mock(HardwareDetector.class);
        runnerProviders = (Instance<ModelRunnerProvider>) mock(Instance.class);

        selectionPolicy = new SelectionPolicy(runnerMetrics, hardwareDetector);
        selectionPolicy.runnerProviders = runnerProviders;

        HardwareCapabilities hw = HardwareCapabilities.builder()
                .hasCUDA(true)
                .availableMemory(16L * 1024 * 1024 * 1024)
                .cpuCores(8)
                .build();
        when(hardwareDetector.detect()).thenReturn(hw);

        when(runnerMetrics.isHealthy(anyString())).thenReturn(true);
        when(runnerMetrics.getCurrentLoad(anyString())).thenReturn(0.5);
    }

    @Test
    void testRankRunners_DeviceMatching() {
        ModelRunnerProvider cpuProvider = mockProvider("cpu-runner", List.of(ModelFormat.GGUF),
                List.of(DeviceType.CPU));
        ModelRunnerProvider cudaProvider = mockProvider("cuda-runner", List.of(ModelFormat.GGUF),
                List.of(DeviceType.CUDA));

        when(runnerProviders.stream()).thenAnswer(i -> Stream.of(cpuProvider, cudaProvider));

        ModelManifest manifest = createManifest(List.of(new SupportedDevice(DeviceType.CPU, 0, false)));
        RequestContext cudaContext = new RequestContext(
                new TenantId("t1"), "u1", "s1", "r1", "tr1", "n1", "or1", 0, 3, false, false,
                Optional.of(DeviceType.CUDA), Duration.ofMinutes(1), false);

        List<RunnerCandidate> ranked = selectionPolicy.rankRunners(manifest, cudaContext,
                List.of("cpu-runner", "cuda-runner"));

        assertFalse(ranked.isEmpty());
        assertEquals("cuda-runner", ranked.get(0).name());
    }

    @Test
    void testRankRunners_LoadBalancing() {
        ModelRunnerProvider p1 = mockProvider("runner-1", List.of(ModelFormat.GGUF), List.of(DeviceType.CPU));
        ModelRunnerProvider p2 = mockProvider("runner-2", List.of(ModelFormat.GGUF), List.of(DeviceType.CPU));

        when(runnerProviders.stream()).thenAnswer(i -> Stream.of(p1, p2));

        when(runnerMetrics.getCurrentLoad("runner-1")).thenReturn(0.9);
        when(runnerMetrics.getCurrentLoad("runner-2")).thenReturn(0.1);

        ModelManifest manifest = createManifest(List.of(new SupportedDevice(DeviceType.CPU, 0, false)));
        RequestContext context = RequestContext.create("t1", "u1", "s1");

        List<RunnerCandidate> ranked = selectionPolicy.rankRunners(manifest, context, List.of("runner-1", "runner-2"));

        assertEquals("runner-2", ranked.get(0).name());
    }

    private ModelRunnerProvider mockProvider(String name, List<ModelFormat> formats, List<DeviceType> devices) {
        ModelRunnerProvider provider = mock(ModelRunnerProvider.class);
        // Updated constructor: no execution mode arg
        RunnerMetadata meta = new RunnerMetadata(name, "1.0", formats, devices, Collections.emptyMap());
        when(provider.metadata()).thenReturn(meta);
        return provider;
    }

    private ModelManifest createManifest(List<SupportedDevice> devices) {
        MemorySize mem = mock(MemorySize.class);
        when(mem.asLongValue()).thenReturn(1024L * 1024);

        return new ModelManifest(
                "m1", "model-1", "1.0", new TenantId("t1"),
                Collections.singletonMap(ModelFormat.GGUF,
                        new ArtifactLocation("http://test", "sha256", "test", Collections.emptyMap())),
                devices,
                new ResourceRequirements(mem, mem, mem, Optional.empty(), Optional.empty()),
                Collections.emptyMap(),
                null, null);
    }
}
