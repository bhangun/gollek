package tech.kayys.gollek.engine.routing.policy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import jakarta.enterprise.inject.Instance;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.model.core.HardwareDetector;
import tech.kayys.gollek.model.core.HardwareCapabilities;
import tech.kayys.gollek.model.core.RunnerMetrics;
import tech.kayys.gollek.engine.model.RunnerCandidate;
import tech.kayys.gollek.engine.model.ModelRunnerProvider;

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

                ModelManifest manifest = createManifest(List.of(new SupportedDevice(DeviceType.CPU, "1.0", null)));
                RequestContext cudaContext = new RequestContext(
                                "t1", "u1", "s1", "r1", "tr1", "n1", "or1", "on1", "or1", 0, 3, false, false,
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

                ModelManifest manifest = createManifest(List.of(new SupportedDevice(DeviceType.CPU, "1.0", null)));

                // Use factory or constructor for RequestContext
                RequestContext context = new RequestContext(
                                "t1", "u1", "s1", "r1", "tr1", "n1", "or1", "on1", "or1", 0, 3, false, false,
                                Optional.empty(), Duration.ofMinutes(1), false);

                List<RunnerCandidate> ranked = selectionPolicy.rankRunners(manifest, context,
                                List.of("runner-1", "runner-2"));

                assertEquals("runner-2", ranked.get(0).name());
        }

        private ModelRunnerProvider mockProvider(String name, List<ModelFormat> formats, List<DeviceType> devices) {
                ModelRunnerProvider provider = mock(ModelRunnerProvider.class);
                RunnerMetadata meta = new RunnerMetadata(name, "1.0", formats, devices, Collections.emptyMap());
                when(provider.metadata()).thenReturn(meta);
                return provider;
        }

        private ModelManifest createManifest(List<SupportedDevice> devices) {
                return new ModelManifest(
                                "m1", "model-1", "1.0", "t1", "test-api-key", "r1",
                                Collections.singletonMap(ModelFormat.GGUF,
                                                new ArtifactLocation("http://test", "sha256", 1024L,
                                                                "application/octet-stream")),
                                devices,
                                new ResourceRequirements(
                                                new MemoryRequirements(1L, null, null),
                                                new ComputeRequirements(null, null, null, null),
                                                new StorageRequirements(1L, null, null)),
                                Collections.emptyMap(),
                                Instant.now(), Instant.now());
        }
}
