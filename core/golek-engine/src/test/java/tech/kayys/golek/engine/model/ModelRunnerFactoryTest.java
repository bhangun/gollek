package tech.kayys.golek.engine.model;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.time.Instant;
import java.util.Collections;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import jakarta.enterprise.inject.Instance;

import tech.kayys.golek.spi.model.DeviceType;
import tech.kayys.golek.spi.model.ModelManifest;
import tech.kayys.golek.spi.model.ResourceMetrics;
import tech.kayys.golek.spi.model.RunnerMetadata;

public class ModelRunnerFactoryTest {

    private ModelRunnerFactory factory;
    private Instance<ModelRunner> runnerProviders;

    @BeforeEach
    @SuppressWarnings("unchecked")
    void setUp() {
        runnerProviders = (Instance<ModelRunner>) mock(Instance.class);

        factory = new ModelRunnerFactory();
        factory.runnerProviders = runnerProviders;
        factory.warmPoolEnabled = true;
        factory.maxPoolSize = 10;
    }

    // Dummy Runner Class for testing instantiation
    public static class TestModelRunner implements ModelRunner {
        @Override
        public String name() {
            return "test-runner";
        }

        @Override
        public String framework() {
            return "test";
        }

        @Override
        public DeviceType deviceType() {
            return DeviceType.CPU;
        }

        @Override
        public RunnerCapabilities capabilities() {
            return null;
        }

        @Override
        public RunnerMetrics metrics() {
            return null;
        }

        @Override
        public void initialize(ModelManifest manifest, tech.kayys.golek.engine.model.RunnerConfiguration config) {
        }

        @Override
        public tech.kayys.golek.spi.inference.InferenceResponse infer(
                tech.kayys.golek.spi.inference.InferenceRequest request) {
            return null;
        }

        @Override
        public java.util.concurrent.CompletionStage<tech.kayys.golek.spi.inference.InferenceResponse> inferAsync(
                tech.kayys.golek.spi.inference.InferenceRequest request) {
            return null;
        }

        @Override
        public boolean health() {
            return true;
        }

        @Override
        public ResourceMetrics getMetrics() {
            return null;
        }

        @Override
        public RunnerMetadata metadata() {
            return null;
        }

        @Override
        public void close() {
        }
    }

    @Test
    void testGetOrCreateRunner_CachingBehavior() {
        String requestIdValue = "t1";
        ModelManifest manifest = createManifest(requestIdValue, "m1");
        String runnerName = "test-runner";

        TestModelRunner templateRunner = new TestModelRunner();

        // Mock iterator to return templateRunner
        when(runnerProviders.iterator())
                .thenAnswer(i -> Collections.singletonList((ModelRunner) templateRunner).iterator());

        factory.init(); // Initialize available runners

        // First call - should create
        ModelRunner runner1 = factory.getOrCreateRunner(runnerName, manifest);
        assertNotNull(runner1);

        // Second call - should return cached instance
        ModelRunner runner2 = factory.getOrCreateRunner(runnerName, manifest);
        assertSame(runner1, runner2);
    }

    // Helper
    private ModelManifest createManifest(String requestId, String modelId) {
        return new ModelManifest(
                modelId, "model", "1.0", requestId,
                Collections.emptyMap(), Collections.emptyList(),
                null, Collections.emptyMap(), Instant.now(), Instant.now());
    }
}
