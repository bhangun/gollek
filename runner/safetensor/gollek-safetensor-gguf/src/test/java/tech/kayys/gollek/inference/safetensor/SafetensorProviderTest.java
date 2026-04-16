package tech.kayys.gollek.inference.safetensor;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.inference.gguf.GGUFProvider;
import tech.kayys.gollek.inference.gguf.GGUFProviderConfig;
import tech.kayys.gollek.inference.libtorch.LibTorchProvider;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.AdapterCapabilityProfile;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.StreamingProvider;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafetensorProviderTest {

    @TempDir
    Path tempDir;

    @Test
    void supportsSuffixedAndUnsuffixedModelId() throws Exception {
        Path file = tempDir.resolve("demo.safetensor");
        Files.writeString(file, "dummy");

        SafetensorProvider provider = newProvider(true, tempDir, ".safetensors,.safetensor", "auto");

        assertTrue(provider.supports("demo", null));
        assertTrue(provider.supports(file.toString(), null));
    }

    @Test
    void supportsSafetensorsExtensionToo() throws Exception {
        Path file = tempDir.resolve("demo2.safetensors");
        Files.writeString(file, "dummy");

        SafetensorProvider provider = newProvider(true, tempDir, ".safetensors,.safetensor", "auto");

        assertTrue(provider.supports("demo2", null));
    }

    @Test
    void inferDelegatesToGgufAfterConversion() throws Exception {
        Path file = tempDir.resolve("delegate.safetensor");
        Files.writeString(file, "dummy");
        Files.writeString(tempDir.resolve("config.json"), "{}");

        FakeLibTorchProvider delegate = new FakeLibTorchProvider();
        SafetensorProvider provider = newProvider(true, tempDir, ".safetensor", "auto");
        provider.libTorchProvider = delegate;
        provider.ggufBackend = new FakeGgufBackend();
        provider.modelConverterService = new FakeModelConverterService(tempDir.resolve("gguf-out"));

        ProviderRequest req = ProviderRequest.builder()
                .requestId("req-1")
                .model("delegate")
                .message(Message.user("hello"))
                .build();

        InferenceResponse response = provider.infer(req).await().indefinitely();

        assertEquals("gguf-delegated", response.getContent());
        assertNotNull(((FakeGgufBackend) provider.ggufBackend).lastInferRequest);
    }

    @Test
    void inferStreamDelegatesToGgufAfterConversion() throws Exception {
        Path file = tempDir.resolve("stream.safetensor");
        Files.writeString(file, "dummy");
        Files.writeString(tempDir.resolve("config.json"), "{}");

        FakeLibTorchProvider delegate = new FakeLibTorchProvider();
        SafetensorProvider provider = newProvider(true, tempDir, ".safetensor", "auto");
        provider.libTorchProvider = delegate;
        provider.ggufBackend = new FakeGgufBackend();
        provider.modelConverterService = new FakeModelConverterService(tempDir.resolve("gguf-out"));

        ProviderRequest req = ProviderRequest.builder()
                .requestId("req-2")
                .model("stream")
                .message(Message.user("hello"))
                .build();

        List<StreamingInferenceChunk> chunks = provider.inferStream(req)
                .collect().asList()
                .await().indefinitely();

        assertEquals(1, chunks.size());
        assertEquals("gguf-ok", chunks.get(0).getDelta());
        assertNotNull(((FakeGgufBackend) provider.ggufBackend).lastStreamRequest);
    }

    @Test
    void healthReturnsUnhealthyWhenDisabled() {
        SafetensorProvider provider = newProvider(false, tempDir, ".safetensors,.safetensor", "auto");

        ProviderHealth health = provider.health().await().indefinitely();

        assertEquals(ProviderHealth.Status.UNHEALTHY, health.status());
    }

    @Test
    void capabilitiesExposeSafetensorsFormat() {
        SafetensorProvider provider = newProvider(true, tempDir, ".safetensors,.safetensor", "auto");

        ProviderCapabilities capabilities = provider.capabilities();

        assertTrue(capabilities.getSupportedFormats().stream()
                .anyMatch(format -> "safetensors".equals(format.getId())));
        assertFalse(capabilities.getSupportedDevices().isEmpty());
        assertTrue(capabilities.getFeatures().contains(AdapterCapabilityProfile.FEATURE_ADAPTER_SUPPORTED));
        assertTrue(capabilities.getFeatures().contains(AdapterCapabilityProfile.FEATURE_ADAPTER_METRICS_SCHEMA_V1));
    }

    @Test
    void capabilitiesExposeMetalWhenGgufAutoMetalIsAvailable() throws Exception {
        String originalOsName = System.getProperty("os.name");
        String originalOsArch = System.getProperty("os.arch");
        String originalUserHome = System.getProperty("user.home");
        String originalMetalEnabled = System.getProperty("gollek.runners.metal.enabled");
        String originalMetalMode = System.getProperty("gollek.runners.metal.mode");

        Path fakeHome = tempDir.resolve("home");
        Path metalLib = fakeHome.resolve(".gollek/libs/libggml-metal.dylib");
        Files.createDirectories(metalLib.getParent());
        Files.writeString(metalLib, "metal");

        System.setProperty("os.name", "Mac OS X");
        System.setProperty("os.arch", "aarch64");
        System.setProperty("user.home", fakeHome.toString());
        System.setProperty("gollek.runners.metal.enabled", "true");
        System.clearProperty("gollek.runners.metal.mode");

        try {
            GGUFProvider ggufProvider = new GGUFProvider(new MetalEnabledGGUFConfig(),
                    null, null, null, null, new NoopAdapterMetricsRecorder());
            GgufSafetensorBackend backend = new GgufSafetensorBackend();
            backend.ggufProvider = ggufProvider;

            SafetensorProvider provider = newProvider(true, tempDir, ".safetensors,.safetensor", "auto");
            provider.ggufBackend = backend;

            ProviderCapabilities capabilities = provider.capabilities();
            assertTrue(capabilities.getSupportedDevices().contains(DeviceType.METAL));
        } finally {
            restoreProperty("os.name", originalOsName);
            restoreProperty("os.arch", originalOsArch);
            restoreProperty("user.home", originalUserHome);
            restoreProperty("gollek.runners.metal.enabled", originalMetalEnabled);
            restoreProperty("gollek.runners.metal.mode", originalMetalMode);
        }
    }

    private SafetensorProvider newProvider(boolean enabled, Path basePath, String extensions, String backend) {
        SafetensorProvider provider = new SafetensorProvider();
        provider.config = new SafetensorProviderConfig() {
            @Override
            public boolean enabled() {
                return enabled;
            }

            @Override
            public String basePath() {
                return basePath.toString();
            }

            @Override
            public String extensions() {
                return extensions;
            }

            @Override
            public String backend() {
                return backend;
            }

            @Override
            public String ggufOutputDir() {
                return basePath.resolve("gguf-out").toString();
            }
        };
        provider.libTorchProvider = new FakeLibTorchProvider();
        return provider;
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    private static class FakeLibTorchProvider extends LibTorchProvider {
        ProviderRequest lastInferRequest;
        ProviderRequest lastStreamRequest;

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request) {
            this.lastInferRequest = request;
            return Uni.createFrom().item(InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(request.getModel())
                    .content("delegated")
                    .build());
        }

        @Override
        public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
            this.lastStreamRequest = request;
            return Multi.createFrom().item(StreamingInferenceChunk.of(request.getRequestId(), 0, "ok"));
        }

        @Override
        public Uni<ProviderHealth> health() {
            return Uni.createFrom().item(ProviderHealth.healthy("delegate healthy"));
        }

        @Override
        public ProviderCapabilities capabilities() {
            return ProviderCapabilities.builder()
                    .supportedDevices(Set.of(tech.kayys.gollek.spi.model.DeviceType.CPU))
                    .build();
        }
    }

    private static class FakeGgufBackend implements SafetensorGgufBackend {
        ProviderRequest lastInferRequest;
        ProviderRequest lastStreamRequest;

        @Override
        public void initialize(tech.kayys.gollek.spi.provider.ProviderConfig config) {
        }

        @Override
        public Uni<InferenceResponse> infer(ProviderRequest request) {
            this.lastInferRequest = request;
            return Uni.createFrom().item(InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .model(request.getModel())
                    .content("gguf-delegated")
                    .build());
        }

        @Override
        public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
            this.lastStreamRequest = request;
            return Multi.createFrom().item(StreamingInferenceChunk.of(request.getRequestId(), 0, "gguf-ok"));
        }

        @Override
        public Uni<ProviderHealth> health() {
            return Uni.createFrom().item(ProviderHealth.healthy("gguf healthy"));
        }

        @Override
        public ProviderCapabilities capabilities() {
            return ProviderCapabilities.builder()
                    .supportedDevices(Set.of(DeviceType.CPU))
                    .build();
        }
    }

    private static class MetalEnabledGGUFConfig implements GGUFProviderConfig {
        @Override
        public boolean enabled() {
            return true;
        }

        @Override
        public boolean verboseLogging() {
            return false;
        }

        @Override
        public java.util.Optional<String> nativeLibraryPath() {
            return java.util.Optional.empty();
        }

        @Override
        public java.util.Optional<String> nativeLibraryDir() {
            return java.util.Optional.empty();
        }

        @Override
        public String modelBasePath() {
            return System.getProperty("user.home") + "/.gollek/models/gguf";
        }

        @Override
        public int maxContextTokens() {
            return 2048;
        }

        @Override
        public boolean gpuEnabled() {
            return false;
        }

        @Override
        public boolean autoMetalEnabled() {
            return true;
        }

        @Override
        public int gpuLayers() {
            return 0;
        }

        @Override
        public int autoMetalLayers() {
            return -1;
        }

        @Override
        public int gpuDeviceId() {
            return 0;
        }

        @Override
        public int threads() {
            return 2;
        }

        @Override
        public int batchSize() {
            return 64;
        }

        @Override
        public boolean mmapEnabled() {
            return true;
        }

        @Override
        public boolean mlockEnabled() {
            return false;
        }

        @Override
        public int sessionPoolMinSize() {
            return 1;
        }

        @Override
        public int sessionPoolMaxSize() {
            return 2;
        }

        @Override
        public Duration sessionPoolIdleTimeout() {
            return Duration.ofMinutes(5);
        }

        @Override
        public int maxConcurrentRequests() {
            return 2;
        }

        @Override
        public boolean coalesceEnabled() {
            return false;
        }

        @Override
        public int coalesceWindowMs() {
            return 0;
        }

        @Override
        public int coalesceMaxBatch() {
            return 0;
        }

        @Override
        public int coalesceMaxQueue() {
            return 0;
        }

        @Override
        public int coalesceSeqMax() {
            return 1;
        }

        @Override
        public Duration defaultTimeout() {
            return Duration.ofSeconds(30);
        }

        @Override
        public int circuitBreakerFailureThreshold() {
            return 5;
        }

        @Override
        public Duration circuitBreakerOpenDuration() {
            return Duration.ofSeconds(30);
        }

        @Override
        public int circuitBreakerHalfOpenPermits() {
            return 2;
        }

        @Override
        public int circuitBreakerHalfOpenSuccessThreshold() {
            return 1;
        }

        @Override
        public boolean prewarmEnabled() {
            return false;
        }

        @Override
        public java.util.Optional<java.util.List<String>> prewarmModels() {
            return java.util.Optional.empty();
        }

        @Override
        public float defaultTemperature() {
            return 0.8f;
        }

        @Override
        public float defaultTopP() {
            return 0.95f;
        }

        @Override
        public int defaultTopK() {
            return 40;
        }

        @Override
        public float defaultRepeatPenalty() {
            return 1.1f;
        }

        @Override
        public boolean defaultJsonMode() {
            return false;
        }

        @Override
        public int defaultRepeatLastN() {
            return 64;
        }

        @Override
        public boolean healthEnabled() {
            return true;
        }

        @Override
        public Duration healthCheckInterval() {
            return Duration.ofSeconds(30);
        }

        @Override
        public long maxMemoryBytes() {
            return 0;
        }

        @Override
        public boolean metricsEnabled() {
            return true;
        }

        @Override
        public boolean loraEnabled() {
            return true;
        }

        @Override
        public String loraAdapterBasePath() {
            return System.getProperty("user.home") + "/.gollek/models/gguf/adapters";
        }

        @Override
        public float loraDefaultScale() {
            return 1.0f;
        }

        @Override
        public int loraMaxActiveAdapters() {
            return 128;
        }

        @Override
        public int loraMaxActiveAdaptersPerTenant() {
            return 16;
        }

        @Override
        public boolean loraRolloutGuardEnabled() {
            return false;
        }

        @Override
        public java.util.Optional<java.util.List<String>> loraRolloutAllowedTenants() {
            return java.util.Optional.empty();
        }

        @Override
        public java.util.Optional<java.util.List<String>> loraRolloutBlockedAdapterIds() {
            return java.util.Optional.empty();
        }

        @Override
        public java.util.Optional<java.util.List<String>> loraRolloutBlockedPathPrefixes() {
            return java.util.Optional.empty();
        }
    }

    private static class FakeModelConverterService extends tech.kayys.gollek.inference.gguf.ModelConverterService {
        private final Path outputDir;

        FakeModelConverterService(Path outputDir) {
            this.outputDir = outputDir;
        }

        @Override
        public boolean isAvailable() {
            return true;
        }

        @Override
        public void convert(Path inputDir, Path outputFile) throws java.io.IOException {
            java.nio.file.Files.createDirectories(outputDir);
            java.nio.file.Files.writeString(outputFile, "gguf");
        }
    }
}
