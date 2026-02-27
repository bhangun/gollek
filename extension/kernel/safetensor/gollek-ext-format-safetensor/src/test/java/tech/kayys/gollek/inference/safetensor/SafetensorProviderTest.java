package tech.kayys.gollek.inference.safetensor;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.inference.libtorch.LibTorchProvider;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.nio.file.Files;
import java.nio.file.Path;
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

        SafetensorProvider provider = newProvider(true, tempDir, ".safetensors,.safetensor");

        assertTrue(provider.supports("demo", null));
        assertTrue(provider.supports(file.toString(), null));
    }

    @Test
    void supportsSafetensorsExtensionToo() throws Exception {
        Path file = tempDir.resolve("demo2.safetensors");
        Files.writeString(file, "dummy");

        SafetensorProvider provider = newProvider(true, tempDir, ".safetensors,.safetensor");

        assertTrue(provider.supports("demo2", null));
    }

    @Test
    void inferDelegatesToLibTorchWithResolvedAbsolutePath() throws Exception {
        Path file = tempDir.resolve("delegate.safetensor");
        Files.writeString(file, "dummy");

        FakeLibTorchProvider delegate = new FakeLibTorchProvider();
        SafetensorProvider provider = newProvider(true, tempDir, ".safetensor");
        provider.libTorchProvider = delegate;

        ProviderRequest req = ProviderRequest.builder()
                .requestId("req-1")
                .model("delegate")
                .message(Message.user("hello"))
                .build();

        InferenceResponse response = provider.infer(req).await().indefinitely();

        assertEquals("delegated", response.getContent());
        assertNotNull(delegate.lastInferRequest);
        assertEquals(file.toString(), delegate.lastInferRequest.getModel());
    }

    @Test
    void inferStreamDelegatesToLibTorchWithResolvedAbsolutePath() throws Exception {
        Path file = tempDir.resolve("stream.safetensor");
        Files.writeString(file, "dummy");

        FakeLibTorchProvider delegate = new FakeLibTorchProvider();
        SafetensorProvider provider = newProvider(true, tempDir, ".safetensor");
        provider.libTorchProvider = delegate;

        ProviderRequest req = ProviderRequest.builder()
                .requestId("req-2")
                .model("stream")
                .message(Message.user("hello"))
                .build();

        List<StreamChunk> chunks = provider.inferStream(req)
                .collect().asList()
                .await().indefinitely();

        assertEquals(1, chunks.size());
        assertEquals("ok", chunks.get(0).getDelta());
        assertNotNull(delegate.lastStreamRequest);
        assertEquals(file.toString(), delegate.lastStreamRequest.getModel());
    }

    @Test
    void healthReturnsUnhealthyWhenDisabled() {
        SafetensorProvider provider = newProvider(false, tempDir, ".safetensors,.safetensor");

        ProviderHealth health = provider.health().await().indefinitely();

        assertEquals(ProviderHealth.Status.UNHEALTHY, health.status());
    }

    @Test
    void capabilitiesExposeSafetensorsFormat() {
        SafetensorProvider provider = newProvider(true, tempDir, ".safetensors,.safetensor");

        ProviderCapabilities capabilities = provider.capabilities();

        assertTrue(capabilities.getSupportedFormats().stream()
                .anyMatch(format -> "safetensors".equals(format.getId())));
        assertFalse(capabilities.getSupportedDevices().isEmpty());
    }

    private SafetensorProvider newProvider(boolean enabled, Path basePath, String extensions) {
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
        };
        provider.libTorchProvider = new FakeLibTorchProvider();
        return provider;
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
        public Multi<StreamChunk> inferStream(ProviderRequest request) {
            this.lastStreamRequest = request;
            return Multi.createFrom().item(StreamChunk.of(request.getRequestId(), 0, "ok"));
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
}
