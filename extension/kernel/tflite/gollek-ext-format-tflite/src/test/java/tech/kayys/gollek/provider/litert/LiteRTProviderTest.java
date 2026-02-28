package tech.kayys.gollek.provider.litert;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LiteRTProviderTest {

    @TempDir
    Path tempDir;

    @Test
    void supportsReturnsFalseWhenAdapterRequested() throws Exception {
        LiteRTProvider provider = providerWithTempConfig();
        Path model = tempDir.resolve("demo.tflite");
        Files.writeString(model, "dummy");

        ProviderRequest request = ProviderRequest.builder()
                .model("demo")
                .message(Message.user("hello"))
                .parameter("adapter_type", "lora")
                .parameter("adapter_path", "x.safetensors")
                .build();

        assertFalse(provider.supports("demo", request));
    }

    @Test
    void inferRejectsAdapterRequests() throws Exception {
        LiteRTProvider provider = providerWithTempConfig();
        Path model = tempDir.resolve("demo.tflite");
        Files.writeString(model, "dummy");

        ProviderRequest request = ProviderRequest.builder()
                .model("demo")
                .message(Message.user("hello"))
                .parameter("adapter_type", "lora")
                .parameter("adapter_path", "x.safetensors")
                .build();

        UnsupportedOperationException ex = assertThrows(
                UnsupportedOperationException.class,
                () -> provider.infer(request).await().indefinitely());
        assertTrue(ex.getMessage().contains("adapter_unsupported"));
    }

    private LiteRTProvider providerWithTempConfig() {
        LiteRTProvider provider = new LiteRTProvider();
        provider.config = config(tempDir);
        provider.adapterMetricsRecorder = new NoopAdapterMetricsRecorder();
        LiteRTSessionManager manager = new LiteRTSessionManager();
        manager.config = provider.config;
        provider.sessionManager = manager;
        return provider;
    }

    private LiteRTProviderConfig config(Path basePath) {
        return new LiteRTProviderConfig() {
            @Override
            public String modelBasePath() {
                return basePath.toString();
            }

            @Override
            public int threads() {
                return 1;
            }

            @Override
            public boolean gpuEnabled() {
                return false;
            }

            @Override
            public boolean npuEnabled() {
                return false;
            }

            @Override
            public String gpuBackend() {
                return "auto";
            }

            @Override
            public String npuType() {
                return "auto";
            }

            @Override
            public Duration defaultTimeout() {
                return Duration.ofSeconds(1);
            }

            @Override
            public SessionConfig session() {
                return new SessionConfig() {
                    @Override
                    public int maxPerTenant() {
                        return 2;
                    }

                    @Override
                    public int idleTimeoutSeconds() {
                        return 300;
                    }

                    @Override
                    public int maxTotal() {
                        return 8;
                    }
                };
            }
        };
    }
}
