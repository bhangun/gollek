package tech.kayys.gollek.provider.litert;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LiteRTProviderTest {

    @TempDir
    Path tempDir;

    private String originalOsName;
    private String originalArch;
    private String originalMetalEnabled;
    private String originalMetalMode;

    @BeforeEach
    void captureProperties() {
        originalOsName = System.getProperty("os.name");
        originalArch = System.getProperty("os.arch");
        originalMetalEnabled = System.getProperty("gollek.runners.metal.enabled");
        originalMetalMode = System.getProperty("gollek.runners.metal.mode");
    }

    @AfterEach
    void restoreProperties() {
        restoreProperty("os.name", originalOsName);
        restoreProperty("os.arch", originalArch);
        restoreProperty("gollek.runners.metal.enabled", originalMetalEnabled);
        restoreProperty("gollek.runners.metal.mode", originalMetalMode);
    }

    @Test
    void supportsReturnsFalseWhenAdapterRequested() throws Exception {
        LiteRTProvider provider = providerWithTempConfig();
        Path model = tempDir.resolve("demo.litertlm");
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
        Path model = tempDir.resolve("demo.litertlm");
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

    @Test
    void capabilitiesExposeMetalWhenAutoMetalEnabledOnAppleSilicon() {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("os.arch", "arm64");
        System.setProperty("gollek.runners.metal.enabled", "true");
        System.setProperty("gollek.runners.metal.mode", "auto");

        LiteRTProvider provider = new LiteRTProvider();
        provider.config = configWithAutoMetal(tempDir);
        provider.adapterMetricsRecorder = new NoopAdapterMetricsRecorder();
        LiteRTSessionManager manager = new LiteRTSessionManager(provider.config);
        provider.sessionManager = manager;

        assertTrue(provider.capabilities().getSupportedDevices().contains(DeviceType.METAL));
    }

    @Test
    void autoMetalRespectsGlobalDisable() {
        System.setProperty("os.name", "Mac OS X");
        System.setProperty("os.arch", "arm64");
        System.setProperty("gollek.runners.metal.enabled", "false");

        LiteRTProvider provider = new LiteRTProvider();
        provider.config = configWithAutoMetal(tempDir);
        provider.adapterMetricsRecorder = new NoopAdapterMetricsRecorder();
        LiteRTSessionManager manager = new LiteRTSessionManager(provider.config);
        provider.sessionManager = manager;

        assertEquals(1, provider.capabilities().getSupportedDevices().size());
        assertTrue(provider.capabilities().getSupportedDevices().contains(DeviceType.CPU));
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    private LiteRTProvider providerWithTempConfig() {
        LiteRTProvider provider = new LiteRTProvider();
        provider.config = config(tempDir);
        provider.adapterMetricsRecorder = new NoopAdapterMetricsRecorder();
        LiteRTSessionManager manager = new LiteRTSessionManager(provider.config);
        provider.sessionManager = manager;
        return provider;
    }

    private LiteRTProviderConfig config(Path basePath) {
        return new LiteRTProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
            }

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
            public boolean autoMetalEnabled() {
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
            public LiteRTProviderConfig.SessionConfig session() {
                return new LiteRTProviderConfig.SessionConfig() {
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

    private LiteRTProviderConfig configWithAutoMetal(Path basePath) {
        return new LiteRTProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
            }

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
            public boolean autoMetalEnabled() {
                return true;
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
            public LiteRTProviderConfig.SessionConfig session() {
                return new LiteRTProviderConfig.SessionConfig() {
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
