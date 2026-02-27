package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.provider.ProviderHealth;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LibTorchProviderTest {

    @TempDir
    Path tempDir;

    @Test
    void supportsModelIdWithConfiguredExtensionFallback() throws Exception {
        Path modelFile = tempDir.resolve("demo-model.custom");
        Files.writeString(modelFile, "dummy");

        LibTorchProvider provider = new LibTorchProvider();
        provider.config = config(true, tempDir, ".pt,.custom");
        setHealthy(provider);

        assertTrue(provider.supports("demo-model",
                ProviderRequest.builder().requestId("req-1").model("demo-model").message(Message.user("dummy"))
                        .build()));
    }

    @Test
    void supportsExtensionWithoutLeadingDotInConfig() throws Exception {
        Path modelFile = tempDir.resolve("another-model.xpt");
        Files.writeString(modelFile, "dummy");

        LibTorchProvider provider = new LibTorchProvider();
        provider.config = config(true, tempDir, "pt,xpt");
        setHealthy(provider);

        assertTrue(provider.supports("another-model",
                ProviderRequest.builder().requestId("req-2").model("another-model").message(Message.user("dummy"))
                        .build()));
    }

    @Test
    void returnsFalseWhenProviderDisabled() throws Exception {
        Path modelFile = tempDir.resolve("disabled-model.pt");
        Files.writeString(modelFile, "dummy");

        LibTorchProvider provider = new LibTorchProvider();
        provider.config = config(false, tempDir, ".pt");
        setHealthy(provider);

        assertFalse(provider.supports("disabled-model",
                ProviderRequest.builder().requestId("req-3").model("disabled-model").message(Message.user("dummy"))
                        .build()));
    }

    private void setHealthy(LibTorchProvider provider) throws Exception {
        Field statusField = LibTorchProvider.class.getDeclaredField("status");
        statusField.setAccessible(true);
        @SuppressWarnings("unchecked")
        AtomicReference<ProviderHealth.Status> statusRef = (AtomicReference<ProviderHealth.Status>) statusField
                .get(provider);
        statusRef.set(ProviderHealth.Status.HEALTHY);
    }

    private LibTorchProviderConfig config(boolean enabled, Path basePath, String extensions) {
        return new LibTorchProviderConfig() {
            @Override
            public boolean enabled() {
                return enabled;
            }

            @Override
            public NativeConfig nativeLib() {
                return () -> Optional.empty();
            }

            @Override
            public ModelConfig model() {
                return new ModelConfig() {
                    @Override
                    public String basePath() {
                        return basePath.toString();
                    }

                    @Override
                    public String extensions() {
                        return extensions;
                    }
                };
            }

            @Override
            public GpuConfig gpu() {
                return new GpuConfig() {
                    @Override
                    public boolean enabled() {
                        return false;
                    }

                    @Override
                    public int deviceIndex() {
                        return 0;
                    }
                };
            }

            @Override
            public SessionConfig session() {
                return new SessionConfig() {
                    @Override
                    public int maxPerTenant() {
                        return 4;
                    }

                    @Override
                    public int idleTimeoutSeconds() {
                        return 300;
                    }

                    @Override
                    public int maxTotal() {
                        return 16;
                    }
                };
            }

            @Override
            public InferenceConfig inference() {
                return new InferenceConfig() {
                    @Override
                    public int timeoutSeconds() {
                        return 30;
                    }

                    @Override
                    public int threads() {
                        return 4;
                    }
                };
            }

            @Override
            public BatchingConfig batching() {
                return new BatchingConfig() {
                    @Override
                    public boolean enabled() {
                        return false;
                    }

                    @Override
                    public int maxBatchSize() {
                        return 16;
                    }

                    @Override
                    public int batchTimeoutMs() {
                        return 50;
                    }
                };
            }

            @Override
            public WarmupConfig warmup() {
                return new WarmupConfig() {
                    @Override
                    public boolean enabled() {
                        return false;
                    }

                    @Override
                    public Optional<String> models() {
                        return Optional.empty();
                    }

                    @Override
                    public boolean dummyForward() {
                        return false;
                    }

                    @Override
                    public String tenantId() {
                        return "__warmup__";
                    }
                };
            }

            @Override
            public GenerationConfig generation() {
                return new GenerationConfig() {
                    @Override
                    public float temperature() {
                        return 0.8f;
                    }

                    @Override
                    public float topP() {
                        return 0.95f;
                    }

                    @Override
                    public int topK() {
                        return 40;
                    }

                    @Override
                    public int maxTokens() {
                        return 512;
                    }

                    @Override
                    public float repeatPenalty() {
                        return 1.1f;
                    }

                    @Override
                    public int repeatLastN() {
                        return 64;
                    }
                };
            }
        };
    }
}
