package tech.kayys.gollek.inference.libtorch;

import tech.kayys.gollek.inference.libtorch.config.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.spi.observability.AdapterSpec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class LibTorchSessionManagerAdapterRoutingTest {

    @TempDir
    Path tempDir;

    @Test
    void runtimeLoraSafetensorsKeepsBaseModelPath() throws Exception {
        Path modelsDir = tempDir.resolve("models");
        Path adaptersDir = tempDir.resolve("adapters");
        Files.createDirectories(modelsDir);
        Files.createDirectories(adaptersDir);

        Path baseModel = modelsDir.resolve("demo.pt");
        Path lora = adaptersDir.resolve("tenant-a.safetensors");
        Files.writeString(baseModel, "dummy");
        Files.writeString(lora, "dummy");

        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = config(modelsDir, adaptersDir, true);

        Path resolved = manager.resolveModelPath("demo", manager.config,
                new AdapterSpec("lora", "tenant-a", lora.toString(), 0.75f));

        assertThat(resolved).isEqualTo(baseModel);
    }

    @Test
    void precompiledAdapterModelOverridesBasePath() throws Exception {
        Path modelsDir = tempDir.resolve("models2");
        Path adaptersDir = tempDir.resolve("adapters2");
        Files.createDirectories(modelsDir);
        Files.createDirectories(adaptersDir);

        Path baseModel = modelsDir.resolve("demo.pt");
        Path mergedModel = adaptersDir.resolve("tenant-b.pt");
        Files.writeString(baseModel, "dummy");
        Files.writeString(mergedModel, "dummy");

        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = config(modelsDir, adaptersDir, true);

        Path resolved = manager.resolveModelPath("demo", manager.config,
                new AdapterSpec("lora", "tenant-b", mergedModel.toString(), 1.0f));

        assertThat(resolved).isEqualTo(mergedModel);
    }

    @Test
    void enforcesPerTenantAdapterPoolQuota() {
        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = config(tempDir.resolve("models3"), tempDir.resolve("adapters3"), true);

        AdapterSpec first = new AdapterSpec("lora", "a1", "/tmp/a1.safetensors", 1.0f);
        AdapterSpec second = new AdapterSpec("lora", "a2", "/tmp/a2.safetensors", 1.0f);
        AdapterSpec third = new AdapterSpec("lora", "a3", "/tmp/a3.safetensors", 1.0f);

        manager.enforceTenantAdapterQuota("tenant-x", "tenant-x:model:" + first.cacheKey(), first);
        manager.enforceTenantAdapterQuota("tenant-x", "tenant-x:model:" + second.cacheKey(), second);

        assertThatThrownBy(() -> manager.enforceTenantAdapterQuota(
                "tenant-x",
                "tenant-x:model:" + third.cacheKey(),
                third))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Adapter pool quota exceeded");
    }

    @Test
    void adaptiveIdleTimeoutRespondsToPressureTelemetry() {
        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = config(tempDir.resolve("models-pressure"), tempDir.resolve("adapters-pressure"), true);

        int baseline = manager.adaptiveIdleTimeoutSeconds();
        assertThat(baseline).isEqualTo(60);

        for (int i = 0; i < 8; i++) {
            manager.recordAdaptiveTelemetryForTest(true, 0);
        }

        int tightened = manager.adaptiveIdleTimeoutSeconds();
        assertThat(tightened).isLessThan(baseline);
        assertThat(manager.adaptivePressureScoreForTest()).isGreaterThan(0.60d);

        for (int i = 0; i < 12; i++) {
            manager.recordAdaptiveTelemetryForTest(false, 0);
        }

        int recovered = manager.adaptiveIdleTimeoutSeconds();
        assertThat(recovered).isGreaterThan(tightened);
        assertThat(manager.adaptivePressureScoreForTest()).isLessThan(0.35d);
    }

    private LibTorchProviderConfig config(Path modelBase, Path adapterBase, boolean allowPrecompiledPath) {
        return configWithRollout(modelBase, adapterBase, allowPrecompiledPath, false,
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    @Test
    void rolloutGuardBlocksNonAllowedTenant() throws Exception {
        Path modelsDir = tempDir.resolve("models4");
        Path adaptersDir = tempDir.resolve("adapters4");
        Files.createDirectories(modelsDir);
        Files.createDirectories(adaptersDir);

        Path baseModel = modelsDir.resolve("demo.pt");
        Path lora = adaptersDir.resolve("tenant-rollout.safetensors");
        Files.writeString(baseModel, "dummy");
        Files.writeString(lora, "dummy");

        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = configWithRollout(
                modelsDir,
                adaptersDir,
                true,
                true,
                Optional.of(List.of("tenant-allowed")),
                Optional.empty(),
                Optional.empty());

        assertThatThrownBy(() -> manager.resolveModelPath(
                "tenant-denied",
                "demo",
                manager.config,
                new AdapterSpec("lora", "tenant-rollout", lora.toString(), 1.0f)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not allowed for adapter rollout");
    }

    @Test
    void rolloutGuardBlocksDeniedAdapterId() throws Exception {
        Path modelsDir = tempDir.resolve("models5");
        Path adaptersDir = tempDir.resolve("adapters5");
        Files.createDirectories(modelsDir);
        Files.createDirectories(adaptersDir);

        Path baseModel = modelsDir.resolve("demo.pt");
        Path lora = adaptersDir.resolve("tenant-blocked-id.safetensors");
        Files.writeString(baseModel, "dummy");
        Files.writeString(lora, "dummy");

        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = configWithRollout(
                modelsDir,
                adaptersDir,
                true,
                true,
                Optional.empty(),
                Optional.of(List.of("blocked-id")),
                Optional.empty());

        assertThatThrownBy(() -> manager.resolveModelPath(
                "tenant-allowed",
                "demo",
                manager.config,
                new AdapterSpec("lora", "blocked-id", lora.toString(), 1.0f)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("blocked by rollout policy");
    }

    @Test
    void rolloutGuardBlocksDeniedPathPrefix() throws Exception {
        Path modelsDir = tempDir.resolve("models6");
        Path adaptersDir = tempDir.resolve("adapters6");
        Files.createDirectories(modelsDir);
        Files.createDirectories(adaptersDir);

        Path baseModel = modelsDir.resolve("demo.pt");
        Path lora = adaptersDir.resolve("tenant-blocked-path.safetensors");
        Files.writeString(baseModel, "dummy");
        Files.writeString(lora, "dummy");

        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = configWithRollout(
                modelsDir,
                adaptersDir,
                true,
                true,
                Optional.empty(),
                Optional.empty(),
                Optional.of(List.of(adaptersDir.toString())));

        assertThatThrownBy(() -> manager.resolveModelPath(
                "tenant-allowed",
                "demo",
                manager.config,
                new AdapterSpec("lora", "tenant-path", lora.toString(), 1.0f)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("blocked by rollout policy");
    }

    @Test
    void rolloutGuardAllowsAllowedTenant() throws Exception {
        Path modelsDir = tempDir.resolve("models7");
        Path adaptersDir = tempDir.resolve("adapters7");
        Files.createDirectories(modelsDir);
        Files.createDirectories(adaptersDir);

        Path baseModel = modelsDir.resolve("demo.pt");
        Path lora = adaptersDir.resolve("tenant-allowed.safetensors");
        Files.writeString(baseModel, "dummy");
        Files.writeString(lora, "dummy");

        LibTorchSessionManager manager = new LibTorchSessionManager();
        manager.config = configWithRollout(
                modelsDir,
                adaptersDir,
                true,
                true,
                Optional.of(List.of("tenant-allowed")),
                Optional.empty(),
                Optional.empty());

        Path resolved = manager.resolveModelPath(
                "tenant-allowed",
                "demo",
                manager.config,
                new AdapterSpec("lora", "allowed-id", lora.toString(), 1.0f));
        assertThat(resolved).isEqualTo(baseModel);
    }

    private LibTorchProviderConfig configWithRollout(
            Path modelBase,
            Path adapterBase,
            boolean allowPrecompiledPath,
            boolean rolloutGuardEnabled,
            Optional<List<String>> allowedTenants,
            Optional<List<String>> blockedAdapterIds,
            Optional<List<String>> blockedPathPrefixes) {
        return new LibTorchProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
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
                        return modelBase.toString();
                    }

                    @Override
                    public String extensions() {
                        return ".pt";
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
                    public boolean autoMpsEnabled() {
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
                        return 2;
                    }

                    @Override
                    public int idleTimeoutSeconds() {
                        return 60;
                    }

                    @Override
                    public int maxTotal() {
                        return 8;
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
                        return 2;
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
                        return 8;
                    }

                    @Override
                    public int batchTimeoutMs() {
                        return 20;
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
            public AdapterConfig adapter() {
                return new AdapterConfig() {
                    @Override
                    public boolean enabled() {
                        return true;
                    }

                    @Override
                    public String basePath() {
                        return adapterBase.toString();
                    }

                    @Override
                    public boolean allowPrecompiledModelPath() {
                        return allowPrecompiledPath;
                    }

                    @Override
                    public int maxActivePoolsPerTenant() {
                        return 0;
                    }

                    @Override
                    public boolean rolloutGuardEnabled() {
                        return rolloutGuardEnabled;
                    }

                    @Override
                    public Optional<List<String>> rolloutAllowedTenants() {
                        return allowedTenants;
                    }

                    @Override
                    public Optional<List<String>> rolloutBlockedAdapterIds() {
                        return blockedAdapterIds;
                    }

                    @Override
                    public Optional<List<String>> rolloutBlockedPathPrefixes() {
                        return blockedPathPrefixes;
                    }
                };
            }

            @Override
            public AdvancedConfig advanced() {
                return new AdvancedConfig() {
                    @Override
                    public boolean enabled() {
                        return false;
                    }

                    @Override
                    public String attentionMode() {
                        return "baseline";
                    }

                    @Override
                    public boolean fp8RowwiseEnabled() {
                        return false;
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseAllowedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseAllowedModels() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseBlockedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseBlockedModels() {
                        return Optional.empty();
                    }

                    @Override
                    public boolean sageAttention2Enabled() {
                        return false;
                    }

                    @Override
                    public Optional<List<String>> sageAttention2AllowedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2AllowedModels() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2BlockedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2BlockedModels() {
                        return Optional.empty();
                    }

                    @Override
                    public String allowedGpuSm() {
                        return "89,90";
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
                        return 128;
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
