package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.spi.provider.ProviderRequest;
import org.junit.jupiter.api.io.TempDir;

import tech.kayys.gollek.inference.libtorch.config.*;
import tech.kayys.gollek.inference.libtorch.config.AdapterConfig;
import tech.kayys.gollek.inference.libtorch.config.AdvancedConfig;
import tech.kayys.gollek.inference.libtorch.config.BatchingConfig;
import tech.kayys.gollek.inference.libtorch.config.GenerationConfig;
import tech.kayys.gollek.inference.libtorch.config.GpuConfig;
import tech.kayys.gollek.inference.libtorch.config.InferenceConfig;
import tech.kayys.gollek.inference.libtorch.config.LibTorchProviderConfig;
import tech.kayys.gollek.inference.libtorch.config.NativeConfig;
import tech.kayys.gollek.inference.libtorch.config.SessionConfig;
import tech.kayys.gollek.inference.libtorch.config.WarmupConfig;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.ProviderHealth;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
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

    @Test
    void resolveExecutionHintsUsesBaselineByDefault() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider);
        assertFalse(hints.hybridFp8Bf16AttentionEnabled());
        assertFalse(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("none", hints.sageAttention2Reason());
        assertFalse(hints.fp8RowwiseEnabled());
        assertEquals(0, hints.fp8RowwiseScaleCount());
        assertEquals(0.0d, hints.fp8RowwiseScaleMean());
        assertEquals("none", hints.fp8RowwiseCalibrationSource());
    }

    @Test
    void resolveExecutionHintsEnablesHybridWhenEffectiveModeIsHybrid() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                false,
                false,
                false,
                "none",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider);
        assertTrue(hints.hybridFp8Bf16AttentionEnabled());
        assertFalse(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("none", hints.sageAttention2Reason());
        assertFalse(hints.fp8RowwiseEnabled());
        assertEquals(0, hints.fp8RowwiseScaleCount());
        assertEquals(0.0d, hints.fp8RowwiseScaleMean());
    }

    @Test
    void resolveExecutionHintsEnablesFp8RowwiseWhenCalibrationExists() throws Exception {
        Path model = tempDir.resolve("model.pt");
        Files.writeString(model, "x");
        Files.writeString(tempDir.resolve("model.pt.fp8.json"), "{\"version\":\"1\",\"row_scales\":[1.0]}");

        LibTorchProvider provider = new LibTorchProvider();
        provider.config = config(true, tempDir, ".pt");
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                true,
                false,
                false,
                "none",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider, model);
        assertTrue(hints.hybridFp8Bf16AttentionEnabled());
        assertFalse(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("none", hints.sageAttention2Reason());
        assertTrue(hints.fp8RowwiseEnabled());
        assertTrue(hints.fp8RowwiseReason().contains("enabled"));
        assertEquals(1, hints.fp8RowwiseScaleCount());
        assertEquals(1.0d, hints.fp8RowwiseScaleMean());
        assertTrue(hints.fp8RowwiseCalibrationSource().contains("model.pt.fp8.json"));
    }

    @Test
    void resolveExecutionHintsCarriesSageAttentionRollbackState() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        provider.config = config(true, tempDir, ".pt");
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                false,
                true,
                false,
                "sageattention2.not-implemented",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider);
        assertTrue(hints.hybridFp8Bf16AttentionEnabled());
        assertTrue(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("sageattention2.not-implemented", hints.sageAttention2Reason());
    }

    @Test
    void resolveExecutionHintsPreservesSageRollbackWhenAdvancedFallsBack() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        provider.config = config(true, tempDir, ".pt");
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                false,
                "baseline",
                false,
                true,
                false,
                "sageattention2.not-implemented",
                "sageattention2.rollback",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider);
        assertFalse(hints.hybridFp8Bf16AttentionEnabled());
        assertTrue(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("sageattention2.not-implemented", hints.sageAttention2Reason());
    }

    @Test
    void resolveExecutionHintsBlocksSageAttention2WhenTenantNotInCanaryList() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        provider.config = configWithSageCanary(Optional.of(List.of("tenant-allowed")), Optional.empty());
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                false,
                true,
                false,
                "sageattention2.not-implemented",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider, tempDir.resolve("dummy.pt"), "tenant-x",
                "model-a");
        assertTrue(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("sageattention2.canary.blocked.tenant", hints.sageAttention2Reason());
    }

    @Test
    void resolveExecutionHintsBlocksSageAttention2WhenModelNotInCanaryList() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        provider.config = configWithSageCanary(Optional.empty(), Optional.of(List.of("model-allowed")));
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                false,
                true,
                false,
                "sageattention2.not-implemented",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider, tempDir.resolve("dummy.pt"), "tenant-a",
                "model-x");
        assertTrue(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("sageattention2.canary.blocked.model", hints.sageAttention2Reason());
    }

    @Test
    void resolveExecutionHintsDeniesSageAttention2WhenTenantInDenyList() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        provider.config = configWithSageCanary(
                Optional.of(List.of("tenant-allowed")),
                Optional.empty(),
                Optional.of(List.of("tenant-denied")),
                Optional.empty());
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                false,
                true,
                false,
                "sageattention2.not-implemented",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider, tempDir.resolve("dummy.pt"),
                "tenant-denied", "model-a");
        assertTrue(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("sageattention2.canary.denied.tenant", hints.sageAttention2Reason());
    }

    @Test
    void resolveExecutionHintsDeniesSageAttention2WhenModelInDenyList() throws Exception {
        LibTorchProvider provider = new LibTorchProvider();
        provider.config = configWithSageCanary(
                Optional.empty(),
                Optional.of(List.of("model-allowed")),
                Optional.empty(),
                Optional.of(List.of("model-denied")));
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                false,
                true,
                false,
                "sageattention2.not-implemented",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider, tempDir.resolve("dummy.pt"), "tenant-a",
                "model-denied");
        assertTrue(hints.sageAttention2Requested());
        assertFalse(hints.sageAttention2Enabled());
        assertEquals("sageattention2.canary.denied.model", hints.sageAttention2Reason());
    }

    @Test
    void resolveExecutionHintsBlocksFp8RowwiseWhenTenantNotInCanaryList() throws Exception {
        Path model = tempDir.resolve("model.pt");
        Files.writeString(model, "x");
        Files.writeString(tempDir.resolve("model.pt.fp8.json"), "{\"version\":\"1\",\"row_scales\":[1.0]}");

        LibTorchProvider provider = new LibTorchProvider();
        provider.config = configWithFp8RowwiseCanary(
                Optional.of(List.of("tenant-allowed")),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                true,
                false,
                false,
                "none",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider, model, "tenant-x", "model-a");
        assertFalse(hints.fp8RowwiseEnabled());
        assertEquals("fp8.rowwise.canary.blocked.tenant", hints.fp8RowwiseReason());
    }

    @Test
    void resolveExecutionHintsDeniesFp8RowwiseWhenModelInDenyList() throws Exception {
        Path model = tempDir.resolve("model.pt");
        Files.writeString(model, "x");
        Files.writeString(tempDir.resolve("model.pt.fp8.json"), "{\"version\":\"1\",\"row_scales\":[1.0]}");

        LibTorchProvider provider = new LibTorchProvider();
        provider.config = configWithFp8RowwiseCanary(
                Optional.empty(),
                Optional.of(List.of("model-allowed")),
                Optional.empty(),
                Optional.of(List.of("model-denied")));
        setEffectiveMode(provider, new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                true,
                "hybrid_fp8_bf16",
                true,
                false,
                false,
                "none",
                "advanced.enabled",
                Optional.of(90),
                Set.of(89, 90)));

        LibTorchExecutionHints hints = invokeResolveExecutionHints(provider, model, "tenant-a", "model-denied");
        assertFalse(hints.fp8RowwiseEnabled());
        assertEquals("fp8.rowwise.canary.denied.model", hints.fp8RowwiseReason());
    }

    private void setHealthy(LibTorchProvider provider) throws Exception {
        Field statusField = LibTorchProvider.class.getDeclaredField("status");
        statusField.setAccessible(true);
        @SuppressWarnings("unchecked")
        AtomicReference<ProviderHealth.Status> statusRef = (AtomicReference<ProviderHealth.Status>) statusField
                .get(provider);
        statusRef.set(ProviderHealth.Status.HEALTHY);
    }

    private void setEffectiveMode(LibTorchProvider provider, LibTorchAdvancedModeResolver.EffectiveAdvancedMode mode)
            throws Exception {
        Field field = LibTorchProvider.class.getDeclaredField("effectiveAdvancedMode");
        field.setAccessible(true);
        @SuppressWarnings("unchecked")
        AtomicReference<LibTorchAdvancedModeResolver.EffectiveAdvancedMode> ref = (AtomicReference<LibTorchAdvancedModeResolver.EffectiveAdvancedMode>) field
                .get(provider);
        ref.set(mode);
    }

    private LibTorchExecutionHints invokeResolveExecutionHints(LibTorchProvider provider) throws Exception {
        return invokeResolveExecutionHints(provider, tempDir.resolve("dummy.pt"));
    }

    private LibTorchExecutionHints invokeResolveExecutionHints(LibTorchProvider provider, Path modelPath)
            throws Exception {
        return invokeResolveExecutionHints(provider, modelPath, "tenant-a", "model-a");
    }

    private LibTorchExecutionHints invokeResolveExecutionHints(
            LibTorchProvider provider,
            Path modelPath,
            String tenantId,
            String modelId) throws Exception {
        var method = LibTorchProvider.class.getDeclaredMethod("resolveExecutionHints", Path.class, String.class,
                String.class);
        method.setAccessible(true);
        return (LibTorchExecutionHints) method.invoke(provider, modelPath, tenantId, modelId);
    }

    private LibTorchProviderConfig configWithSageCanary(
            Optional<List<String>> allowedTenants,
            Optional<List<String>> allowedModels) {
        return configWithSageCanary(allowedTenants, allowedModels, Optional.empty(), Optional.empty());
    }

    private LibTorchProviderConfig configWithSageCanary(
            Optional<List<String>> allowedTenants,
            Optional<List<String>> allowedModels,
            Optional<List<String>> blockedTenants,
            Optional<List<String>> blockedModels) {
        LibTorchProviderConfig base = config(true, tempDir, ".pt");
        return new DelegatingLibTorchProviderConfig(base) {
            @Override
            public AdvancedConfig advanced() {
                AdvancedConfig advanced = super.advanced();
                return new AdvancedConfig() {
                    @Override
                    public boolean enabled() {
                        return advanced.enabled();
                    }

                    @Override
                    public String attentionMode() {
                        return advanced.attentionMode();
                    }

                    @Override
                    public boolean fp8RowwiseEnabled() {
                        return advanced.fp8RowwiseEnabled();
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseAllowedTenants() {
                        return advanced.fp8RowwiseAllowedTenants();
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseAllowedModels() {
                        return advanced.fp8RowwiseAllowedModels();
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseBlockedTenants() {
                        return advanced.fp8RowwiseBlockedTenants();
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseBlockedModels() {
                        return advanced.fp8RowwiseBlockedModels();
                    }

                    @Override
                    public boolean sageAttention2Enabled() {
                        return advanced.sageAttention2Enabled();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2AllowedTenants() {
                        return allowedTenants;
                    }

                    @Override
                    public Optional<List<String>> sageAttention2AllowedModels() {
                        return allowedModels;
                    }

                    @Override
                    public Optional<List<String>> sageAttention2BlockedTenants() {
                        return blockedTenants;
                    }

                    @Override
                    public Optional<List<String>> sageAttention2BlockedModels() {
                        return blockedModels;
                    }

                    @Override
                    public String allowedGpuSm() {
                        return advanced.allowedGpuSm();
                    }
                };
            }
        };
    }

    private LibTorchProviderConfig configWithFp8RowwiseCanary(
            Optional<List<String>> allowedTenants,
            Optional<List<String>> allowedModels,
            Optional<List<String>> blockedTenants,
            Optional<List<String>> blockedModels) {
        LibTorchProviderConfig base = config(true, tempDir, ".pt");
        return new DelegatingLibTorchProviderConfig(base) {
            @Override
            public AdvancedConfig advanced() {
                AdvancedConfig advanced = super.advanced();
                return new AdvancedConfig() {
                    @Override
                    public boolean enabled() {
                        return advanced.enabled();
                    }

                    @Override
                    public String attentionMode() {
                        return advanced.attentionMode();
                    }

                    @Override
                    public boolean fp8RowwiseEnabled() {
                        return true;
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseAllowedTenants() {
                        return allowedTenants;
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseAllowedModels() {
                        return allowedModels;
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseBlockedTenants() {
                        return blockedTenants;
                    }

                    @Override
                    public Optional<List<String>> fp8RowwiseBlockedModels() {
                        return blockedModels;
                    }

                    @Override
                    public boolean sageAttention2Enabled() {
                        return advanced.sageAttention2Enabled();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2AllowedTenants() {
                        return advanced.sageAttention2AllowedTenants();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2AllowedModels() {
                        return advanced.sageAttention2AllowedModels();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2BlockedTenants() {
                        return advanced.sageAttention2BlockedTenants();
                    }

                    @Override
                    public Optional<List<String>> sageAttention2BlockedModels() {
                        return advanced.sageAttention2BlockedModels();
                    }

                    @Override
                    public String allowedGpuSm() {
                        return advanced.allowedGpuSm();
                    }
                };
            }
        };
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
            public AdapterConfig adapter() {
                return new AdapterConfig() {
                    @Override
                    public boolean enabled() {
                        return true;
                    }

                    @Override
                    public String basePath() {
                        return basePath.resolve("adapters").toString();
                    }

                    @Override
                    public boolean allowPrecompiledModelPath() {
                        return true;
                    }

                    @Override
                    public int maxActivePoolsPerTenant() {
                        return 0;
                    }

                    @Override
                    public boolean rolloutGuardEnabled() {
                        return false;
                    }

                    @Override
                    public Optional<java.util.List<String>> rolloutAllowedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> rolloutBlockedAdapterIds() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> rolloutBlockedPathPrefixes() {
                        return Optional.empty();
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
                    public Optional<java.util.List<String>> fp8RowwiseAllowedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> fp8RowwiseAllowedModels() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> fp8RowwiseBlockedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> fp8RowwiseBlockedModels() {
                        return Optional.empty();
                    }

                    @Override
                    public boolean sageAttention2Enabled() {
                        return false;
                    }

                    @Override
                    public Optional<java.util.List<String>> sageAttention2AllowedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> sageAttention2AllowedModels() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> sageAttention2BlockedTenants() {
                        return Optional.empty();
                    }

                    @Override
                    public Optional<java.util.List<String>> sageAttention2BlockedModels() {
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

    private abstract static class DelegatingLibTorchProviderConfig implements LibTorchProviderConfig {
        private final LibTorchProviderConfig delegate;

        DelegatingLibTorchProviderConfig(LibTorchProviderConfig delegate) {
            this.delegate = delegate;
        }

        @Override
        public boolean enabled() {
            return delegate.enabled();
        }

        @Override
        public NativeConfig nativeLib() {
            return delegate.nativeLib();
        }

        @Override
        public ModelConfig model() {
            return delegate.model();
        }

        @Override
        public GpuConfig gpu() {
            return delegate.gpu();
        }

        @Override
        public SessionConfig session() {
            return delegate.session();
        }

        @Override
        public InferenceConfig inference() {
            return delegate.inference();
        }

        @Override
        public BatchingConfig batching() {
            return delegate.batching();
        }

        @Override
        public WarmupConfig warmup() {
            return delegate.warmup();
        }

        @Override
        public GenerationConfig generation() {
            return delegate.generation();
        }

        @Override
        public AdapterConfig adapter() {
            return delegate.adapter();
        }

        @Override
        public AdvancedConfig advanced() {
            return delegate.advanced();
        }
    }
}
