package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class LibTorchAdvancedModeResolverTest {

    @AfterEach
    void clearOverrides() {
        System.clearProperty("gollek.libtorch.gpu.sm");
    }

    @Test
    void fallsBackWhenAdvancedDisabled() {
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(false, true, "hybrid_fp8_bf16", false, false, "89,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isFalse();
        assertThat(mode.reason()).isEqualTo("advanced.disabled");
        assertThat(mode.attentionMode()).isEqualTo("baseline");
        assertThat(mode.sageAttention2Requested()).isFalse();
        assertThat(mode.sageAttention2RollbackReason()).isEqualTo("none");
    }

    @Test
    void fallsBackWhenGpuDisabled() {
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(true, false, "hybrid_fp8_bf16", false, false, "89,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isFalse();
        assertThat(mode.reason()).isEqualTo("gpu.disabled");
    }

    @Test
    void fallsBackWhenSmUnknown() {
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(true, true, "hybrid_fp8_bf16", false, false, "89,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isFalse();
        assertThat(mode.reason()).isEqualTo("gpu.sm.unknown");
    }

    @Test
    void enablesAdvancedWhenSmAllowed() {
        System.setProperty("gollek.libtorch.gpu.sm", "90");
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(true, true, "hybrid_fp8_bf16", true, false, "89,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isTrue();
        assertThat(mode.reason()).isEqualTo("advanced.enabled");
        assertThat(mode.attentionMode()).isEqualTo("hybrid_fp8_bf16");
        assertThat(mode.fp8RowwiseEnabled()).isTrue();
        assertThat(mode.sageAttention2Requested()).isFalse();
        assertThat(mode.detectedGpuSm()).contains(90);
    }

    @Test
    void fallsBackWhenSmNotAllowed() {
        System.setProperty("gollek.libtorch.gpu.sm", "86");
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(true, true, "hybrid_fp8_bf16", true, false, "89,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isFalse();
        assertThat(mode.reason()).isEqualTo("gpu.sm.not-allowed");
        assertThat(mode.detectedGpuSm()).contains(86);
    }

    @Test
    void ignoresInvalidAllowedSmTokens() {
        System.setProperty("gollek.libtorch.gpu.sm", "90");
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(true, true, "hybrid_fp8_bf16", false, false, "89,x,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isTrue();
        assertThat(mode.allowedGpuSm()).containsExactly(89, 90);
    }

    @Test
    void rollsBackWhenOnlySageAttention2IsRequested() {
        System.setProperty("gollek.libtorch.gpu.sm", "90");
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(true, true, "baseline", false, true, "89,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isFalse();
        assertThat(mode.reason()).isEqualTo("sageattention2.rollback");
        assertThat(mode.sageAttention2Requested()).isTrue();
        assertThat(mode.sageAttention2Enabled()).isFalse();
        assertThat(mode.sageAttention2RollbackReason()).isEqualTo("sageattention2.not-implemented");
    }

    @Test
    void keepsHybridActiveWhileRollingBackSageAttention2() {
        System.setProperty("gollek.libtorch.gpu.sm", "90");
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchProviderConfig config = config(true, true, "hybrid_fp8_bf16", true, true, "89,90");

        var mode = resolver.resolve(config, null);

        assertThat(mode.advancedEnabled()).isTrue();
        assertThat(mode.reason()).isEqualTo("advanced.enabled");
        assertThat(mode.attentionMode()).isEqualTo("hybrid_fp8_bf16");
        assertThat(mode.fp8RowwiseEnabled()).isTrue();
        assertThat(mode.sageAttention2Requested()).isTrue();
        assertThat(mode.sageAttention2Enabled()).isFalse();
        assertThat(mode.sageAttention2RollbackReason()).isEqualTo("sageattention2.not-implemented");
    }

    private LibTorchProviderConfig config(boolean advancedEnabled,
            boolean gpuEnabled,
            String attentionMode,
            boolean fp8Rowwise,
            boolean sageAttention2,
            String allowedGpuSm) {
        LibTorchProviderConfig config = mock(LibTorchProviderConfig.class);
        LibTorchProviderConfig.GpuConfig gpu = mock(LibTorchProviderConfig.GpuConfig.class);
        LibTorchProviderConfig.AdvancedConfig advanced = mock(LibTorchProviderConfig.AdvancedConfig.class);

        when(config.gpu()).thenReturn(gpu);
        when(gpu.enabled()).thenReturn(gpuEnabled);
        when(gpu.deviceIndex()).thenReturn(0);

        when(config.advanced()).thenReturn(advanced);
        when(advanced.enabled()).thenReturn(advancedEnabled);
        when(advanced.attentionMode()).thenReturn(attentionMode);
        when(advanced.fp8RowwiseEnabled()).thenReturn(fp8Rowwise);
        when(advanced.sageAttention2Enabled()).thenReturn(sageAttention2);
        when(advanced.allowedGpuSm()).thenReturn(allowedGpuSm);

        return config;
    }
}
