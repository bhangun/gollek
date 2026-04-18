package tech.kayys.gollek.inference.llamacpp;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mockito;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class LoraAdapterManagerTest {

    @TempDir
    Path tempDir;

    @Test
    void enforcesPerTenantAdapterQuota() throws Exception {
        LlamaCppProviderConfig config = Mockito.mock(LlamaCppProviderConfig.class);
        Mockito.when(config.loraEnabled()).thenReturn(true);
        Mockito.when(config.loraDefaultScale()).thenReturn(1.0f);
        Mockito.when(config.loraAdapterBasePath()).thenReturn(tempDir.toString());
        Mockito.when(config.sessionPoolMaxSize()).thenReturn(1);
        Mockito.when(config.loraMaxActiveAdapters()).thenReturn(2);
        Mockito.when(config.loraMaxActiveAdaptersPerTenant()).thenReturn(2);
        Mockito.when(config.metricsEnabled()).thenReturn(false);
        Mockito.when(config.loraRolloutGuardEnabled()).thenReturn(false);

        Path adapterA = tempDir.resolve("a.safetensors");
        Path adapterB = tempDir.resolve("b.safetensors");
        Path adapterC = tempDir.resolve("c.safetensors");
        Files.writeString(adapterA, "x");
        Files.writeString(adapterB, "x");
        Files.writeString(adapterC, "x");

        LoraAdapterManager manager = new LoraAdapterManager(config, new NoopAdapterMetricsRecorder());

        assertThat(manager.resolve(requestFor(adapterA), "tenant-a")).isPresent();
        assertThat(manager.resolve(requestFor(adapterB), "tenant-a")).isPresent();

        assertThatThrownBy(() -> manager.resolve(requestFor(adapterC), "tenant-a"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("LoRA adapter quota exceeded");
    }

    @Test
    void rolloutGuardBlocksNonAllowedTenant() throws Exception {
        LlamaCppProviderConfig config = Mockito.mock(LlamaCppProviderConfig.class);
        Mockito.when(config.loraEnabled()).thenReturn(true);
        Mockito.when(config.loraDefaultScale()).thenReturn(1.0f);
        Mockito.when(config.loraAdapterBasePath()).thenReturn(tempDir.toString());
        Mockito.when(config.sessionPoolMaxSize()).thenReturn(4);
        Mockito.when(config.loraMaxActiveAdapters()).thenReturn(64);
        Mockito.when(config.loraMaxActiveAdaptersPerTenant()).thenReturn(8);
        Mockito.when(config.metricsEnabled()).thenReturn(false);
        Mockito.when(config.loraRolloutGuardEnabled()).thenReturn(true);
        Mockito.when(config.loraRolloutAllowedTenants())
                .thenReturn(java.util.Optional.of(java.util.List.of("tenant-allowed")));
        Mockito.when(config.loraRolloutBlockedAdapterIds()).thenReturn(java.util.Optional.empty());
        Mockito.when(config.loraRolloutBlockedPathPrefixes()).thenReturn(java.util.Optional.empty());

        Path adapter = tempDir.resolve("guard.safetensors");
        Files.writeString(adapter, "x");

        LoraAdapterManager manager = new LoraAdapterManager(config, new NoopAdapterMetricsRecorder());

        assertThatThrownBy(() -> manager.resolve(requestFor(adapter), "tenant-denied"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not allowed for LoRA rollout");
    }

    @Test
    void rolloutGuardBlocksDeniedAdapterId() throws Exception {
        LlamaCppProviderConfig config = Mockito.mock(LlamaCppProviderConfig.class);
        Mockito.when(config.loraEnabled()).thenReturn(true);
        Mockito.when(config.loraDefaultScale()).thenReturn(1.0f);
        Mockito.when(config.loraAdapterBasePath()).thenReturn(tempDir.toString());
        Mockito.when(config.sessionPoolMaxSize()).thenReturn(4);
        Mockito.when(config.loraMaxActiveAdapters()).thenReturn(64);
        Mockito.when(config.loraMaxActiveAdaptersPerTenant()).thenReturn(8);
        Mockito.when(config.metricsEnabled()).thenReturn(false);
        Mockito.when(config.loraRolloutGuardEnabled()).thenReturn(true);
        Mockito.when(config.loraRolloutAllowedTenants()).thenReturn(java.util.Optional.empty());
        Mockito.when(config.loraRolloutBlockedAdapterIds())
                .thenReturn(java.util.Optional.of(java.util.List.of("adapter-deny")));
        Mockito.when(config.loraRolloutBlockedPathPrefixes()).thenReturn(java.util.Optional.empty());

        Path adapter = tempDir.resolve("deny-id.safetensors");
        Files.writeString(adapter, "x");

        LoraAdapterManager manager = new LoraAdapterManager(config, new NoopAdapterMetricsRecorder());

        assertThatThrownBy(() -> manager.resolve(requestFor(adapter, "adapter-deny"), "tenant-a"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("blocked by rollout policy");
    }

    @Test
    void rolloutGuardBlocksDeniedPathPrefix() throws Exception {
        LlamaCppProviderConfig config = Mockito.mock(LlamaCppProviderConfig.class);
        Mockito.when(config.loraEnabled()).thenReturn(true);
        Mockito.when(config.loraDefaultScale()).thenReturn(1.0f);
        Mockito.when(config.loraAdapterBasePath()).thenReturn(tempDir.toString());
        Mockito.when(config.sessionPoolMaxSize()).thenReturn(4);
        Mockito.when(config.loraMaxActiveAdapters()).thenReturn(64);
        Mockito.when(config.loraMaxActiveAdaptersPerTenant()).thenReturn(8);
        Mockito.when(config.metricsEnabled()).thenReturn(false);
        Mockito.when(config.loraRolloutGuardEnabled()).thenReturn(true);
        Mockito.when(config.loraRolloutAllowedTenants()).thenReturn(java.util.Optional.empty());
        Mockito.when(config.loraRolloutBlockedAdapterIds()).thenReturn(java.util.Optional.empty());
        Mockito.when(config.loraRolloutBlockedPathPrefixes())
                .thenReturn(java.util.Optional.of(java.util.List.of(tempDir.toString())));

        Path adapter = tempDir.resolve("deny-path.safetensors");
        Files.writeString(adapter, "x");

        LoraAdapterManager manager = new LoraAdapterManager(config, new NoopAdapterMetricsRecorder());

        assertThatThrownBy(() -> manager.resolve(requestFor(adapter), "tenant-a"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("blocked by rollout policy");
    }

    @Test
    void rolloutGuardAllowsAllowedTenant() throws Exception {
        LlamaCppProviderConfig config = Mockito.mock(LlamaCppProviderConfig.class);
        Mockito.when(config.loraEnabled()).thenReturn(true);
        Mockito.when(config.loraDefaultScale()).thenReturn(1.0f);
        Mockito.when(config.loraAdapterBasePath()).thenReturn(tempDir.toString());
        Mockito.when(config.sessionPoolMaxSize()).thenReturn(4);
        Mockito.when(config.loraMaxActiveAdapters()).thenReturn(64);
        Mockito.when(config.loraMaxActiveAdaptersPerTenant()).thenReturn(8);
        Mockito.when(config.metricsEnabled()).thenReturn(false);
        Mockito.when(config.loraRolloutGuardEnabled()).thenReturn(true);
        Mockito.when(config.loraRolloutAllowedTenants())
                .thenReturn(java.util.Optional.of(java.util.List.of("tenant-allowed")));
        Mockito.when(config.loraRolloutBlockedAdapterIds()).thenReturn(java.util.Optional.empty());
        Mockito.when(config.loraRolloutBlockedPathPrefixes()).thenReturn(java.util.Optional.empty());

        Path adapter = tempDir.resolve("allowed.safetensors");
        Files.writeString(adapter, "x");

        LoraAdapterManager manager = new LoraAdapterManager(config, new NoopAdapterMetricsRecorder());

        assertThat(manager.resolve(requestFor(adapter), "tenant-allowed")).isPresent();
    }

    private ProviderRequest requestFor(Path adapterPath) {
        return requestFor(adapterPath, null);
    }

    private ProviderRequest requestFor(Path adapterPath, String adapterId) {
        ProviderRequest.Builder builder = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .parameter("adapter_type", "lora")
                .parameter("adapter_path", adapterPath.getFileName().toString());
        if (adapterId != null) {
            builder.parameter("adapter_id", adapterId);
        }
        return builder.build();
    }

}
