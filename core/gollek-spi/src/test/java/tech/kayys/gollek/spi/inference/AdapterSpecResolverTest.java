package tech.kayys.gollek.spi.inference;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import static org.assertj.core.api.Assertions.assertThat;

class AdapterSpecResolverTest {

    @Test
    void resolvesGenericAdapterKeys() {
        ProviderRequest request = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .parameter("adapter_type", "lora")
                .parameter("adapter_id", "tenant-a")
                .parameter("adapter_path", "/tmp/a.safetensors")
                .parameter("adapter_scale", 0.8f)
                .build();

        var spec = AdapterSpecResolver.fromProviderRequest(request, 1.0f).orElseThrow();
        assertThat(spec.type()).isEqualTo("lora");
        assertThat(spec.adapterId()).isEqualTo("tenant-a");
        assertThat(spec.adapterPath()).isEqualTo("/tmp/a.safetensors");
        assertThat(spec.scale()).isEqualTo(0.8f);
    }

    @Test
    void resolvesLegacyLoraKeys() {
        ProviderRequest request = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .parameter("lora_adapter_id", "legacy")
                .parameter("lora_adapter_path", "/tmp/lora.bin")
                .parameter("lora_scale", "1.3")
                .build();

        var spec = AdapterSpecResolver.fromProviderRequest(request, 1.0f).orElseThrow();
        assertThat(spec.type()).isEqualTo("lora");
        assertThat(spec.adapterId()).isEqualTo("legacy");
        assertThat(spec.adapterPath()).isEqualTo("/tmp/lora.bin");
        assertThat(spec.scale()).isEqualTo(1.3f);
    }
}
