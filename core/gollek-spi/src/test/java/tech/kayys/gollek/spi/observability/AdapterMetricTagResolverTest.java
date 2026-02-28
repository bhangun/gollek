package tech.kayys.gollek.spi.observability;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import static org.assertj.core.api.Assertions.assertThat;

class AdapterMetricTagResolverTest {

    @Test
    void resolvesTypeFromAdapterSpecKeys() {
        ProviderRequest request = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .parameter("adapter_type", "lora")
                .parameter("adapter_id", "tenant-a")
                .build();

        assertThat(AdapterMetricTagResolver.resolveAdapterType(request)).isEqualTo("lora");
        assertThat(AdapterMetricTagResolver.hasAdapterRequest(request)).isTrue();
    }

    @Test
    void resolvesExplicitTypeWithoutAdapterIdOrPath() {
        ProviderRequest request = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .parameter("adapter_type", "ia3")
                .build();

        assertThat(AdapterMetricTagResolver.resolveAdapterType(request)).isEqualTo("ia3");
    }

    @Test
    void resolvesLoraWhenAdapterPathExists() {
        ProviderRequest request = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .parameter("adapter_path", "/tmp/custom.bin")
                .build();

        assertThat(AdapterMetricTagResolver.resolveAdapterType(request)).isEqualTo("lora");
    }

    @Test
    void resolvesUnspecifiedWhenOnlyMetadataAdapterHintExists() {
        ProviderRequest request = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .metadata("adapter_id", "tenant-a")
                .build();

        assertThat(AdapterMetricTagResolver.resolveAdapterType(request)).isEqualTo("unspecified");
        assertThat(AdapterMetricTagResolver.hasAdapterRequest(request)).isTrue();
    }

    @Test
    void resolvesNoneWhenNoAdapterSignalExists() {
        ProviderRequest request = ProviderRequest.builder()
                .model("qwen")
                .message(Message.user("hello"))
                .build();

        assertThat(AdapterMetricTagResolver.resolveAdapterType(request)).isEqualTo("none");
        assertThat(AdapterMetricTagResolver.hasAdapterRequest(request)).isFalse();
    }
}
