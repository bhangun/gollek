package tech.kayys.gollek.engine.routing;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AdapterRoutingSupportTest {

    @Test
    void detectsAdapterHintsFromProviderRequestMetadata() {
        ProviderRequest request = ProviderRequest.builder()
                .model("m")
                .message(Message.user("hi"))
                .metadata(Map.of("lora_adapter_id", "adapter-x"))
                .build();

        assertTrue(AdapterRoutingSupport.hasAdapterRequest(request));
    }

    @Test
    void detectsAdapterHintsFromInferenceRequestParameters() {
        InferenceRequest request = InferenceRequest.builder()
                .model("m")
                .message(Message.user("hi"))
                .parameter("adapter_id", "adapter-y")
                .build();

        assertTrue(AdapterRoutingSupport.hasAdapterRequest(request));
    }

    @Test
    void detectsAdapterUnsupportedCapabilityFlag() {
        ProviderCapabilities unsupported = ProviderCapabilities.builder()
                .features(Set.of("adapter_unsupported"))
                .build();
        ProviderCapabilities supported = ProviderCapabilities.builder()
                .features(Set.of("adapter_supported"))
                .build();

        assertTrue(AdapterRoutingSupport.isAdapterUnsupported(unsupported));
        assertFalse(AdapterRoutingSupport.isAdapterUnsupported(supported));
    }
}
