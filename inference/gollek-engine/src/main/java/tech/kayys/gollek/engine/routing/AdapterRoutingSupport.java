package tech.kayys.gollek.engine.routing;

import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.InferenceRequest;

final class AdapterRoutingSupport {

    private AdapterRoutingSupport() {
    }

    static boolean hasAdapterRequest(ProviderRequest request) {
        return hasNonBlankString(request.getParameters().get("adapter_id"))
                || hasNonBlankString(request.getParameters().get("adapter_path"))
                || hasNonBlankString(request.getParameters().get("lora_adapter_id"))
                || hasNonBlankString(request.getParameters().get("lora_adapter_path"))
                || hasNonBlankString(request.getParameters().get("lora_adapter"))
                || hasNonBlankString(request.getMetadata().get("adapter_id"))
                || hasNonBlankString(request.getMetadata().get("adapter_path"))
                || hasNonBlankString(request.getMetadata().get("lora_adapter_id"))
                || hasNonBlankString(request.getMetadata().get("lora_adapter_path"))
                || hasNonBlankString(request.getMetadata().get("lora_adapter"));
    }

    static boolean hasAdapterRequest(InferenceRequest request) {
        return hasNonBlankString(request.getParameters().get("adapter_id"))
                || hasNonBlankString(request.getParameters().get("adapter_path"))
                || hasNonBlankString(request.getParameters().get("lora_adapter_id"))
                || hasNonBlankString(request.getParameters().get("lora_adapter_path"))
                || hasNonBlankString(request.getParameters().get("lora_adapter"))
                || hasNonBlankString(request.getMetadata().get("adapter_id"))
                || hasNonBlankString(request.getMetadata().get("adapter_path"))
                || hasNonBlankString(request.getMetadata().get("lora_adapter_id"))
                || hasNonBlankString(request.getMetadata().get("lora_adapter_path"))
                || hasNonBlankString(request.getMetadata().get("lora_adapter"));
    }

    static boolean isAdapterUnsupported(ProviderCapabilities capabilities) {
        return capabilities != null && capabilities.hasFeature("adapter_unsupported");
    }

    private static boolean hasNonBlankString(Object value) {
        return value instanceof String stringValue && !stringValue.isBlank();
    }
}
