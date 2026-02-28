package tech.kayys.gollek.spi.observability;

import tech.kayys.gollek.spi.inference.AdapterSpec;
import tech.kayys.gollek.spi.inference.AdapterSpecResolver;
import tech.kayys.gollek.spi.provider.ProviderRequest;

/**
 * Resolves stable adapter metric tag values from provider requests.
 */
public final class AdapterMetricTagResolver {

    private AdapterMetricTagResolver() {
    }

    public static String resolveAdapterType(ProviderRequest request) {
        AdapterSpec adapterSpec = AdapterSpecResolver.fromProviderRequest(request, 1.0f).orElse(null);
        if (adapterSpec != null) {
            return adapterSpec.type();
        }

        String explicitType = asNonBlankString(request.getParameters().get("adapter_type"));
        if (explicitType == null) {
            explicitType = asNonBlankString(request.getMetadata().get("adapter_type"));
        }
        if (explicitType != null) {
            return explicitType;
        }

        if (hasAdapterHint(request)) {
            return "unspecified";
        }
        return "none";
    }

    public static boolean hasAdapterRequest(ProviderRequest request) {
        return AdapterSpecResolver.fromProviderRequest(request, 1.0f).isPresent() || hasAdapterHint(request);
    }

    private static boolean hasAdapterHint(ProviderRequest request) {
        return asNonBlankString(request.getParameters().get("adapter_id")) != null
                || asNonBlankString(request.getParameters().get("adapter_path")) != null
                || asNonBlankString(request.getParameters().get("lora_adapter_id")) != null
                || asNonBlankString(request.getParameters().get("lora_adapter_path")) != null
                || asNonBlankString(request.getParameters().get("lora_adapter")) != null
                || asNonBlankString(request.getMetadata().get("adapter_id")) != null
                || asNonBlankString(request.getMetadata().get("adapter_path")) != null;
    }

    private static String asNonBlankString(Object value) {
        if (!(value instanceof String stringValue)) {
            return null;
        }
        return stringValue.isBlank() ? null : stringValue;
    }
}
