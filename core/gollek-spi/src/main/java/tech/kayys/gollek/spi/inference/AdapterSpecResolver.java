package tech.kayys.gollek.spi.inference;

import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.util.Map;
import java.util.Optional;

/**
 * Resolves adapter directives from request parameters in a provider-neutral way.
 */
public final class AdapterSpecResolver {

    private AdapterSpecResolver() {
    }

    public static Optional<AdapterSpec> fromProviderRequest(ProviderRequest request, float defaultScale) {
        return fromMap(request.getParameters(), defaultScale);
    }

    public static Optional<AdapterSpec> fromInferenceRequest(InferenceRequest request, float defaultScale) {
        return fromMap(request.getParameters(), defaultScale);
    }

    private static Optional<AdapterSpec> fromMap(Map<String, Object> params, float defaultScale) {
        String type = readString(params, "adapter_type").orElse("lora");
        String id = readString(params, "adapter_id")
                .or(() -> readString(params, "lora_adapter_id"))
                .orElse(null);
        String path = readString(params, "adapter_path")
                .or(() -> readString(params, "lora_adapter_path"))
                .or(() -> readString(params, "lora_adapter"))
                .orElse(null);
        float scale = readScale(params.get("adapter_scale"), defaultScale);
        if (!params.containsKey("adapter_scale") && params.containsKey("lora_scale")) {
            scale = readScale(params.get("lora_scale"), defaultScale);
        }

        if (id == null && path == null) {
            return Optional.empty();
        }
        return Optional.of(new AdapterSpec(type, id, path, scale));
    }

    private static Optional<String> readString(Map<String, Object> map, String key) {
        if (!map.containsKey(key)) {
            return Optional.empty();
        }
        Object value = map.get(key);
        if (value == null) {
            return Optional.empty();
        }
        String normalized = value.toString().trim();
        return normalized.isEmpty() ? Optional.empty() : Optional.of(normalized);
    }

    private static float readScale(Object raw, float fallback) {
        if (raw instanceof Number n) {
            return n.floatValue();
        }
        if (raw instanceof String s) {
            try {
                return Float.parseFloat(s.trim());
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }
}
