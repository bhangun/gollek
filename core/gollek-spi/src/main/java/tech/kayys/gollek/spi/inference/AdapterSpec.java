package tech.kayys.gollek.spi.inference;

import java.util.Objects;

/**
 * Provider-agnostic adapter selection for PEFT-style inference.
 */
public record AdapterSpec(
        String type,
        String adapterId,
        String adapterPath,
        float scale) {

    public AdapterSpec {
        type = normalize(type, "lora");
        adapterId = normalize(adapterId, null);
        adapterPath = normalize(adapterPath, null);
        if (adapterId == null && adapterPath != null) {
            adapterId = adapterPath;
        }
    }

    public boolean isType(String adapterType) {
        return type.equalsIgnoreCase(Objects.requireNonNullElse(adapterType, ""));
    }

    public String cacheKey() {
        return type + "|" + adapterId + "|" + adapterPath + "|" + scale;
    }

    private static String normalize(String value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String normalized = value.trim();
        if (normalized.isEmpty()) {
            return fallback;
        }
        return normalized;
    }
}
