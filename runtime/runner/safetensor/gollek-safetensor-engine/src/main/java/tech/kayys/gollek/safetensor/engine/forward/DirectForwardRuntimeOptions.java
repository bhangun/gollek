package tech.kayys.gollek.safetensor.engine.forward;

final class DirectForwardRuntimeOptions {
    private DirectForwardRuntimeOptions() {
    }

    static boolean envTruthy(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return false;
        }
        String normalized = value.trim().toLowerCase();
        return "1".equals(normalized)
                || "true".equals(normalized)
                || "yes".equals(normalized)
                || "on".equals(normalized);
    }

    static int envInt(String name, int defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    static Boolean parseOptionalBoolean(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Boolean.parseBoolean(value);
    }

    static Boolean runtimeOptionalBooleanProperty(String property, Boolean fallback) {
        Boolean runtimeValue = parseOptionalBoolean(System.getProperty(property));
        return runtimeValue != null ? runtimeValue : fallback;
    }
}
