package tech.kayys.gollek.safetensor.engine.generation;

import io.quarkus.arc.Arc;

final class DirectInferenceDependencies {
    private DirectInferenceDependencies() {
    }

    static <T> T resolve(T current, Class<T> type, String name) {
        if (current != null) {
            return current;
        }
        try {
            if (Arc.container() != null) {
                var instance = Arc.container().instance(type);
                if (instance.isAvailable()) {
                    return instance.get();
                }
            }
        } catch (Exception ignored) {
            // Fall through and raise a clearer error below.
        }
        throw new IllegalStateException(name + " is not available");
    }
}
