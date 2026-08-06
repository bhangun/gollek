package tech.kayys.gollek.runtime.data;

/**
 * Represents a handle to external memory or special artifacts like a KV Cache within the unified runtime execution context.
 */
public final class GHandle implements GData {
    private final String id;
    private final Object ref;

    public GHandle(String id, Object ref) {
        this.id = id;
        this.ref = ref;
    }

    public Object ref() {
        return ref;
    }

    @Override
    public String id() {
        return id;
    }
}
