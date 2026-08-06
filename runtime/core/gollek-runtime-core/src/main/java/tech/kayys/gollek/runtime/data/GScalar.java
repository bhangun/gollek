package tech.kayys.gollek.runtime.data;

/**
 * Represents a scalar value (e.g., token ID, temperature) within the unified runtime execution context.
 */
public final class GScalar implements GData {
    private final String id;
    private final float value;

    public GScalar(String id, float value) {
        this.id = id;
        this.value = value;
    }

    public float value() {
        return value;
    }

    @Override
    public String id() {
        return id;
    }
}
