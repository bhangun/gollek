package tech.kayys.gollek.runtime.data;

/**
 * Unified data contract for the Gollek Runtime.
 * All runtime artifacts (tensors, scalars, handles like KV cache) must implement this.
 */
public sealed interface GData permits GTensor, GScalar, GHandle {
    /**
     * @return The unique identifier of this data element in the execution context.
     */
    String id();
}
