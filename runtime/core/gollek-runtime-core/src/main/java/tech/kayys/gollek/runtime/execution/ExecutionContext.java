package tech.kayys.gollek.runtime.execution;

import tech.kayys.gollek.runtime.data.GData;
// import tech.kayys.gollek.backend.BackendRegistry; // Assuming this will exist
// import tech.kayys.gollek.backend.ExecutionStrategy; // Assuming this will exist
import java.util.HashMap;
import java.util.Map;

/**
 * Unified execution context.
 * Replaces the old Map<String, Tensor> with a unified registry of GData (Tensors, Scalars, Handles).
 */
public final class ExecutionContext {
    private final Map<String, GData> data = new HashMap<>();
    
    // Commented out until we bring in the backend modules
    // private final BackendRegistry backends;
    // private final ExecutionStrategy strategy;

    public ExecutionContext(/*BackendRegistry backends, ExecutionStrategy strategy*/) {
        // this.backends = backends;
        // this.strategy = strategy;
    }

    /**
     * Stores a unified data artifact (tensor, scalar, handle) into the context.
     */
    public void put(GData d) {
        data.put(d.id(), d);
    }

    /**
     * Retrieves a unified data artifact from the context.
     */
    @SuppressWarnings("unchecked")
    public <T extends GData> T get(String id) {
        return (T) data.get(id);
    }

    // public Backend selectBackend(Op op) {
    //     return new AdaptiveScheduler()
    //         .select(op, backends.ordered()
    //             .stream()
    //             .map(BackendDescriptor::backend)
    //             .toList());
    // }
}
