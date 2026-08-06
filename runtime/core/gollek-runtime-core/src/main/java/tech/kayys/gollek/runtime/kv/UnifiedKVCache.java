package tech.kayys.gollek.runtime.kv;

import tech.kayys.gollek.runtime.data.GHandle;
import java.util.HashMap;
import java.util.Map;

/**
 * High-level KV Cache representation in the unified runtime.
 * It stores GHandles to the underlying native memory blocks rather than 
 * being hardcoded to Tensor or raw memory segments.
 */
public final class UnifiedKVCache {
    private final Map<Integer, GHandle> keys = new HashMap<>();
    private final Map<Integer, GHandle> values = new HashMap<>();

    public void put(int layer, GHandle k, GHandle v) {
        keys.put(layer, k);
        values.put(layer, v);
    }

    public GHandle key(int layer) {
        return keys.get(layer);
    }

    public GHandle value(int layer) {
        return values.get(layer);
    }
}
