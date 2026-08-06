package tech.kayys.gollek.runtime.weight;

import tech.kayys.gollek.core.weight.WeightStore;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A simple in-memory weight store using a ConcurrentHashMap.
 */
public final class InMemoryWeightStore implements WeightStore {
    private final Map<String, Tensor> map = new ConcurrentHashMap<>();

    public InMemoryWeightStore() {
    }

    public InMemoryWeightStore(Map<String, Tensor> initialWeights) {
        map.putAll(initialWeights);
    }

    public void put(String key, Tensor tensor) {
        Objects.requireNonNull(key, "Weight key cannot be null");
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        map.put(key, tensor);
    }

    public void putAll(Map<String, Tensor> weights) {
        map.putAll(weights);
    }

    @Override
    public Tensor get(String key) {
        Tensor t = map.get(key);
        if (t == null) {
            throw new MissingWeightException(
                    String.format("Weight '%s' not found in memory store. Available keys: %s",
                            key, getSubsetOfKeys(10)));
        }
        return t;
    }

    @Override
    public boolean contains(String key) {
        return map.containsKey(key);
    }

    public int size() {
        return map.size();
    }

    public void clear() {
        for (Tensor tensor : map.values()) {
            tensor.release();
        }
        map.clear();
    }

    private List<String> getSubsetOfKeys(int limit) {
        return map.keySet().stream().limit(limit).toList();
    }
}
