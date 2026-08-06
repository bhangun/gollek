package tech.kayys.gollek.runtime.weight;

import tech.kayys.gollek.core.weight.WeightStore;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;

import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

/**
 * Loads weights lazily from multiple sources (Safetensors, directories, etc.).
 */
public final class LazyWeightStore implements WeightStore, AutoCloseable {
    private final List<WeightSource> sources = new ArrayList<>();
    private final Map<String, Tensor> cache = new ConcurrentHashMap<>();

    public LazyWeightStore() {
    }

    public void addSource(WeightSource source) {
        sources.add(source);
    }

    public void discover(Path directory, ComputeBackend backend) throws IOException {
        try (Stream<Path> stream = Files.list(directory)) {
            List<Path> paths = stream.toList();
            for (Path path : paths) {
                try {
                    addSource(WeightSourceFactory.create(path, backend));
                } catch (Exception ignored) {
                    // Skip unsupported files
                }
            }
        }
        // Also add the directory itself as a DirectoryWeightSource if it has .bin files
        if (Files.isDirectory(directory)) {
            addSource(new DirectoryWeightSource(directory, backend));
        }
    }

    @Override
    public Tensor get(String key) {
        return cache.computeIfAbsent(key, k -> {
            for (WeightSource source : sources) {
                Tensor t = source.get(k);
                if (t != null) return t;
            }
            throw new MissingWeightException("Weight not found in any source: " + k);
        });
    }

    @Override
    public boolean contains(String key) {
        if (cache.containsKey(key)) return true;
        for (WeightSource source : sources) {
            if (source.contains(key)) return true;
        }
        return false;
    }

    public void evict(String key) {
        Tensor t = cache.remove(key);
        if (t != null) {
            t.release();
        }
    }

    @Override
    public void close() throws Exception {
        cache.values().forEach(Tensor::release);
        cache.clear();
        for (WeightSource source : sources) {
            source.close();
        }
    }
}
