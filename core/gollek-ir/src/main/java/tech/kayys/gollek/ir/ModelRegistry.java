package tech.kayys.gollek.ir;

import tech.kayys.alkhawarizm.core.tensor.ModelWeightLoader;
import tech.kayys.alkhawarizm.core.tensor.WeightAdapter;
import java.nio.file.Path;
import java.util.*;

public final class ModelRegistry {
    private final List<ModelWeightLoader> loaders = new ArrayList<>();

    public void register(ModelWeightLoader loader) {
        loaders.add(loader);
    }

    public WeightAdapter load(Path path) {
        for (ModelWeightLoader l : loaders) {
            if (l.supports(path)) {
                return l.load(path);
            }
        }
        throw new RuntimeException("Unsupported format: " + path);
    }
}