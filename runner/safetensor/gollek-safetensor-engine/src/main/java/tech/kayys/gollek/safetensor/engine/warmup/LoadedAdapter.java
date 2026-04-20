package tech.kayys.gollek.safetensor.engine.warmup;

import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import java.nio.file.Path;
import java.util.*;

/**
 * A loaded LoRA adapter with weights and configuration.
 *
 * @param adapterPath     the directory or file path this adapter was loaded from
 * @param weights         map of tensor name → AccelTensor (lora_A, lora_B matrices)
 * @param config          adapter configuration (rank, alpha, target modules, etc.)
 * @param loadedAt        timestamp when this adapter was loaded (epoch millis)
 */
public record LoadedAdapter(
        Path adapterPath,
        Map<String, AccelTensor> weights,
        AdapterConfig config,
        long loadedAt
) implements AutoCloseable {

    private static final Logger log = Logger.getLogger(LoadedAdapter.class);

    public Optional<AccelTensor> getLoraA(String baseModuleName) {
        AccelTensor tensor = weights.get(baseModuleName + ".lora_A.weight");
        if (tensor == null) {
            tensor = weights.get(baseModuleName + ".lora_A");
        }
        return Optional.ofNullable(tensor);
    }

    public Optional<AccelTensor> getLoraB(String baseModuleName) {
        AccelTensor tensor = weights.get(baseModuleName + ".lora_B.weight");
        if (tensor == null) {
            tensor = weights.get(baseModuleName + ".lora_B");
        }
        return Optional.ofNullable(tensor);
    }

    public Optional<LoraPair> getLoraPair(String baseModuleName) {
        Optional<AccelTensor> loraA = getLoraA(baseModuleName);
        Optional<AccelTensor> loraB = getLoraB(baseModuleName);
        if (loraA.isPresent() && loraB.isPresent()) {
            return Optional.of(new LoraPair(loraA.get(), loraB.get()));
        }
        return Optional.empty();
    }

    public boolean hasModule(String baseModuleName) {
        return getLoraPair(baseModuleName).isPresent();
    }

    public Set<String> getModuleNames() {
        Set<String> modules = new HashSet<>();
        for (String key : weights.keySet()) {
            if (key.endsWith(".lora_A.weight") || key.endsWith(".lora_A")) {
                String baseName = key
                        .replace(".lora_A.weight", "")
                        .replace(".lora_A", "");
                modules.add(baseName);
            }
        }
        return modules;
    }

    public int rank() {
        return config.rank();
    }

    public float alpha() {
        return config.alpha();
    }

    public float scalingFactor() {
        return config.alpha() / config.rank();
    }

    @Override
    public void close() {
        for (AccelTensor tensor : weights.values()) {
            try {
                tensor.close();
            } catch (Exception e) {
                log.warnf(e, "LoadedAdapter: failed to close tensor");
            }
        }
    }
}
