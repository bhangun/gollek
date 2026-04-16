package tech.kayys.gollek.safetensor.engine.warmup;

import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;

import java.nio.file.Path;
import java.util.*;

/**
 * A loaded LoRA adapter with weights and configuration.
 *
 * @param adapterPath     the directory or file path this adapter was loaded from
 * @param weights         map of tensor name → LibTorch TorchTensor (lora_A, lora_B matrices)
 * @param config          adapter configuration (rank, alpha, target modules, etc.)
 * @param loadedAt        timestamp when this adapter was loaded (epoch millis)
 */
public record LoadedAdapter(
        Path adapterPath,
        Map<String, TorchTensor> weights,
        AdapterConfig config,
        long loadedAt
) implements AutoCloseable {

    private static final Logger log = Logger.getLogger(LoadedAdapter.class);

    /**
     * Get the LoRA A matrix for a specific base module.
     *
     * @param baseModuleName the base module name (e.g., "model.layers.0.self_attn.q_proj")
     * @return the lora_A tensor, or empty if not present
     */
    public Optional<TorchTensor> getLoraA(String baseModuleName) {
        TorchTensor tensor = weights.get(baseModuleName + ".lora_A.weight");
        if (tensor == null) {
            tensor = weights.get(baseModuleName + ".lora_A");
        }
        return Optional.ofNullable(tensor);
    }

    /**
     * Get the LoRA B matrix for a specific base module.
     *
     * @param baseModuleName the base module name (e.g., "model.layers.0.self_attn.q_proj")
     * @return the lora_B tensor, or empty if not present
     */
    public Optional<TorchTensor> getLoraB(String baseModuleName) {
        TorchTensor tensor = weights.get(baseModuleName + ".lora_B.weight");
        if (tensor == null) {
            tensor = weights.get(baseModuleName + ".lora_B");
        }
        return Optional.ofNullable(tensor);
    }

    /**
     * Get both LoRA matrices (A, B) for a specific base module.
     *
     * @param baseModuleName the base module name
     * @return an optional containing both tensors if present, empty otherwise
     */
    public Optional<LoraPair> getLoraPair(String baseModuleName) {
        Optional<TorchTensor> loraA = getLoraA(baseModuleName);
        Optional<TorchTensor> loraB = getLoraB(baseModuleName);
        if (loraA.isPresent() && loraB.isPresent()) {
            return Optional.of(new LoraPair(loraA.get(), loraB.get()));
        }
        return Optional.empty();
    }

    /**
     * Check if this adapter contains weights for a specific module.
     *
     * @param baseModuleName the base module name to check
     * @return true if both lora_A and lora_B are present
     */
    public boolean hasModule(String baseModuleName) {
        return getLoraPair(baseModuleName).isPresent();
    }

    /**
     * Get all base module names that have LoRA weights in this adapter.
     *
     * @return set of module names
     */
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

    /**
     * Get the adapter rank (dimension of the low-rank decomposition).
     *
     * @return the rank value from config
     */
    public int rank() {
        return config.rank();
    }

    /**
     * Get the alpha scaling factor.
     *
     * @return the alpha value
     */
    public float alpha() {
        return config.alpha();
    }

    /**
     * Get the effective scaling factor (alpha / rank).
     *
     * @return the scaling factor to apply during LoRA computation
     */
    public float scalingFactor() {
        return config.alpha() / config.rank();
    }

    @Override
    public void close() {
        for (TorchTensor tensor : weights.values()) {
            try {
                tensor.close();
            } catch (Exception e) {
                log.warnf(e, "LoadedAdapter: failed to close tensor");
            }
        }
    }
}
