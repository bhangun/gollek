package tech.kayys.gollek.safetensor.engine.warmup;

import java.util.List;
import java.util.Map;

/**
 * LoRA adapter configuration parsed from adapter_config.json.
 *
 * @param rank             the LoRA rank (dimension of low-rank decomposition)
 * @param alpha            the alpha scaling factor
 * @param dropout          dropout rate applied during training (typically 0.0 for inference)
 * @param bias             whether bias terms are included ("none", "all", "lora_only")
 * @param targetModules    list of module names/types to apply LoRA to
 * @param taskType         the task type (e.g., "CAUSAL_LM", "SEQ_2_SEQ_LM")
 * @param inferenceMode    whether adapter was exported for inference-only
 * @param baseModelName    name or path of the base model this adapter was trained on
 * @param properties       additional custom properties from config
 */
public record AdapterConfig(
        int rank,
        float alpha,
        float dropout,
        String bias,
        List<String> targetModules,
        String taskType,
        boolean inferenceMode,
        String baseModelName,
        Map<String, Object> properties
) {
    /**
     * Compute the effective scaling factor: alpha / rank.
     * This is the factor applied during LoRA forward pass.
     */
    public float scalingFactor() {
        return alpha / rank;
    }

    /**
     * Check if a module name matches any target module pattern.
     *
     * @param moduleName the module name to check
     * @return true if this module should receive LoRA weights
     */
    public boolean matchesTarget(String moduleName) {
        if (targetModules == null || targetModules.isEmpty()) {
            return true; // Apply to all if no targets specified
        }
        for (String target : targetModules) {
            if (moduleName.contains(target)) {
                return true;
            }
        }
        return false;
    }
}
