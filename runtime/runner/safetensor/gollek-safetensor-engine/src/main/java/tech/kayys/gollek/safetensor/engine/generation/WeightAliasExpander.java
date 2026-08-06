package tech.kayys.gollek.safetensor.engine.generation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

/**
 * Adds canonical tensor-name aliases for common text-decoder wrapper prefixes.
 */
final class WeightAliasExpander {
    private static final List<Alias> COMMON_TEXT_MODEL_ALIASES = List.of(
            new Alias("text_model.lm_head.", "lm_head."),
            new Alias("model.text_model.lm_head.", "lm_head."),
            new Alias("language_model.lm_head.", "lm_head."),
            new Alias("model.language_model.lm_head.", "lm_head."),
            new Alias("text_model.model.lm_head.", "lm_head."),
            new Alias("model.text_model.model.lm_head.", "lm_head."),
            new Alias("language_model.model.lm_head.", "lm_head."),
            new Alias("text_model.", "model."),
            new Alias("model.text_model.", "model."),
            new Alias("language_model.model.", "model."),
            new Alias("mlp.switch_mlp.", "mlp.experts."),

            new Alias("model.language_model.", "model."),
            new Alias("text_model.model.", "model."),
            new Alias("model.text_model.model.", "model."));

    private WeightAliasExpander() {
    }

    static void applyCommonAliases(Map<String, AccelTensor> weights) {
        applyAliases(weights, COMMON_TEXT_MODEL_ALIASES);
        
        Map<String, AccelTensor> qweightUpdates = new HashMap<>();
        for (Map.Entry<String, AccelTensor> entry : weights.entrySet()) {
            if (entry.getKey().endsWith(".qweight")) {
                qweightUpdates.putIfAbsent(entry.getKey().replace(".qweight", ".weight"), entry.getValue());
            }
        }
        weights.putAll(qweightUpdates);
    }

    static void applyAliases(Map<String, AccelTensor> weights, List<Alias> aliases) {
        if (weights == null || weights.isEmpty() || aliases == null || aliases.isEmpty()) {
            return;
        }
        Map<String, AccelTensor> updates = new HashMap<>();
        List<Map.Entry<String, AccelTensor>> entries = new ArrayList<>(weights.entrySet());

        for (Alias alias : aliases) {
            for (Map.Entry<String, AccelTensor> entry : entries) {
                String key = entry.getKey();
                if (key.startsWith(alias.from())) {
                    String rewritten = alias.to() + key.substring(alias.from().length());
                    if (!weights.containsKey(rewritten)) {
                        updates.putIfAbsent(rewritten, entry.getValue());
                    }
                }
            }
        }
        weights.putAll(updates);
    }

    record Alias(String from, String to) {
    }
}
