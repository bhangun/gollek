package tech.kayys.gollek.core.params;
import java.util.*;

/**
 * 
 * PARAM NAMING — CURRENT PROBLEM
 * - error-prone
 * - inconsistent across modules
 * - hard to refactor
 * - impossible to safely map (ONNX / GGUF / HF)
 * ============================
 * PARAM NAMING — NEW APPROACH
 * We will use structured, hierarchical paths:
 * layers.0.attn.q_proj.weight
 * layers.0.attn.k_proj.weight
 * layers.0.attn.v_proj.weight
 * layers.0.attn.o_proj.weight
 * layers.0.mlp.gate_proj.weight
 * layers.0.mlp.up_proj.weight
 * layers.0.mlp.down_proj.weight
 * layers.0.input_layernorm.weight
 * layers.0.post_attention_layernorm.weight
 * 
 * ParamPath base = ParamPath.root("layers").index(0);
 * String qWeight = base.child("attn").child("q_proj").child("weight").key();
 * Result:
 * layers.0.attn.q_proj.weight
 * Benefits
 * - deterministic
 * - hierarchical
 * - compatible with HuggingFace / PyTorch
 * - no typo bugs
 * - easy mapping
 * 
 * Example usage:
 * ParamPath p =
 * ParamPath.root("layers").index(0).child("attn").child("q_proj").child("weight");
 * System.out.println(p.key());
 * // layers.0.attn.q_proj.weight
 */

public final class ParamPath {
    private final List<String> parts;

    private ParamPath(List<String> parts) {
        this.parts = parts;
    }

    public static ParamPath root(String name) {
        return new ParamPath(List.of(name));
    }

    public ParamPath child(String name) {
        List<String> newParts = new ArrayList<>(parts);
        newParts.add(name);
        return new ParamPath(newParts);
    }

    public ParamPath index(int i) {
        return child(String.valueOf(i));
    }

    public String key() {
        return String.join(".", parts);
    }

    @Override
    public String toString() {
        return key();
    }
}