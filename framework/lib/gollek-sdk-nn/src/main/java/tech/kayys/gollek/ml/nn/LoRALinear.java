package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * LoRA (Low-Rank Adaptation) — parameter-efficient fine-tuning that injects
 * trainable low-rank matrices into frozen pre-trained weights.
 *
 * <p>Based on <em>"LoRA: Low-Rank Adaptation of Large Language Models"</em>
 * (Hu et al., 2021).
 *
 * <p>Instead of updating W ∈ ℝ^{d×k} directly, LoRA learns:
 * <pre>
 *   W' = W + ΔW = W + B·A
 *   where A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, rank r ≪ min(d,k)
 * </pre>
 *
 * <p>Only A and B are trained; W is frozen. This reduces trainable parameters
 * from d×k to r×(d+k), typically 100-1000× fewer.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Wrap a frozen Linear layer with LoRA
 * var lora = new LoRALinear(frozenLinear, rank=8, alpha=16f);
 * GradTensor out = lora.forward(x);
 *
 * // Only lora.loraParameters() are trained
 * var optimizer = new AdamW(lora.loraParameters(), lr=1e-4f);
 * }</pre>
 */
public final class LoRALinear extends NNModule {

    private final NNModule base;       // frozen pre-trained linear
    private final Parameter loraA;   // [rank, inFeatures]  — initialized N(0, 1/√rank)
    private final Parameter loraB;   // [outFeatures, rank] — initialized zeros
    private final float scale;       // alpha / rank

    /**
     * Wraps a frozen linear layer with LoRA adapters.
     *
     * @param base        frozen pre-trained {@link Linear} (or any module)
     * @param inFeatures  input dimension
     * @param outFeatures output dimension
     * @param rank        LoRA rank r (typically 4, 8, or 16)
     * @param alpha       LoRA scaling factor (typically 2× rank)
     */
    public LoRALinear(NNModule base, int inFeatures, int outFeatures, int rank, float alpha) {
        this.base  = register("base", base);
        this.scale = alpha / rank;

        // A: random init, B: zero init (so ΔW=0 at start)
        float std = (float) (1.0 / Math.sqrt(rank));
        this.loraA = registerParameter("lora_A",
            GradTensor.randn(rank, inFeatures).mul(std));
        this.loraB = registerParameter("lora_B",
            GradTensor.zeros(outFeatures, rank));

        // Freeze base parameters
        for (Parameter p : base.parameters()) p.data().requiresGrad(false);
    }

    /**
     * Forward pass: base output + LoRA delta.
     *
     * <p>{@code y = base(x) + scale · (x @ A^T) @ B^T}
     *
     * @param x input tensor {@code [..., inFeatures]}
     * @return output tensor {@code [..., outFeatures]}
     */
    @Override
    public GradTensor forward(GradTensor x) {
        GradTensor baseOut = base.forward(x);
        // LoRA path: x @ A^T → [*, rank], then @ B^T → [*, outFeatures]
        GradTensor loraOut = x.matmul(loraA.data().transpose())
                              .matmul(loraB.data().transpose())
                              .mul(scale);
        return baseOut.add(loraOut);
    }

    /**
     * Returns only the LoRA parameters (A and B matrices).
     * Pass these to the optimizer — base parameters stay frozen.
     *
     * @return list containing only lora_A and lora_B parameters
     */
    public java.util.List<Parameter> loraParameters() {
        return java.util.List.of(loraA, loraB);
    }

    /**
     * Merges LoRA weights into the base weight matrix for inference.
     * After merging, the LoRA adapters can be discarded.
     *
     * @return merged weight tensor {@code W + scale·B·A}
     */
    public GradTensor mergedWeight() {
        // B @ A → [outFeatures, inFeatures]
        GradTensor delta = loraB.data().matmul(loraA.data()).mul(scale);
        // Get base weight (assumes base is a Linear)
        GradTensor baseW = base.namedParameters().values().iterator().next().data();
        return baseW.add(delta);
    }

    @Override public String toString() {
        return String.format("LoRALinear(rank=%d, scale=%.2f, base=%s)",
            (int) loraA.data().shape()[0], scale, base);
    }
}
