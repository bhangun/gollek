package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * A single Transformer decoder layer with causal self-attention and optional cross-attention.
 * <p>
 * The decoder layer is designed for autoregressive sequence generation and machine translation.
 * It includes:
 * <ol>
 *   <li><b>Causal Self-Attention:</b> Attends to past and current tokens only (not future)</li>
 *   <li><b>Cross-Attention:</b> Attends to encoder output (for seq2seq models)</li>
 *   <li><b>Feed-Forward Network:</b> Position-wise MLP</li>
 * </ol>
 * <p>
 * All sub-layers use pre-normalization (LayerNorm before the module) and residual connections.
 * <p>
 * <b>Architecture (Pre-norm, with encoder):</b>
 * <pre>
 * Decoder Input x
 *   ↓
 *   ├→ LayerNorm → CausalMaskedAttention → Dropout ↓
 *   ├──────────────────────────────────────────┤ (Add)
 *   ↓
 *   ├→ LayerNorm → CrossAttention(encoder) → Dropout ↓
 *   ├─────────────────────────────────────────────┤ (Add)
 *   ↓
 *   ├→ LayerNorm → Linear → GELU → Linear → Dropout ↓
 *   ├──────────────────────────────────────────────────┤ (Add)
 *   ↓
 * Output
 * </pre>
 * <p>
 * Equivalent to {@code torch.nn.TransformerDecoderLayer}.
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> [batch, seq_len, dModel]</li>
 *   <li><b>Encoder Output:</b> [batch, enc_seq_len, dModel] (optional)</li>
 *   <li><b>Output:</b> [batch, seq_len, dModel]</li>
 * </ul>
 *
 * <h3>Parameters</h3>
 * <ul>
 *   <li><b>dModel</b> - embedding dimension</li>
 *   <li><b>nHeads</b> - number of attention heads</li>
 *   <li><b>dFF</b> - feed-forward hidden dimension (typically 4*dModel)</li>
 *   <li><b>dropoutP</b> - dropout probability</li>
 * </ul>
 *
 * <h3>Example: Seq2Seq</h3>
 * <pre>{@code
 * // Encoder and decoder layers
 * var decoderLayer = new TransformerDecoderLayer(512, 8, 2048, 0.1f);
 *
 * // Encoder output: [batch=2, enc_len=20, dModel=512]
 * var encoderOutput = GradTensor.randn(2, 20, 512);
 *
 * // Decoder input: [batch=2, dec_len=10, dModel=512]
 * var decoderInput = GradTensor.randn(2, 10, 512);
 *
 * // Forward with cross-attention to encoder
 * var output = decoderLayer.forward(decoderInput, encoderOutput);  // [2, 10, 512]
 * }</pre>
 *
 * <h3>Example: Decoder-Only (Language Model)</h3>
 * <pre>{@code
 * var layer = new TransformerDecoderLayer(768, 12, 3072, 0.1f);
 *
 * // No encoder, just autoregressive self-attention
 * var input = GradTensor.randn(batch, seq_len, 768);
 * var output = layer.forward(input);  // Uses causal masking, no cross-attention
 * }</pre>
 *
 * <h3>Causal Attention Explanation</h3>
 * <p>
 * Causal attention prevents the model from looking at future tokens during decoding.
 * For position i, the model can attend to positions 0..i but not i+1..seq_len.
 * This is essential for autoregressive generation where tokens are predicted one at a time.
 * <p>
 * Without causal masking, a language model would "cheat" by looking ahead during training,
 * leading to degraded performance at test time when future tokens aren't available.
 *
 * <h3>Key Design Features</h3>
 * <ul>
 *   <li><b>Causal Self-Attention:</b> Automatically applies causal mask to prevent future peeking</li>
 *   <li><b>Cross-Attention:</b> Optional, allows attending to encoder output</li>
 *   <li><b>Pre-normalization:</b> LayerNorm before each sub-layer</li>
 *   <li><b>Residual connections:</b> Skip connections between layers</li>
 *   <li><b>Dropout:</b> Applied after each sub-layer</li>
 * </ul>
 *
 * @see TransformerEncoderLayer
 * @see MultiHeadAttention
 * @see LayerNorm
 */
public class TransformerDecoderLayer extends NNModule {

    private final MultiHeadAttention selfAttn;
    private final MultiHeadAttention crossAttn;
    private final Linear ff1;
    private final Linear ff2;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final LayerNorm norm3;
    private final Dropout dropout;
    private final GELU activation;

    /**
     * Create a Transformer decoder layer.
     *
     * @param dModel   embedding dimension (e.g., 512)
     * @param nHeads   number of attention heads (must divide dModel)
     * @param dFF      feed-forward hidden dimension (typically 4*dModel)
     * @param dropoutP dropout probability
     *
     * @throws IllegalArgumentException if dModel is not divisible by nHeads
     */
    public TransformerDecoderLayer(int dModel, int nHeads, int dFF, float dropoutP) {
        if (dModel <= 0 || nHeads <= 0 || dFF <= 0) {
            throw new IllegalArgumentException(
                "dModel, nHeads, and dFF must be positive, got: " + dModel + ", " + nHeads + ", " + dFF);
        }

        this.selfAttn = register("self_attn", new MultiHeadAttention(dModel, nHeads));
        this.crossAttn = register("cross_attn", new MultiHeadAttention(dModel, nHeads));
        this.ff1 = register("ff1", new Linear(dModel, dFF));
        this.ff2 = register("ff2", new Linear(dFF, dModel));
        this.norm1 = register("norm1", new LayerNorm(dModel));
        this.norm2 = register("norm2", new LayerNorm(dModel));
        this.norm3 = register("norm3", new LayerNorm(dModel));
        this.dropout = register("dropout", new Dropout(dropoutP));
        this.activation = register("activation", new GELU());
    }

    /**
     * Forward pass for decoder-only mode (autoregressive language modeling).
     * Uses causal self-attention to prevent attending to future tokens.
     *
     * @param input decoder input of shape [batch, seq_len, dModel]
     * @return output of shape [batch, seq_len, dModel]
     */
    @Override
    public GradTensor forward(GradTensor input) {
        return forward(input, null);
    }

    /**
     * Forward pass with optional encoder output for cross-attention.
     * <p>
     * Supports both decoder-only mode (encOutput=null) and seq2seq mode (encOutput provided).
     *
     * @param input     decoder input of shape [batch, seq_len, dModel]
     * @param encOutput encoder output of shape [batch, enc_seq_len, dModel] for cross-attention
     *                  (or null for decoder-only mode with only causal self-attention)
     * @return output of shape [batch, seq_len, dModel]
     */
    public GradTensor forward(GradTensor input, GradTensor encOutput) {
        // 1. Causal Self-Attention: each position attends to itself and past positions only
        GradTensor x = input.add(dropout.forward(
            selfAttn.forward(norm1.forward(input), true)  // true = apply causal mask
        ));

        // 2. Cross-Attention (optional): if encoder output provided, attend to it
        if (encOutput != null) {
            GradTensor normed = norm2.forward(x);
            // Decoder attends to encoder output (query=decoder, key/value=encoder)
            x = x.add(dropout.forward(
                crossAttn.forward(normed, encOutput, encOutput, false)  // false = no causal mask
            ));
        }

        // 3. Feed-Forward Network with residual connection
        // Use appropriate layer norm depending on whether cross-attention was applied
        LayerNorm ffNorm = encOutput != null ? norm3 : norm2;
        GradTensor ffOut = ff2.forward(activation.forward(ff1.forward(ffNorm.forward(x))));
        return x.add(dropout.forward(ffOut));
    }

    @Override
    public String toString() {
        return "TransformerDecoderLayer(\n  " + selfAttn + "\n  " + crossAttn + "\n)";
    }
}
