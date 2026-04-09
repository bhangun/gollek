package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * A single Transformer encoder layer.
 * <p>
 * The encoder layer is a fundamental building block of the Transformer architecture.
 * It applies self-attention to allow each token to attend to all other tokens,
 * followed by a position-wise feed-forward network. Both sub-layers use pre-normalization
 * (LayerNorm before the module) and residual connections for stable training.
 * <p>
 * <b>Architecture (Pre-norm):</b>
 * <pre>
 * Input x
 *   ↓
 *   ├→ LayerNorm → MultiHeadAttention → Dropout ↓
 *   ├──────────────────────────────────────────┤ (Add)
 *   ↓
 *   ├→ LayerNorm → Linear(expand) → GELU → Linear(contract) → Dropout ↓
 *   ├────────────────────────────────────────────────────────────────┤ (Add)
 *   ↓
 * Output
 * </pre>
 * <p>
 * This is the pre-norm architecture (used in modern models like GPT-3, BERT variants).
 * Equivalent to {@code torch.nn.TransformerEncoderLayer}.
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> [batch, seq_len, dModel]</li>
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
 * <h3>Example</h3>
 * <pre>{@code
 * // Create a single encoder layer matching BERT dimensions
 * var layer = new TransformerEncoderLayer(768, 12, 3072, 0.1f);
 *
 * // Input shape: [batch_size=2, seq_len=10, dModel=768]
 * var input = GradTensor.randn(2, 10, 768);
 * var output = layer.forward(input);  // [2, 10, 768]
 * }</pre>
 *
 * <h3>Typical Usage in Transformer</h3>
 * <pre>{@code
 * // Build a 12-layer encoder (like BERT)
 * List<NNModule> layers = new ArrayList<>();
 * for (int i = 0; i < 12; i++) {
 *     layers.add(new TransformerEncoderLayer(768, 12, 3072, 0.1f));
 * }
 * var encoder = new Sequential(layers.toArray(new Module[0]));
 *
 * var embeddings = embedding.forward(tokenIds);
 * var encoded = encoder.forward(embeddings);
 * }</pre>
 *
 * <h3>Key Design Features</h3>
 * <ul>
 *   <li><b>Pre-normalization:</b> LayerNorm applied before each sub-layer (better gradient flow)</li>
 *   <li><b>Residual connections:</b> Skip connections allow gradients to bypass sub-layers</li>
 *   <li><b>Dropout:</b> Applied after attention and feed-forward for regularization</li>
 *   <li><b>GELU activation:</b> Smooth activation function effective in transformers</li>
 * </ul>
 *
 * @see TransformerDecoderLayer
 * @see MultiHeadAttention
 * @see LayerNorm
 */
public class TransformerEncoderLayer extends NNModule {

    private final MultiHeadAttention selfAttn;
    private final Linear ff1;
    private final Linear ff2;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final Dropout dropout;
    private final GELU activation;

    /**
     * Create a Transformer encoder layer.
     *
     * @param dModel   embedding dimension (e.g., 768 for BERT)
     * @param nHeads   number of attention heads (must divide dModel)
     * @param dFF      feed-forward hidden dimension (typically 4*dModel, e.g., 3072)
     * @param dropoutP dropout probability
     *
     * @throws IllegalArgumentException if dModel is not divisible by nHeads
     */
    public TransformerEncoderLayer(int dModel, int nHeads, int dFF, float dropoutP) {
        if (dModel <= 0 || nHeads <= 0 || dFF <= 0) {
            throw new IllegalArgumentException(
                "dModel, nHeads, and dFF must be positive, got: " + dModel + ", " + nHeads + ", " + dFF);
        }

        this.selfAttn = register("self_attn", new MultiHeadAttention(dModel, nHeads));
        this.ff1 = register("ff1", new Linear(dModel, dFF));
        this.ff2 = register("ff2", new Linear(dFF, dModel));
        this.norm1 = register("norm1", new LayerNorm(dModel));
        this.norm2 = register("norm2", new LayerNorm(dModel));
        this.dropout = register("dropout", new Dropout(dropoutP));
        this.activation = register("activation", new GELU());
    }

    /**
     * Apply the encoder layer transformation.
     *
     * @param input tensor of shape [batch, seq_len, dModel]
     * @return output tensor of shape [batch, seq_len, dModel]
     */
    @Override
    public GradTensor forward(GradTensor input) {
        // Pre-norm: LayerNorm → MultiHeadAttention → Dropout → Add(residual)
        GradTensor attnOut = selfAttn.forward(norm1.forward(input));
        GradTensor x = input.add(dropout.forward(attnOut));

        // Pre-norm: LayerNorm → FFN(Linear+GELU+Linear) → Dropout → Add(residual)
        GradTensor ffOut = ff2.forward(activation.forward(ff1.forward(norm2.forward(x))));
        return x.add(dropout.forward(ffOut));
    }

    @Override
    public String toString() {
        return "TransformerEncoderLayer(\n  " + selfAttn + "\n  " + ff1 + "\n  " + ff2 + "\n)";
    }
}
