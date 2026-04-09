package tech.kayys.gollek.ml.nn;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Injects information about the relative or absolute position of tokens in a sequence.
 * <p>
 * Standard Transformers do not have recurrence (like RNNs) or convolution, meaning they
 * are permutation-invariant. To allow the model to use the order of the sequence, we
 * must add "positional encodings" to the input embeddings at the bottom of the encoder
 * and decoder stacks.
 * <p>
 * This class implements fixed sinusoidal positional encodings as described in
 * "Attention Is All You Need":
 * <pre>
 *   PE(pos, 2i)   = sin(pos / 10000^(2i / dModel))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i / dModel))
 * </pre>
 *
 * <h3>Shape</h3>
 * <ul>
 *   <li><b>Input:</b> [batch, seq_len, dModel]</li>
 *   <li><b>Output:</b> [batch, seq_len, dModel] (same as input)</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var posEnc = new PositionalEncoding(768, 512, 0.1f);
 * var embeddings = embed.forward(tokens); // [batch, seq, 768]
 * var x = posEnc.forward(embeddings);     // [batch, seq, 768] with position info
 * }</pre>
 */
public class PositionalEncoding extends NNModule {

    private final GradTensor pe;
    private final Dropout dropout;

    /**
     * Create a positional encoding module.
     *
     * @param dModel   dimension of embeddings
     * @param maxLen   maximum sequence length (buffer size)
     * @param dropoutP dropout probability
     */
    public PositionalEncoding(int dModel, int maxLen, float dropoutP) {
        if (dModel % 2 != 0) {
            throw new IllegalArgumentException("dModel must be even for sinusoidal encoding, got: " + dModel);
        }

        float[] peData = new float[maxLen * dModel];
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < dModel; i += 2) {
                double divTerm = Math.exp(i * -Math.log(10000.0) / dModel);
                peData[pos * dModel + i] = (float) Math.sin(pos * divTerm);
                peData[pos * dModel + i + 1] = (float) Math.cos(pos * divTerm);
            }
        }

        // Positional encoding is a fixed (non-trainable) buffer
        this.pe = GradTensor.of(peData, 1, maxLen, dModel).requiresGrad(false);
        this.dropout = register("dropout", new Dropout(dropoutP));
    }

    /**
     * Create PositionalEncoding with default dropout (0.1).
     *
     * @param dModel   dimension of embeddings
     * @param maxLen   maximum sequence length (buffer size)
     */
    public PositionalEncoding(int dModel, int maxLen) {
        this(dModel, maxLen, 0.1f);
    }

    /**
     * Add positional information to the input embeddings.
     *
     * @param input tensor of shape [batch, seq_len, dModel]
     * @return tensor with added positional encoding and dropout applied
     */
    @Override
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        int batch = (int) shape[0];
        int seqLen = (int) shape[1];
        int dModel = (int) shape[2];

        // Slice the pre-computed PE buffer to the actual sequence length of the input
        float[] peSlice = new float[seqLen * dModel];
        System.arraycopy(pe.data(), 0, peSlice, 0, seqLen * dModel);
        GradTensor currentPe = GradTensor.of(peSlice, 1, seqLen, dModel);

        // input = input + PE (PE is broadcasted across the batch dimension)
        // Note: GradTensor.add() with broadcasting supports [batch, seq, dim] + [1, seq, dim]
        return dropout.forward(input.add(currentPe));
    }

    @Override
    public String toString() {
        return "PositionalEncoding(maxLen=" + pe.shape()[1] + ", dim=" + pe.shape()[2] + ")";
    }
}
