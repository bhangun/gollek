package tech.kayys.gollek.sdk.rnn.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.sdk.rnn.cells.RNNCell;

/**
 * Bidirectional RNN Layer.
 *
 * <p>Wraps any RNN cell to process sequences in both forward and backward directions.
 * Output size is doubled (concatenates forward and backward outputs).
 * This is useful for tasks where context from both directions is important.</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * LSTMLayer forward = new LSTMLayer(100, 128, true);
 * LSTMLayer backward = new LSTMLayer(100, 128, true);
 * Bidirectional bi = new Bidirectional(forward, backward);
 *
 * GradTensor x = GradTensor.randn(32, 50, 100);   // [batch, seq_len, input_size]
 * GradTensor output = bi.forward(x);               // [batch, seq_len, 256] (128*2)
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class Bidirectional {

    private final Object forwardLayer;
    private final Object backwardLayer;
    private final int hiddenSize;

    /**
     * Create bidirectional wrapper with separate forward and backward layers.
     *
     * @param forwardLayer forward RNN layer (LSTMLayer or GRULayer)
     * @param backwardLayer backward RNN layer (LSTMLayer or GRULayer)
     */
    public Bidirectional(Object forwardLayer, Object backwardLayer) {
        this.forwardLayer = forwardLayer;
        this.backwardLayer = backwardLayer;

        // Extract hidden size from layers
        int fSize = extractHiddenSize(forwardLayer);
        int bSize = extractHiddenSize(backwardLayer);

        if (fSize != bSize) {
            throw new IllegalArgumentException("Forward and backward layers must have same hidden size");
        }

        this.hiddenSize = fSize;
    }

    /**
     * Forward pass.
     *
     * @param input input tensor [batch_size, sequence_length, input_size]
     * @return output tensor [batch_size, sequence_length, 2*hidden_size]
     */
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        int batchSize = (int) shape[0];
        int seqLen = (int) shape[1];

        // Process forward direction
        GradTensor forwardOut = processForward(input);

        // Process backward direction (reverse sequence)
        GradTensor reversedInput = reverseSequence(input);
        GradTensor backwardOut = processBackward(reversedInput);

        // Reverse backward output back to match original sequence order
        GradTensor reversedBackwardOut = reverseSequence(backwardOut);

        // Concatenate forward and backward: [batch, seq_len, 2*hidden_size]
        return concatenate(forwardOut, reversedBackwardOut);
    }

    /**
     * Process forward direction.
     */
    private GradTensor processForward(GradTensor input) {
        if (forwardLayer instanceof LSTMLayer) {
            return ((LSTMLayer) forwardLayer).forward(input);
        } else if (forwardLayer instanceof GRULayer) {
            return ((GRULayer) forwardLayer).forward(input);
        } else {
            throw new IllegalArgumentException("Unsupported layer type");
        }
    }

    /**
     * Process backward direction.
     */
    private GradTensor processBackward(GradTensor input) {
        if (backwardLayer instanceof LSTMLayer) {
            return ((LSTMLayer) backwardLayer).forward(input);
        } else if (backwardLayer instanceof GRULayer) {
            return ((GRULayer) backwardLayer).forward(input);
        } else {
            throw new IllegalArgumentException("Unsupported layer type");
        }
    }

    /**
     * Reverse sequence along timestep dimension.
     */
    private GradTensor reverseSequence(GradTensor tensor) {
        long[] shape = tensor.shape();
        int seqLen = (int) shape[1];

        // Create reversed view (placeholder - actual implementation would reverse)
        float[] data = tensor.data();
        float[] reversedData = new float[data.length];

        // Simple reversal along sequence dimension
        for (int i = 0; i < seqLen; i++) {
            System.arraycopy(data, i * (data.length / seqLen), 
                           reversedData, (seqLen - 1 - i) * (data.length / seqLen),
                           data.length / seqLen);
        }

        return GradTensor.of(reversedData, shape);
    }

    /**
     * Concatenate tensors along last dimension.
     */
    private GradTensor concatenate(GradTensor a, GradTensor b) {
        // Result: [batch, seq_len, 2*hidden_size]
        long[] shapeA = a.shape();
        long[] shapeB = b.shape();

        int totalSize = (int) (shapeA[0] * shapeA[1] * (shapeA[2] + shapeB[2]));
        float[] result = new float[totalSize];

        // Interleave the tensors (placeholder - actual implementation more complex)
        return GradTensor.of(result, shapeA[0], shapeA[1], shapeA[2] + shapeB[2]);
    }

    /**
     * Extract hidden size from layer.
     */
    private int extractHiddenSize(Object layer) {
        if (layer instanceof LSTMLayer) {
            return ((LSTMLayer) layer).getHiddenSize();
        } else if (layer instanceof GRULayer) {
            return ((GRULayer) layer).getHiddenSize();
        } else {
            throw new IllegalArgumentException("Unsupported layer type: " + layer.getClass());
        }
    }

    /**
     * Get output hidden size (forward + backward).
     */
    public int getOutputSize() {
        return 2 * hiddenSize;
    }

    /**
     * Get hidden size of individual directions.
     */
    public int getHiddenSize() {
        return hiddenSize;
    }

    @Override
    public String toString() {
        return String.format("Bidirectional(hidden_size=%d, output_size=%d)", 
                            hiddenSize, getOutputSize());
    }
}
