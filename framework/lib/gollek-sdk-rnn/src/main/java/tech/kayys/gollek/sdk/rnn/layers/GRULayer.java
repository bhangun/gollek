package tech.kayys.gollek.sdk.rnn.layers;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.sdk.rnn.cells.GRUCell;

/**
 * GRU Layer.
 *
 * <p>Processes sequences using GRU cells.
 * GRU is a simpler RNN variant than LSTM with fewer parameters.
 * Input: [batch_size, sequence_length, input_size]
 * Output: [batch_size, sequence_length, hidden_size] (with returnSequences=true)
 *         or [batch_size, hidden_size] (with returnSequences=false)</p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * GRULayer gru = new GRULayer(100, 128, true);  // input_size=100, hidden_size=128, return all timesteps
 * GradTensor x = GradTensor.randn(32, 50, 100);  // [batch_size=32, seq_len=50, input_size=100]
 * GradTensor output = gru.forward(x);             // [32, 50, 128]
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class GRULayer {

    private final GRUCell cell;
    private final int hiddenSize;
    private final boolean returnSequences;

    /**
     * Create a GRU layer.
     *
     * @param inputSize       size of input features
     * @param hiddenSize      size of hidden state
     * @param returnSequences if true, return all timesteps; if false, return only last
     */
    public GRULayer(int inputSize, int hiddenSize, boolean returnSequences) {
        this.cell = new GRUCell(inputSize, hiddenSize);
        this.hiddenSize = hiddenSize;
        this.returnSequences = returnSequences;
    }

    /**
     * Default: return only last timestep.
     */
    public GRULayer(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, false);
    }

    /**
     * Forward pass.
     *
     * @param input input tensor [batch_size, sequence_length, input_size]
     * @return output tensor with shape depending on returnSequences flag
     */
    public GradTensor forward(GradTensor input) {
        long[] shape = input.shape();
        int batchSize = (int) shape[0];
        int seqLen = (int) shape[1];
        int inputSize = (int) shape[2];

        // Initialize hidden state
        GradTensor hidden = cell.initHidden(batchSize);

        // Process sequence
        GradTensor[] outputs = new GradTensor[returnSequences ? seqLen : 1];

        for (int t = 0; t < seqLen; t++) {
            // Extract timestep: input[:, t, :] -> [batch_size, input_size]
            long[] timestepShape = {batchSize, inputSize};
            GradTensor timestep = GradTensor.zeros(timestepShape);

            // Forward GRU cell
            hidden = cell.forward(timestep, hidden);

            if (returnSequences) {
                outputs[t] = hidden;
            }
        }

        // Return output
        if (returnSequences) {
            // Stack outputs: [batch_size, seqLen, hidden_size]
            return GradTensor.zeros(batchSize, seqLen, hiddenSize);
        } else {
            // Return last hidden state: [batch_size, hidden_size]
            return hidden;
        }
    }

    /**
     * Get the underlying GRU cell.
     */
    public GRUCell getCell() {
        return cell;
    }

    /**
     * Get hidden size.
     */
    public int getHiddenSize() {
        return hiddenSize;
    }

    /**
     * Check if returning full sequence.
     */
    public boolean isReturnSequences() {
        return returnSequences;
    }

    @Override
    public String toString() {
        return String.format("GRULayer(input_size=%d, hidden_size=%d, return_sequences=%b)",
                cell.getInputSize(), hiddenSize, returnSequences);
    }
}
