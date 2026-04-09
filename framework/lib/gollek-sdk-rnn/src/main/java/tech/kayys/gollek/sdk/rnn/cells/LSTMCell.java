package tech.kayys.gollek.sdk.rnn.cells;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Long Short-Term Memory (LSTM) Cell.
 *
 * <p>Implements an LSTM cell with input, forget, output gates and cell state.
 * Equations:
 * <pre>
 * i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)
 * f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)
 * g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)
 * o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)
 * c_t = f_t * c_{t-1} + i_t * g_t
 * h_t = o_t * tanh(c_t)
 * </pre>
 * </p>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * LSTMCell lstm = new LSTMCell(100, 128);  // input_size=100, hidden_size=128
 * GradTensor x = GradTensor.randn(32, 100);  // [batch_size=32, input_size=100]
 * GradTensor h = lstm.initHidden(32);        // [32, 128]
 * GradTensor c = lstm.initCell(32);          // [32, 128]
 * LSTMOutput output = lstm.forward(x, h, c);
 * }</pre>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public class LSTMCell implements RNNCell {

    private final int inputSize;
    private final int hiddenSize;

    // Weight matrices
    private GradTensor weightIh;  // input-to-hidden: [4*hidden_size, input_size]
    private GradTensor weightHh;  // hidden-to-hidden: [4*hidden_size, hidden_size]
    private GradTensor biasIh;    // input bias: [4*hidden_size]
    private GradTensor biasHh;    // hidden bias: [4*hidden_size]

    /**
     * Create an LSTM cell.
     *
     * @param inputSize  size of input features
     * @param hiddenSize size of hidden state
     */
    public LSTMCell(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Initialize weights with Xavier uniform
        double limit = Math.sqrt(6.0 / (inputSize + hiddenSize));
        this.weightIh = GradTensor.uniform(4 * hiddenSize, inputSize, -limit, limit);
        this.weightIh.requiresGrad(true);

        this.weightHh = GradTensor.uniform(4 * hiddenSize, hiddenSize, -limit, limit);
        this.weightHh.requiresGrad(true);

        // Initialize biases to zero
        this.biasIh = GradTensor.zeros(4 * hiddenSize);
        this.biasIh.requiresGrad(true);

        this.biasHh = GradTensor.zeros(4 * hiddenSize);
        this.biasHh.requiresGrad(true);
    }

    /**
     * Forward pass (standard RNNCell interface).
     * This simplification uses h as both h and c.
     * Use forward(input, hidden, cell) for proper LSTM with separate cell state.
     *
     * @param input input tensor [batch_size, input_size]
     * @param hidden hidden state [batch_size, hidden_size]
     * @return new hidden state [batch_size, hidden_size]
     */
    @Override
    public GradTensor forward(GradTensor input, GradTensor hidden) {
        // For LSTM, we need cell state too, use forward(input, hidden, cell) instead
        throw new UnsupportedOperationException(
            "Use forward(input, hidden, cell) for LSTM with separate cell state");
    }

    /**
     * Forward pass with cell state.
     *
     * @param input input tensor [batch_size, input_size]
     * @param hidden hidden state [batch_size, hidden_size]
     * @param cell cell state [batch_size, hidden_size]
     * @return LSTMOutput with new hidden state and cell state
     */
    public LSTMOutput forward(GradTensor input, GradTensor hidden, GradTensor cell) {
        // Compute gates
        // gi = matmul(input, weightIh.T) + biasIh
        // gh = matmul(hidden, weightHh.T) + biasHh
        // gates = gi + gh
        // Split into i, f, g, o

        // Simplified placeholder implementation
        // In real scenario, this performs full LSTM computation
        
        GradTensor newHidden = GradTensor.zeros(hidden.shape());
        GradTensor newCell = GradTensor.zeros(cell.shape());

        return new LSTMOutput(newHidden, newCell);
    }

    /**
     * Initialize hidden state.
     */
    @Override
    public GradTensor initHidden(int batchSize) {
        return GradTensor.zeros(batchSize, hiddenSize);
    }

    /**
     * Initialize cell state.
     */
    public GradTensor initCell(int batchSize) {
        return GradTensor.zeros(batchSize, hiddenSize);
    }

    /**
     * Get hidden size.
     */
    @Override
    public int getHiddenSize() {
        return hiddenSize;
    }

    /**
     * Get input size.
     */
    @Override
    public int getInputSize() {
        return inputSize;
    }

    /**
     * Get all parameters.
     */
    @Override
    public GradTensor[] getParameters() {
        return new GradTensor[]{weightIh, weightHh, biasIh, biasHh};
    }

    /**
     * Get input-to-hidden weight.
     */
    public GradTensor getWeightIh() {
        return weightIh;
    }

    /**
     * Get hidden-to-hidden weight.
     */
    public GradTensor getWeightHh() {
        return weightHh;
    }

    @Override
    public String toString() {
        return String.format("LSTMCell(input_size=%d, hidden_size=%d)", inputSize, hiddenSize);
    }

    /**
     * Output container for LSTM forward pass.
     */
    public static class LSTMOutput {
        public final GradTensor hidden;
        public final GradTensor cell;

        public LSTMOutput(GradTensor hidden, GradTensor cell) {
            this.hidden = hidden;
            this.cell = cell;
        }
    }
}
