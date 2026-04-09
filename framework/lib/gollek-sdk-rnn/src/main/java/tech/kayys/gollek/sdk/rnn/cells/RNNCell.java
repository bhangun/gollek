package tech.kayys.gollek.sdk.rnn.cells;

import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Base interface for RNN cells.
 *
 * <p>Defines the contract for recurrent neural network cells.
 * Implementations should handle hidden state updates and gradient flow.</p>
 *
 * @author Gollek Team
 * @version 0.1.0
 */
public interface RNNCell {

    /**
     * Process a single timestep.
     *
     * @param input input tensor of shape [batch_size, input_size]
     * @param hidden previous hidden state of shape [batch_size, hidden_size]
     * @return new hidden state of shape [batch_size, hidden_size]
     */
    GradTensor forward(GradTensor input, GradTensor hidden);

    /**
     * Get the hidden state size.
     */
    int getHiddenSize();

    /**
     * Get the input size.
     */
    int getInputSize();

    /**
     * Initialize hidden state.
     *
     * @param batchSize batch size
     * @return initial hidden state of shape [batch_size, hidden_size]
     */
    GradTensor initHidden(int batchSize);

    /**
     * Get all learnable parameters.
     *
     * @return array of learnable parameters
     */
    GradTensor[] getParameters();
}
