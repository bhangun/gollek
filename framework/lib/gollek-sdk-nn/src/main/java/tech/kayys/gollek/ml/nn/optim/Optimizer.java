package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;

/**
 * Base interface for all gradient-based optimizers.
 * <p>
 * Optimizers update neural network parameters based on computed gradients
 * to minimize the loss function. Different optimizers use different update
 * strategies: simple gradient descent, momentum-based methods, adaptive methods, etc.
 * <p>
 * Implementations must maintain parameter references and update them in-place
 * during the {@link #step()} call.
 *
 * <h3>Typical Training Loop</h3>
 * <pre>{@code
 * var model = new MyModel();
 * var optimizer = new SGD(model.parameters(), 0.01f);  // learning rate 0.01
 * var loss_fn = new CrossEntropyLoss();
 *
 * for (int epoch = 0; epoch < numEpochs; epoch++) {
 *     for (var batch : dataloader) {
 *         // 1. Forward pass
 *         var predictions = model.forward(batch.input);
 *         var loss = loss_fn.compute(predictions, batch.target);
 *
 *         // 2. Backward pass (compute gradients)
 *         loss.backward();
 *
 *         // 3. Optimizer step (update parameters)
 *         optimizer.step();
 *
 *         // 4. Clear gradients for next iteration
 *         optimizer.zeroGrad();
 *     }
 * }
 * }</pre>
 *
 * <h3>Available Implementations</h3>
 * <ul>
 *   <li><b>SGD:</b> Simple stochastic gradient descent with optional momentum</li>
 *   <li><b>AdamW:</b> Adam optimizer with decoupled weight decay (recommended)</li>
 *   <li><b>Adam:</b> Adaptive moment estimation (in development)</li>
 *   <li><b>RMSprop:</b> Root mean square propagation (in development)</li>
 * </ul>
 *
 * <h3>Learning Rate Scheduling</h3>
 * <p>
 * Learning rate can be adjusted during training using the setLearningRate() method:
 * <pre>{@code
 * float initialLr = 0.001f;
 * var optimizer = new AdamW(model.parameters(), initialLr);
 *
 * for (int epoch = 0; epoch < 100; epoch++) {
 *     // Adjust learning rate (e.g., cosine annealing, step decay)
 *     float newLr = initialLr * (float) Math.cos(Math.PI * epoch / 100);
 *     optimizer.setLearningRate(newLr);
 *
 *     // Training...
 * }
 * }</pre>
 *
 * <h3>Key Concepts</h3>
 * <ul>
 *   <li><b>Gradient:</b> Direction of steepest increase of loss; optimizer moves opposite</li>
 *   <li><b>Learning rate:</b> Step size for parameter updates; too high = instability, too low = slow</li>
 *   <li><b>Momentum:</b> Accumulates gradients to accelerate convergence</li>
 *   <li><b>Adaptive methods:</b> Scale learning rate per parameter based on gradient history</li>
 *   <li><b>Weight decay:</b> L2 regularization penalty to prevent overfitting</li>
 * </ul>
 *
 * <h3>Recommendations</h3>
 * <ul>
 *   <li>For transformers and modern deep learning: <b>AdamW</b> (decoupled weight decay)</li>
 *   <li>For simple tasks: <b>SGD with momentum</b></li>
 *   <li>For unknown scenarios: Start with <b>AdamW(lr=1e-3)</b></li>
 *   <li>Always use learning rate scheduling for training stability</li>
 * </ul>
 *
 * @see tech.kayys.gollek.ml.nn.optim.SGD
 * @see tech.kayys.gollek.ml.nn.optim.AdamW
 */
public interface Optimizer {

    /**
     * Perform a single optimization step.
     * <p>
     * Updates all parameters based on their accumulated gradients.
     * Must be called after loss.backward() and before zeroGrad().
     *
     * @throws IllegalStateException if gradients have not been computed
     */
    void step();

    /**
     * Zero all parameter gradients.
     * <p>
     * Call this after optimizer.step() to prepare for the next forward pass.
     * Prevents gradient accumulation across iterations.
     */
    void zeroGrad();

    /**
     * Get the current learning rate.
     *
     * @return the learning rate (positive float)
     */
    float learningRate();

    /**
     * Returns the list of parameters managed by this optimizer.
     *
     * @return parameter list
     */
    List<Parameter> parameters();

    /**
     * Update the learning rate (for learning rate scheduling).
     * <p>
     * Commonly used with learning rate schedules to adjust the learning rate
     * during training (e.g., cosine annealing, step decay, exponential decay).
     *
     * @param lr new learning rate (should be positive)
     *
     * @throws IllegalArgumentException if lr is not positive
     */
    void setLearningRate(float lr);
}
