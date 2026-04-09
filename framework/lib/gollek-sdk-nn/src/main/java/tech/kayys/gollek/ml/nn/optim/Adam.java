package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * Adam Optimizer (original, without weight decay).
 * <p>
 * Adam (Adaptive Moment Estimation) combines momentum-based gradient descent with
 * adaptive learning rates. It computes individual adaptive learning rates for each parameter.
 * <p>
 * State tracking per parameter:
 * <ul>
 *   <li>m_t: exponential moving average of gradients (first moment)</li>
 *   <li>v_t: exponential moving average of squared gradients (second moment)</li>
 * </ul>
 *
 * <h3>Update Rule</h3>
 * <pre>
 * m_t = β1 * m_{t-1} + (1 - β1) * g_t           (first moment, momentum)
 * v_t = β2 * v_{t-1} + (1 - β2) * g_t²          (second moment, RMSprop)
 * m_hat = m_t / (1 - β1^t)                       (bias correction)
 * v_hat = v_t / (1 - β2^t)                       (bias correction)
 * θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)
 * </pre>
 *
 * <h3>Parameters</h3>
 * <ul>
 *   <li><b>α (lr):</b> Learning rate (typically 1e-3)</li>
 *   <li><b>β1:</b> Momentum coefficient (typically 0.9)</li>
 *   <li><b>β2:</b> RMSprop coefficient (typically 0.999)</li>
 *   <li><b>ε:</b> Numerical stability (typically 1e-8)</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var optimizer = new Adam(model.parameters(), 0.001f);  // lr=0.001
 * optimizer.step();
 * optimizer.zeroGrad();
 * }</example>
 *
 * <h3>Advantages</h3>
 * <ul>
 *   <li>Adaptive learning rates per parameter</li>
 *   <li>Combines momentum and RMSprop</li>
 *   <li>Works well with little tuning</li>
 *   <li>Good default for most problems</li>
 * </ul>
 *
 * <h3>Notes</h3>
 * <ul>
 *   <li>Original Adam (this implementation)</li>
 *   <li>For weight decay: use AdamW (decoupled weight decay)</li>
 *   <li>More memory than SGD (maintains m and v for each parameter)</li>
 * </ul>
 *
 * @see AdamW
 * @see SGD
 */
public class Adam implements Optimizer {

    private final List<Parameter> parameters;
    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final Map<Parameter, float[]> m;  // first moment
    private final Map<Parameter, float[]> v;  // second moment
    private int step = 0;

    /**
     * Create an Adam optimizer with default parameters.
     *
     * @param parameters model parameters to optimize
     * @param learningRate learning rate (typically 1e-3)
     */
    public Adam(List<Parameter> parameters, float learningRate) {
        this(parameters, learningRate, 0.9f, 0.999f, 1e-8f);
    }

    /**
     * Create an Adam optimizer with custom parameters.
     *
     * @param parameters model parameters to optimize
     * @param learningRate learning rate
     * @param beta1 momentum coefficient (typically 0.9)
     * @param beta2 RMSprop coefficient (typically 0.999)
     * @param epsilon numerical stability constant
     */
    public Adam(List<Parameter> parameters, float learningRate, float beta1, float beta2, float epsilon) {
        this.parameters = parameters;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.m = new HashMap<>();
        this.v = new HashMap<>();

        // Initialize first and second moments
        for (Parameter param : parameters) {
            m.put(param, new float[(int) param.data().numel()]);
            v.put(param, new float[(int) param.data().numel()]);
        }
    }

    /**
     * Perform a single Adam optimization step.
     */
    @Override
    public void step() {
        step++;
        for (Parameter param : parameters) {
            float[] data = param.data().data();
            float[] grad = param.grad().data();
            float[] m_t = m.get(param);
            float[] v_t = v.get(param);

            for (int i = 0; i < data.length; i++) {
                // Update biased first moment estimate
                m_t[i] = beta1 * m_t[i] + (1 - beta1) * grad[i];
                // Update biased second raw moment estimate
                v_t[i] = beta2 * v_t[i] + (1 - beta2) * grad[i] * grad[i];

                // Compute bias-corrected first moment estimate
                float m_hat = m_t[i] / (1 - (float) Math.pow(beta1, step));
                // Compute bias-corrected second raw moment estimate
                float v_hat = v_t[i] / (1 - (float) Math.pow(beta2, step));

                // Update parameter
                data[i] -= learningRate * m_hat / ((float) Math.sqrt(v_hat) + epsilon);
            }
        }
    }

    /**
     * Zero all parameter gradients.
     */
    @Override
    public void zeroGrad() {
        for (Parameter param : parameters) {
            param.zeroGrad();
        }
    }

    /**
     * Get current learning rate.
     *
     * @return learning rate
     */
    @Override
    public float learningRate() {
        return learningRate;
    }

    /**
     * Set learning rate (for learning rate scheduling).
     *
     * @param lr new learning rate
     */
    @Override
    public void setLearningRate(float lr) {
        if (lr <= 0) {
            throw new IllegalArgumentException("learning rate must be positive, got: " + lr);
        }
        this.learningRate = lr;
    }

    @Override
    public List<Parameter> parameters() {
        return parameters;
    }

    @Override
    public String toString() {
        return "Adam(lr=" + learningRate + ", beta1=" + beta1 + ", beta2=" + beta2 + ")";
    }
}
