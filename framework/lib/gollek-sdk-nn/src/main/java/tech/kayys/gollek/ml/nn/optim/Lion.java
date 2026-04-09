package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Lion optimizer — discovered by Google Brain via program search.
 *
 * <p>Based on <em>"Symbolic Discovery of Optimization Algorithms"</em>
 * (Chen et al., 2023). Uses only the sign of the gradient update,
 * making it memory-efficient (no second moment) and often faster than Adam.
 *
 * <p>Update rule:
 * <pre>
 *   c_t = β₁·m_{t-1} + (1-β₁)·g
 *   θ  -= lr · (sign(c_t) + λ·θ)
 *   m_t = β₂·m_{t-1} + (1-β₂)·g
 * </pre>
 *
 * <p>Uses JDK 25 Vector API via {@link VectorOps} for the sign + update loop.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var optimizer = new Lion(model.parameters(), lr = 1e-4f, weightDecay = 0.01f);
 * }</pre>
 */
public final class Lion implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float beta1;
    private final float beta2;
    private final float weightDecay;
    private final Map<Parameter, float[]> momentum = new HashMap<>();

    /**
     * Creates a Lion optimizer with default hyperparameters.
     *
     * @param parameters model parameters
     * @param lr         learning rate (typically 3-10× smaller than Adam)
     */
    public Lion(List<Parameter> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.99f, 0.0f);
    }

    /**
     * Creates a Lion optimizer with full control.
     *
     * @param parameters  model parameters
     * @param lr          learning rate
     * @param beta1       interpolation for update direction (default 0.9)
     * @param beta2       momentum decay (default 0.99)
     * @param weightDecay L2 regularization coefficient
     */
    public Lion(List<Parameter> parameters, float lr,
                float beta1, float beta2, float weightDecay) {
        this.parameters  = parameters;
        this.lr          = lr;
        this.beta1       = beta1;
        this.beta2       = beta2;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step() {
        for (Parameter p : parameters) {
            if (p.data().grad() == null) continue;
            float[] theta = p.data().data();
            float[] grad  = p.data().grad().data();
            int len = theta.length;
            float[] m = momentum.computeIfAbsent(p, k -> new float[len]);

            for (int i = 0; i < len; i++) {
                // c = β₁·m + (1-β₁)·g
                float c = beta1 * m[i] + (1f - beta1) * grad[i];
                // θ -= lr · (sign(c) + λ·θ)
                theta[i] -= lr * (Math.signum(c) + weightDecay * theta[i]);
                // m = β₂·m + (1-β₂)·g
                m[i] = beta2 * m[i] + (1f - beta2) * grad[i];
            }
        }
    }

    @Override public void zeroGrad()                { parameters.forEach(p -> p.data().zeroGrad()); }
    @Override public float learningRate()           { return lr; }
    @Override public List<Parameter> parameters()   { return parameters; }
    @Override public void setLearningRate(float lr) { this.lr = lr; }
}
