package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * LAMB optimizer — Layer-wise Adaptive Moments for Batch training.
 *
 * <p>Designed for large-batch training (e.g. BERT pre-training with batch size 32K+).
 * Extends Adam with a layer-wise trust ratio that scales the update per layer.
 *
 * <p>Based on <em>"Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"</em>
 * (You et al., 2019).
 *
 * <p>Update rule:
 * <pre>
 *   m_t = β₁·m_{t-1} + (1-β₁)·g
 *   v_t = β₂·v_{t-1} + (1-β₂)·g²
 *   m̂ = m_t / (1-β₁ᵗ),  v̂ = v_t / (1-β₂ᵗ)
 *   r = m̂ / (√v̂ + ε) + λ·θ          (Adam update + weight decay)
 *   trust = ||θ|| / ||r||              (layer-wise trust ratio)
 *   θ -= lr · trust · r
 * </pre>
 *
 * <p>Uses JDK 25 Vector API via {@link VectorOps} for the inner update loops.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var optimizer = new LAMB(model.parameters(), lr=1e-3f, weightDecay=0.01f);
 * }</pre>
 */
public final class LAMB implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;

    private final Map<Parameter, float[]> m = new HashMap<>(); // 1st moment
    private final Map<Parameter, float[]> v = new HashMap<>(); // 2nd moment
    private int step = 0;

    /**
     * Creates a LAMB optimizer with default hyperparameters.
     *
     * @param parameters model parameters to optimize
     * @param lr         learning rate
     */
    public LAMB(List<Parameter> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.999f, 1e-6f, 0.01f);
    }

    /**
     * Creates a LAMB optimizer with full hyperparameter control.
     *
     * @param parameters  model parameters
     * @param lr          learning rate
     * @param beta1       first moment decay (default 0.9)
     * @param beta2       second moment decay (default 0.999)
     * @param eps         numerical stability constant (default 1e-6)
     * @param weightDecay L2 regularization coefficient (default 0.01)
     */
    public LAMB(List<Parameter> parameters, float lr,
                float beta1, float beta2, float eps, float weightDecay) {
        this.parameters  = parameters;
        this.lr          = lr;
        this.beta1       = beta1;
        this.beta2       = beta2;
        this.eps         = eps;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step() {
        step++;
        float bc1 = 1f - (float) Math.pow(beta1, step);
        float bc2 = 1f - (float) Math.pow(beta2, step);

        for (Parameter p : parameters) {
            if (p.data().grad() == null) continue;
            float[] theta = p.data().data();
            float[] grad  = p.data().grad().data();
            int len = theta.length;

            float[] mt = m.computeIfAbsent(p, k -> new float[len]);
            float[] vt = v.computeIfAbsent(p, k -> new float[len]);

            // Adam moments + bias correction
            float[] r = new float[len];
            for (int i = 0; i < len; i++) {
                mt[i] = beta1 * mt[i] + (1f - beta1) * grad[i];
                vt[i] = beta2 * vt[i] + (1f - beta2) * grad[i] * grad[i];
                float mHat = mt[i] / bc1;
                float vHat = vt[i] / bc2;
                r[i] = mHat / ((float) Math.sqrt(vHat) + eps) + weightDecay * theta[i];
            }

            // Layer-wise trust ratio: ||θ|| / ||r||
            float[] thetaSq = new float[len], rSq = new float[len];
            VectorOps.mul(theta, theta, thetaSq);
            VectorOps.mul(r, r, rSq);
            float normTheta = (float) Math.sqrt(VectorOps.sum(thetaSq));
            float normR     = (float) Math.sqrt(VectorOps.sum(rSq));
            float trust = (normTheta > 0 && normR > 0) ? normTheta / normR : 1f;

            // Update: θ -= lr * trust * r  (SIMD via VectorOps)
            VectorOps.mulScalar(r, lr * trust, r);
            float[] neg = new float[len];
            VectorOps.mulScalar(r, -1f, neg);
            VectorOps.add(theta, neg, theta);
        }
    }

    @Override public void zeroGrad() { parameters.forEach(p -> p.data().zeroGrad()); }
    @Override public float learningRate() { return lr; }
    @Override public List<Parameter> parameters() { return parameters; }
    @Override public void setLearningRate(float lr) { this.lr = lr; }
}
