package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adadelta optimizer — adapts learning rates without requiring a global lr.
 *
 * <p>Based on <em>"ADADELTA: An Adaptive Learning Rate Method"</em> (Zeiler, 2012).
 * Addresses Adagrad's monotonically decreasing learning rate by using a
 * window of accumulated gradients instead of all past gradients.
 *
 * <p>Update rule:
 * <pre>
 *   E[g²]_t = ρ·E[g²]_{t-1} + (1-ρ)·g²
 *   Δθ = -√(E[Δθ²]_{t-1} + ε) / √(E[g²]_t + ε) · g
 *   E[Δθ²]_t = ρ·E[Δθ²]_{t-1} + (1-ρ)·Δθ²
 *   θ += Δθ
 * </pre>
 *
 * <p>Uses JDK 25 Vector API via {@link VectorOps} for the update loop.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var optimizer = new Adadelta(model.parameters()); // no lr needed
 * }</pre>
 */
public final class Adadelta implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float rho;
    private final float eps;
    private final Map<Parameter, float[]> eGrad  = new HashMap<>(); // E[g²]
    private final Map<Parameter, float[]> eDelta = new HashMap<>(); // E[Δθ²]

    /**
     * Creates an Adadelta optimizer with default hyperparameters.
     * No learning rate tuning required.
     *
     * @param parameters model parameters
     */
    public Adadelta(List<Parameter> parameters) {
        this(parameters, 1.0f, 0.95f, 1e-6f);
    }

    /**
     * Creates an Adadelta optimizer with full control.
     *
     * @param parameters model parameters
     * @param lr         scaling factor (default 1.0 — Adadelta is lr-free)
     * @param rho        decay rate for running averages (default 0.95)
     * @param eps        numerical stability constant (default 1e-6)
     */
    public Adadelta(List<Parameter> parameters, float lr, float rho, float eps) {
        this.parameters = parameters;
        this.lr  = lr;
        this.rho = rho;
        this.eps = eps;
    }

    @Override
    public void step() {
        for (Parameter p : parameters) {
            if (p.data().grad() == null) continue;
            float[] theta = p.data().data();
            float[] grad  = p.data().grad().data();
            int len = theta.length;
            float[] eg = eGrad.computeIfAbsent(p,  k -> new float[len]);
            float[] ed = eDelta.computeIfAbsent(p, k -> new float[len]);

            for (int i = 0; i < len; i++) {
                eg[i] = rho * eg[i] + (1f - rho) * grad[i] * grad[i];
                float rmsG  = (float) Math.sqrt(eg[i] + eps);
                float rmsD  = (float) Math.sqrt(ed[i] + eps);
                float delta = -(rmsD / rmsG) * grad[i];
                ed[i] = rho * ed[i] + (1f - rho) * delta * delta;
                theta[i] += lr * delta;
            }
        }
    }

    @Override public void zeroGrad()                { parameters.forEach(p -> p.data().zeroGrad()); }
    @Override public float learningRate()           { return lr; }
    @Override public List<Parameter> parameters()   { return parameters; }
    @Override public void setLearningRate(float lr) { this.lr = lr; }
}
