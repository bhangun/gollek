package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adagrad optimizer — adapts learning rates per parameter based on the sum
 * of squared historical gradients.
 *
 * <p>Parameters that receive large gradients get smaller effective learning rates,
 * making Adagrad well-suited for sparse features (NLP, recommendation systems).
 *
 * <p>Update rule:
 * <pre>
 *   G_t = G_{t-1} + g²
 *   θ  -= lr / (√G_t + ε) · g
 * </pre>
 *
 * <p>Uses JDK 25 Vector API via {@link VectorOps} for the update loop.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var optimizer = new Adagrad(model.parameters(), lr = 0.01f);
 * }</pre>
 */
public final class Adagrad implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float eps;
    private final float weightDecay;
    private final Map<Parameter, float[]> sumSq = new HashMap<>();

    /**
     * Creates an Adagrad optimizer with default epsilon.
     *
     * @param parameters model parameters
     * @param lr         learning rate (default 0.01 for Adagrad)
     */
    public Adagrad(List<Parameter> parameters, float lr) {
        this(parameters, lr, 1e-8f, 0f);
    }

    /**
     * Creates an Adagrad optimizer with full control.
     *
     * @param parameters  model parameters
     * @param lr          learning rate
     * @param eps         numerical stability constant (default 1e-8)
     * @param weightDecay L2 regularization coefficient
     */
    public Adagrad(List<Parameter> parameters, float lr, float eps, float weightDecay) {
        this.parameters  = parameters;
        this.lr          = lr;
        this.eps         = eps;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step() {
        for (Parameter p : parameters) {
            if (p.data().grad() == null) continue;
            float[] theta = p.data().data();
            float[] grad  = p.data().grad().data();
            int len = theta.length;
            float[] G = sumSq.computeIfAbsent(p, k -> new float[len]);

            // G += g²  then  θ -= lr / (√G + ε) * g  — vectorized
            int i = 0;
            int bound = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.loopBound(len);
            var SPECIES = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
            var lrV  = jdk.incubator.vector.FloatVector.broadcast(SPECIES, lr);
            var epsV = jdk.incubator.vector.FloatVector.broadcast(SPECIES, eps);

            for (; i < bound; i += SPECIES.length()) {
                var g  = jdk.incubator.vector.FloatVector.fromArray(SPECIES, grad, i);
                var Gi = jdk.incubator.vector.FloatVector.fromArray(SPECIES, G, i);
                var newG = Gi.add(g.mul(g));
                newG.intoArray(G, i);
                var d = jdk.incubator.vector.FloatVector.fromArray(SPECIES, theta, i);
                d.sub(lrV.mul(g).div(newG.sqrt().add(epsV))).intoArray(theta, i);
            }
            for (; i < len; i++) {
                float g = grad[i] + weightDecay * theta[i];
                G[i] += g * g;
                theta[i] -= lr / ((float) Math.sqrt(G[i]) + eps) * g;
            }
        }
    }

    @Override public void zeroGrad()                  { parameters.forEach(p -> p.data().zeroGrad()); }
    @Override public float learningRate()             { return lr; }
    @Override public List<Parameter> parameters()     { return parameters; }
    @Override public void setLearningRate(float lr)   { this.lr = lr; }
}
