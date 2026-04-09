package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;

/**
 * Lookahead optimizer — wraps any base optimizer with a slow-weights update.
 *
 * <p>Based on <em>"Lookahead Optimizer: k steps forward, 1 step back"</em>
 * (Zhang et al., 2019). Improves convergence stability by maintaining slow
 * weights that interpolate toward fast weights every k steps.
 *
 * <p>Update rule (every k steps):
 * <pre>
 *   slow_weights += alpha * (fast_weights - slow_weights)
 *   fast_weights  = slow_weights
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var base = new Adam(model.parameters(), 0.001f);
 * var optimizer = new Lookahead(base, k=5, alpha=0.5f);
 * }</pre>
 */
public final class Lookahead implements Optimizer {

    private final Optimizer inner;
    private final int k;
    private final float alpha;
    private int stepCount = 0;

    /** Slow weights snapshot per parameter (flat index → slow weight array). */
    private final java.util.Map<Parameter, float[]> slowWeights = new java.util.HashMap<>();

    /**
     * Wraps a base optimizer with Lookahead.
     *
     * @param inner base optimizer (Adam, SGD, etc.)
     * @param k     number of inner steps before slow-weight update (default 5)
     * @param alpha slow-weight interpolation factor (default 0.5)
     */
    public Lookahead(Optimizer inner, int k, float alpha) {
        this.inner = inner;
        this.k     = k;
        this.alpha = alpha;
    }

    /** Wraps with default k=5, alpha=0.5. */
    public Lookahead(Optimizer inner) { this(inner, 5, 0.5f); }

    @Override
    public void step() {
        inner.step();
        stepCount++;

        if (stepCount % k == 0) {
            // Slow-weight update: slow += alpha * (fast - slow)
            for (Parameter p : inner.parameters()) {
                float[] fast = p.data().data();
                float[] slow = slowWeights.computeIfAbsent(p, x -> fast.clone());
                for (int i = 0; i < fast.length; i++) {
                    slow[i] += alpha * (fast[i] - slow[i]);
                    fast[i]  = slow[i]; // reset fast to slow
                }
            }
        }
    }

    @Override public void zeroGrad()                { inner.zeroGrad(); }
    @Override public float learningRate()           { return inner.learningRate(); }
    @Override public void setLearningRate(float lr) { inner.setLearningRate(lr); }

    /** @return the wrapped base optimizer */
    public Optimizer inner() { return inner; }

    /** Exposes parameters from the inner optimizer. */
    public List<Parameter> parameters() { return inner.parameters(); }
}
