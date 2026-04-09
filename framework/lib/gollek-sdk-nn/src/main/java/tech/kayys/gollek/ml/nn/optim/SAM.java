package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.List;

/**
 * SAM (Sharpness Aware Minimization) optimizer — seeks flat minima for
 * better generalization by perturbing weights toward the sharpest direction.
 *
 * <p>Based on <em>"Sharpness-Aware Minimization for Efficiently Improving
 * Generalization"</em> (Foret et al., 2021).
 *
 * <p>Two-step update per batch:
 * <ol>
 *   <li>Compute gradient at θ, perturb: θ̂ = θ + ρ·g/||g||</li>
 *   <li>Compute gradient at θ̂, update θ with base optimizer</li>
 *   <li>Restore θ from perturbation</li>
 * </ol>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * var sam = new SAM(model.parameters(), new SGD(model.parameters(), 0.1f), rho=0.05f);
 *
 * // First forward+backward (at θ)
 * loss.backward();
 * sam.firstStep();
 *
 * // Second forward+backward (at θ̂)
 * model.zeroGrad();
 * loss2.backward();
 * sam.secondStep();
 * }</pre>
 */
public final class SAM implements Optimizer {

    private final List<Parameter> parameters;
    private final Optimizer base;
    private final float rho;
    private final float[][] savedWeights; // snapshot before perturbation

    /**
     * Creates a SAM optimizer.
     *
     * @param parameters model parameters
     * @param base       base optimizer for the second step (SGD or Adam)
     * @param rho        perturbation radius (default 0.05)
     */
    public SAM(List<Parameter> parameters, Optimizer base, float rho) {
        this.parameters   = parameters;
        this.base         = base;
        this.rho          = rho;
        this.savedWeights = new float[parameters.size()][];
    }

    /** Creates SAM with default rho=0.05. */
    public SAM(List<Parameter> parameters, Optimizer base) { this(parameters, base, 0.05f); }

    /**
     * First step: saves current weights and perturbs toward sharpest direction.
     * Call after the first backward pass.
     */
    public void firstStep() {
        // Compute global gradient norm
        float totalSq = 0f;
        for (Parameter p : parameters) {
            if (p.data().grad() == null) continue;
            float[] g = p.data().grad().data();
            float[] sq = new float[g.length];
            VectorOps.mul(g, g, sq);
            totalSq += VectorOps.sum(sq);
        }
        float norm = (float) Math.sqrt(totalSq) + 1e-12f;
        float scale = rho / norm;

        // Save weights and apply perturbation: θ̂ = θ + ρ·g/||g||
        for (int i = 0; i < parameters.size(); i++) {
            Parameter p = parameters.get(i);
            float[] theta = p.data().data();
            savedWeights[i] = theta.clone();
            if (p.data().grad() == null) continue;
            float[] g = p.data().grad().data();
            for (int j = 0; j < theta.length; j++) theta[j] += scale * g[j];
        }
    }

    /**
     * Second step: restores original weights and applies base optimizer update.
     * Call after the second backward pass (at perturbed weights).
     */
    public void secondStep() {
        // Restore original weights
        for (int i = 0; i < parameters.size(); i++) {
            if (savedWeights[i] != null)
                System.arraycopy(savedWeights[i], 0, parameters.get(i).data().data(), 0, savedWeights[i].length);
        }
        // Apply base optimizer with gradients computed at θ̂
        base.step();
    }

    /** Delegates to secondStep for compatibility with standard training loops. */
    @Override public void step() { secondStep(); }

    @Override public void zeroGrad()                { parameters.forEach(p -> p.data().zeroGrad()); }
    @Override public float learningRate()           { return base.learningRate(); }
    @Override public void setLearningRate(float lr) { base.setLearningRate(lr); }
    @Override public List<Parameter> parameters()   { return parameters; }
}
