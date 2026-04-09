package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;

/**
 * Adam optimizer with decoupled weight decay (AdamW).
 * <p>
 * Implements the algorithm from "Decoupled Weight Decay Regularization"
 * by Loshchilov & Hutter. This is the default optimizer for training
 * transformer models.
 * <p>
 * Equivalent to {@code torch.optim.AdamW}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var optimizer = new AdamW(model.parameters(), 1e-4f);
 * // Training loop:
 * optimizer.zeroGrad();
 * var loss = lossFunction.compute(output, target);
 * loss.backward();
 * optimizer.step();
 * }</pre>
 */
public class AdamW implements Optimizer {

    private final List<Parameter> params;
    private float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private int step = 0;

    private final float[][] m;  // first moment
    private final float[][] v;  // second moment

    public AdamW(List<Parameter> params, float lr) {
        this(params, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public AdamW(List<Parameter> params, float lr, float weightDecay) {
        this(params, lr, 0.9f, 0.999f, 1e-8f, weightDecay);
    }

    public AdamW(List<Parameter> params, float lr, float beta1, float beta2, float eps, float weightDecay) {
        this.params = params;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.m = new float[params.size()][];
        this.v = new float[params.size()][];
        for (int i = 0; i < params.size(); i++) {
            int n = (int) params.get(i).numel();
            m[i] = new float[n];
            v[i] = new float[n];
        }
    }

    @Override
    public void step() {
        step++;
        float bc1 = 1.0f - (float) Math.pow(beta1, step);
        float bc2 = 1.0f - (float) Math.pow(beta2, step);

        for (int p = 0; p < params.size(); p++) {
            Parameter param = params.get(p);
            GradTensor gradTensor = param.grad();
            if (gradTensor == null) continue;

            float[] data = param.data().data();
            float[] grad = gradTensor.data();
            float[] mArr = this.m[p];
            float[] vArr = this.v[p];

            for (int i = 0; i < data.length; i++) {
                // Decoupled weight decay
                data[i] *= (1.0f - lr * weightDecay);

                // Update biased first moment estimate
                mArr[i] = beta1 * mArr[i] + (1.0f - beta1) * grad[i];
                // Update biased second raw moment estimate
                vArr[i] = beta2 * vArr[i] + (1.0f - beta2) * grad[i] * grad[i];

                // Bias-corrected estimates
                float mHat = mArr[i] / bc1;
                float vHat = vArr[i] / bc2;

                // Update parameters
                data[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }
        }
    }

    @Override
    public void zeroGrad() {
        for (Parameter p : params) p.zeroGrad();
    }

    @Override
    public float learningRate() { return lr; }

    @Override
    public List<Parameter> parameters() {
        return params;
    }

    @Override
    public void setLearningRate(float lr) { this.lr = lr; }

    /** Current step count. */
    public int currentStep() { return step; }

    @Override
    public String toString() {
        return "AdamW(lr=" + lr + ", betas=(" + beta1 + ", " + beta2 + "), eps=" + eps
            + ", weight_decay=" + weightDecay + ")";
    }
}
