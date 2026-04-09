package tech.kayys.gollek.ml.nn.optim;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

import java.util.List;

/**
 * Stochastic Gradient Descent optimizer with optional momentum.
 * <p>
 * {@code v = momentum * v - lr * grad}
 * {@code param = param + v}
 * <p>
 * Equivalent to {@code torch.optim.SGD}.
 */
public class SGD implements Optimizer {

    private final List<Parameter> params;
    private float lr;
    private final float momentum;
    private final float weightDecay;
    private final float[][] velocities;

    public SGD(List<Parameter> params, float lr) {
        this(params, lr, 0.0f, 0.0f);
    }

    public SGD(List<Parameter> params, float lr, float momentum) {
        this(params, lr, momentum, 0.0f);
    }

    public SGD(List<Parameter> params, float lr, float momentum, float weightDecay) {
        this.params = params;
        this.lr = lr;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.velocities = new float[params.size()][];
        for (int i = 0; i < params.size(); i++) {
            velocities[i] = new float[(int) params.get(i).numel()];
        }
    }

    @Override
    public void step() {
        for (int p = 0; p < params.size(); p++) {
            Parameter param = params.get(p);
            GradTensor gradTensor = param.grad();
            if (gradTensor == null) continue;

            float[] data = param.data().data();
            float[] grad = gradTensor.data();
            float[] v = velocities[p];

            for (int i = 0; i < data.length; i++) {
                float g = grad[i];
                if (weightDecay != 0) {
                    g += weightDecay * data[i];
                }
                v[i] = momentum * v[i] - lr * g;
                data[i] += v[i];
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

    @Override
    public String toString() {
        return "SGD(lr=" + lr + ", momentum=" + momentum + ", weight_decay=" + weightDecay + ")";
    }
}
