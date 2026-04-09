package tech.kayys.gollek.ml.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.optim.SGD;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class LinearTest {

    @Test
    void testLinearForwardAndBackward() {
        Linear linear = new Linear(2, 1);
        
        // Setup simple weights: w=[0.5, -0.5], b=[0.1]
        float[] w = linear.namedParameters().get("weight").data().data();
        w[0] = 0.5f; w[1] = -0.5f;
        float[] b = linear.namedParameters().get("bias").data().data();
        b[0] = 0.1f;

        // input: x=[2.0, 4.0]
        GradTensor x = GradTensor.of(new float[]{2.0f, 4.0f}, 1, 2);

        // Forward pass
        // y = (2*0.5) + (4*-0.5) + 0.1 = 1.0 - 2.0 + 0.1 = -0.9
        GradTensor y = linear.forward(x);
        
        assertThat(y.shape()).containsExactly(1, 1);
        assertThat(y.item()).isCloseTo(-0.9f, within(1e-5f));

        // Backward pass
        y.backward();

        // gradient of w with respect to y is x => [2.0, 4.0]
        assertThat(linear.namedParameters().get("weight").data().grad().data()).containsExactly(2.0f, 4.0f);
        // gradient of b is 1.0
        assertThat(linear.namedParameters().get("bias").data().grad().item()).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void testOptimizationStep() {
        Linear linear = new Linear(1, 1);
        linear.namedParameters().get("weight").data().data()[0] = 2.0f;
        linear.namedParameters().get("bias").data().data()[0] = 0.0f;

        SGD optimizer = new SGD(linear.parameters(), 0.1f);

        // We want model(2.0) to be 5.0
        // Currently model(2.0) = 4.0
        GradTensor x = GradTensor.of(new float[]{2.0f}, 1, 1);
        GradTensor target = GradTensor.of(new float[]{5.0f}, 1, 1);

        for (int i = 0; i < 20; i++) {
            optimizer.zeroGrad();
            GradTensor pred = linear.forward(x);
            // MSE loss = (pred - target)^2
            GradTensor error = pred.sub(target);
            GradTensor loss = error.pow(2.0f);
            
            loss.backward();
            optimizer.step();
        }

        // After some iterations, prediction should be close to 5
        GradTensor finalPred = linear.forward(x);
        assertThat(finalPred.item()).isCloseTo(5.0f, within(0.1f));
    }
}
