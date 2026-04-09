package tech.kayys.gollek.ml.autograd;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class GradTensorTest {

    @Test
    void testBasicAutograd() {
        GradTensor x = GradTensor.scalar(2.0f).requiresGrad(true);
        GradTensor w = GradTensor.scalar(3.0f).requiresGrad(true);
        GradTensor b = GradTensor.scalar(1.0f).requiresGrad(true);

        // y = w * x + b => (3 * 2) + 1 = 7
        GradTensor y = w.mul(x).add(b);

        assertThat(y.item()).isCloseTo(7.0f, within(1e-5f));

        y.backward();

        // dy/dw = x = 2
        // dy/dx = w = 3
        // dy/db = 1
        assertThat(w.grad().item()).isCloseTo(2.0f, within(1e-5f));
        assertThat(x.grad().item()).isCloseTo(3.0f, within(1e-5f));
        assertThat(b.grad().item()).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void testMatmulAutograd() {
        // [1, 2] -> [2.0, 3.0]
        GradTensor x = GradTensor.of(new float[]{2.0f, 3.0f}, 1, 2).requiresGrad(true);
        
        // [2, 1] -> [4.0, 5.0]
        GradTensor w = GradTensor.of(new float[]{4.0f, 5.0f}, 2, 1).requiresGrad(true);

        // y = x * w  => (2*4) + (3*5) = 23.0
        GradTensor y = x.matmul(w);

        assertThat(y.data()[0]).isCloseTo(23.0f, within(1e-5f));

        y.backward();

        // grad_x = w.T => [4.0, 5.0]
        assertThat(x.grad().data()).containsExactly(4.0f, 5.0f);

        // grad_w = x.T => [2.0, 3.0]
        assertThat(w.grad().data()).containsExactly(2.0f, 3.0f);
    }
    
    @Test
    void testReLU() {
        GradTensor x = GradTensor.of(new float[]{-2.0f, 3.0f}).requiresGrad(true);
        GradTensor y = x.relu();

        assertThat(y.data()).containsExactly(0.0f, 3.0f);

        // Gradient of sum(y) with respect to x
        GradTensor sum = y.sum();
        sum.backward();

        assertThat(x.grad().data()).containsExactly(0.0f, 1.0f);
    }
}
