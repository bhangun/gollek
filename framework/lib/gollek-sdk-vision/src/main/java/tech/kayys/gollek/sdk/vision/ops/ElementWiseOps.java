package tech.kayys.gollek.sdk.vision.ops;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

/**
 * Element-wise operations for vision models with broadcasting support.
 */
public class ElementWiseOps {

    /**
     * Element-wise addition with broadcasting support.
     *
     * <p>Supports same-shape addition and scalar broadcast (b has 1 element).
     *
     * @param a first tensor
     * @param b second tensor (same shape, or scalar)
     * @return a + b
     */
    public static GradTensor add(GradTensor a, GradTensor b) {
        long[] shapeA = a.shape(), shapeB = b.shape();
        if (java.util.Arrays.equals(shapeA, shapeB)) {
            float[] result = new float[a.data().length];
            VectorOps.add(a.data(), b.data(), result);
            return GradTensor.of(result, shapeA);
        }
        // Scalar broadcast
        if (b.numel() == 1) {
            float[] result = new float[a.data().length];
            VectorOps.addScalar(a.data(), b.item(), result);
            return GradTensor.of(result, shapeA);
        }
        // Delegate to GradTensor.add which handles broadcast via autograd
        return a.add(b);
    }

    /**
     * Element-wise multiplication.
     *
     * @param a first tensor
     * @param b second tensor (same shape)
     * @return a * b
     */
    public static GradTensor mul(GradTensor a, GradTensor b) {
        float[] result = new float[a.data().length];
        VectorOps.mul(a.data(), b.data(), result);
        return GradTensor.of(result, a.shape());
    }

    /**
     * Residual connection: {@code identity + residual}.
     *
     * @param identity skip connection input
     * @param residual output of residual block
     * @return identity + residual
     */
    public static GradTensor residual(GradTensor identity, GradTensor residual) {
        return add(identity, residual);
    }
}
