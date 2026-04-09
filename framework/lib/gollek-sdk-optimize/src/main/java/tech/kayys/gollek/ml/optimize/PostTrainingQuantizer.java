package tech.kayys.gollek.ml.optimize;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Post-training INT8 quantization — reduces model size by 4× with minimal accuracy loss.
 *
 * <p>Uses symmetric min-max quantization per tensor:
 * <pre>
 *   scale     = (max - min) / 255
 *   zeroPoint = -min / scale
 *   q[i]      = clamp(round(x[i] / scale + zeroPoint), 0, 255)
 * </pre>
 *
 * <p>Dequantization reverses the mapping:
 * <pre>
 *   x[i] = (q[i] - zeroPoint) * scale
 * </pre>
 *
 * <p>Uses {@link VectorOps#max} for SIMD-accelerated min/max scan.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var quantizer = new PostTrainingQuantizer();
 * Map<String, QuantizedTensor> qModel = quantizer.quantizeModel(model.stateDict());
 * System.out.printf("Compression: %.1fx%n",
 *     quantizer.compressionRatio(model.stateDict(), qModel));
 * }</pre>
 */
public final class PostTrainingQuantizer {

    /**
     * An INT8-quantized tensor with its dequantization parameters.
     *
     * @param data      quantized values as {@code byte[]} (unsigned 0–255 stored as signed)
     * @param scale     dequantization scale factor
     * @param zeroPoint dequantization zero-point offset
     * @param shape     original tensor shape
     */
    public record QuantizedTensor(byte[] data, float scale, float zeroPoint, long[] shape) {

        /**
         * Returns the number of quantized elements.
         *
         * @return total element count
         */
        public int numel() { return data.length; }
    }

    /**
     * Quantizes a single {@link GradTensor} to INT8 using min-max calibration.
     *
     * <p>The min/max scan uses {@link VectorOps#max} for SIMD acceleration.
     *
     * @param t the tensor to quantize (must have at least one element)
     * @return {@link QuantizedTensor} with INT8 data and dequantization params
     */
    public QuantizedTensor quantize(GradTensor t) {
        float[] data = t.data();
        float max =  VectorOps.max(data);
        float min = -VectorOps.max(negate(data)); // min = -max(-x)

        float scale     = (max - min) / 255f;
        if (scale == 0f) scale = 1e-8f;          // avoid division by zero
        float zeroPoint = -min / scale;

        byte[] q = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            int v = Math.round(data[i] / scale + zeroPoint);
            q[i] = (byte) Math.max(0, Math.min(255, v));
        }
        return new QuantizedTensor(q, scale, zeroPoint, t.shape().clone());
    }

    /**
     * Dequantizes an INT8 {@link QuantizedTensor} back to a float32 {@link GradTensor}.
     *
     * <p>The reconstruction error is bounded by {@code scale / 2} per element.
     *
     * @param q the quantized tensor
     * @return dequantized {@link GradTensor} with the original shape
     */
    public GradTensor dequantize(QuantizedTensor q) {
        float[] data = new float[q.data().length];
        for (int i = 0; i < data.length; i++) {
            data[i] = ((q.data()[i] & 0xFF) - q.zeroPoint()) * q.scale();
        }
        return GradTensor.of(data, q.shape());
    }

    /**
     * Quantizes all tensors in a model state dict.
     *
     * @param stateDict map of parameter name → {@link GradTensor}
     * @return map of parameter name → {@link QuantizedTensor}
     */
    public Map<String, QuantizedTensor> quantizeModel(Map<String, GradTensor> stateDict) {
        Map<String, QuantizedTensor> result = new LinkedHashMap<>();
        stateDict.forEach((name, tensor) -> result.put(name, quantize(tensor)));
        return result;
    }

    /**
     * Returns the theoretical compression ratio of INT8 vs float32.
     *
     * <p>Always returns {@code 4.0} since float32 (4 bytes) → INT8 (1 byte).
     *
     * @param original  original float32 state dict (unused, for API symmetry)
     * @param quantized quantized state dict (unused, for API symmetry)
     * @return {@code 4.0f}
     */
    public float compressionRatio(Map<String, GradTensor> original,
                                   Map<String, QuantizedTensor> quantized) {
        return 4.0f; // float32 (4 bytes) → int8 (1 byte)
    }

    /** Element-wise negation for min computation. */
    private static float[] negate(float[] a) {
        float[] neg = new float[a.length];
        VectorOps.mulScalar(a, -1f, neg);
        return neg;
    }
}
