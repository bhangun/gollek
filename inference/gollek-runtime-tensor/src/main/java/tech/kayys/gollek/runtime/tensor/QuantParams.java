package tech.kayys.gollek.runtime.tensor;

/**
 * Quantization parameters for quantized tensors in the Gollek inference runtime.
 * <p>
 * This record encapsulates the scale and zero-point parameters used to dequantize
 * integer tensor values back to floating-point representation. Quantization is a
 * key optimization technique for reducing model size and accelerating inference
 * on hardware with integer math units.
 * <p>
 * <h2>Quantization Formula</h2>
 * <p>
 * The mapping from quantized integer values to real floating-point values is:
 * </p>
 * <pre>
 *   real_value = (int_value - zeroPoint) × scale
 * </pre>
 * <p>
 * Where:
 * </p>
 * <ul>
 *   <li><strong>int_value:</strong> The stored quantized integer (e.g., INT8, INT4)</li>
 *   <li><strong>zeroPoint:</strong> The integer value that maps to 0.0 (handles asymmetric distributions)</li>
 *   <li><strong>scale:</strong> The floating-point multiplier to restore magnitude</li>
 * </ul>
 * <p>
 * <h2>Quantization Strategies</h2>
 * <h3>Symmetric Quantization</h3>
 * <p>
 * Uses zero_point = 0, assuming the data distribution is centered around zero.
 * Common for weights in neural networks:
 * </p>
 * <pre>{@code
 * QuantParams params = QuantParams.symmetric(0.0039f);  // scale only
 * }</pre>
 * <h3>Asymmetric Quantization</h3>
 * <p>
 * Uses a non-zero zero_point to handle data with asymmetric distributions
 * (e.g., ReLU outputs that are all positive):
 * </p>
 * <pre>{@code
 * QuantParams params = new QuantParams(0.0039f, 128);  // scale + zero_point
 * }</pre>
 * <h2>Quantization Granularity</h2>
 * <ul>
 *   <li><strong>Per-tensor:</strong> Single scale/zero_point for entire tensor</li>
 *   <li><strong>Per-channel:</strong> Separate scale per output channel (common in convolutions)</li>
 *   <li><strong>Per-group:</strong> Scale per group of elements (e.g., per 128 weights)</li>
 * </ul>
 * <p>
 * This record represents per-tensor parameters. For finer granularity, backends
 * may store additional parameter arrays.
 * </p>
 * <p>
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Dequantize an INT8 value
 * QuantParams params = tensor.quantParams();
 * int quantizedValue = 42;
 * float realValue = (quantizedValue - params.zeroPoint()) * params.scale();
 * 
 * // Create symmetric quantization params
 * float maxAbs = 10.0f;
 * float scale = maxAbs / 127.0f;  // INT8 range
 * QuantParams symmetric = QuantParams.symmetric(scale);
 * }</pre>
 *
 * @param scale The scaling factor for dequantization. Represents the real-world
 *              value of one quantization step.
 * @param zeroPoint The integer value that maps to zero. Used for asymmetric
 *                  quantization to preserve the zero point accurately.
 * @see DType#isQuantized()
 * @see Tensor#quantParams()
 * @since 1.0
 */
public record QuantParams(float scale, int zeroPoint) {

    /**
     * Creates symmetric quantization parameters with zero_point = 0.
     * <p>
     * Symmetric quantization assumes the data distribution is centered around
     * zero, which is common for neural network weights after training.
     * </p>
     *
     * @param scale the dequantization scale factor
     * @return quantization parameters with zero_point = 0
     */
    public static QuantParams symmetric(float scale) {
        return new QuantParams(scale, 0);
    }
}
