package tech.kayys.gollek.runtime.tensor;

/**
 * Supported data types for tensors across all backends in the Gollek inference runtime.
 * <p>
 * This enum defines the complete set of data types supported for tensor operations,
 * covering standard floating-point, integer, and quantized formats used in modern
 * inference engines (GGML, TensorRT, ONNX Runtime, LibTorch).
 * <p>
 * <h2>Data Type Categories</h2>
 * <h3>Floating-Point Types</h3>
 * <ul>
 *   <li>{@link #FLOAT32} — IEEE 754 single-precision (32-bit), the default for most models</li>
 *   <li>{@link #FLOAT16} — IEEE 754 half-precision (16-bit), reduces memory bandwidth</li>
 *   <li>{@link #BFLOAT16} — Brain floating-point (16-bit), preserves dynamic range for ML</li>
 * </ul>
 * <h3>Integer Types</h3>
 * <ul>
 *   <li>{@link #INT8} — 8-bit signed integer, used for quantized inference</li>
 *   <li>{@link #INT4} — 4-bit packed integer, aggressive quantization</li>
 *   <li>{@link #INT32} — 32-bit signed integer, for indices and counters</li>
 *   <li>{@link #INT64} — 64-bit signed integer, for large tensors</li>
 * </ul>
 * <h3>Quantized Types</h3>
 * <ul>
 *   <li>{@link #QINT8} — Quantized 8-bit with per-tensor scale and zero-point</li>
 *   <li>{@link #QINT4} — Quantized 4-bit with per-group scale parameters</li>
 * </ul>
 * <p>
 * <h2>Quantization</h2>
 * <p>
 * Quantized types map integer values to real numbers using the formula:
 * </p>
 * <pre>
 *   real_value = (int_value - zeroPoint) × scale
 * </pre>
 * <p>
 * Quantization reduces memory footprint and can accelerate inference on hardware
 * with integer math units (e.g., NVIDIA Tensor Cores, Intel VNNI).
 * </p>
 * <p>
 * <h2>Backend Compatibility</h2>
 * <p>
 * Not all backends support all data types. For example:
 * </p>
 * <ul>
 *   <li>GGML excels at INT4/QINT4 quantization for LLMs</li>
 *   <li>LibTorch provides comprehensive FP16/BF16 support</li>
 *   <li>ONNX Runtime optimizes INT8 quantization</li>
 * </ul>
 * <p>
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Check dtype properties
 * if (dtype.isFloatingPoint()) {
 *     // Use FP-specific optimizations
 * }
 * 
 * if (dtype.isQuantized()) {
 *     // Apply dequantization before certain operations
 *     QuantParams params = tensor.quantParams();
 * }
 * 
 * // Calculate memory requirements
 * long bytes = tensor.numel() * dtype.elementBytes();
 * }</pre>
 *
 * @see Tensor
 * @see QuantParams
 * @since 1.0
 */
public enum DType {

    /**
     * IEEE 754 single-precision floating-point (32-bit).
     * <p>
     * The default data type for most neural network models, providing
     * good precision and wide hardware support.
     */
    FLOAT32(4, "float32"),

    /**
     * IEEE 754 half-precision floating-point (16-bit).
     * <p>
     * Reduces memory usage and bandwidth by 50% compared to FLOAT32.
     * Supported on NVIDIA GPUs (Tensor Cores) and modern accelerators.
     */
    FLOAT16(2, "float16"),

    /**
     * Brain floating-point (16-bit).
     * <p>
     * Similar to FLOAT16 but with 8 exponent bits (vs 5) and 7 mantissa
     * bits (vs 10). Preserves dynamic range better for deep learning
     * gradients. Used in Google TPU and NVIDIA Ampere+.
     */
    BFLOAT16(2, "bfloat16"),

    /**
     * 8-bit signed integer.
     * <p>
     * Commonly used for quantized inference where weights and activations
     * are converted from floating-point to INT8 for efficiency.
     */
    INT8(1, "int8"),

    /**
     * 4-bit packed integer.
     * <p>
     * Aggressive quantization format storing two 4-bit values per byte.
     * Used in highly compressed models (e.g., GGML Q4_K). Note: actual
     * element size is 0.5 bytes, but reported as 1 byte for alignment.
     */
    INT4(1, "int4"),

    /**
     * 32-bit signed integer.
     * <p>
     * Used for index tensors, sequence lengths, and other integer data.
     */
    INT32(4, "int32"),

    /**
     * 64-bit signed integer (long).
     * <p>
     * Used for large tensors exceeding 32-bit index range.
     */
    INT64(8, "int64"),

    /**
     * Quantized 8-bit integer with per-tensor scale and zero-point.
     * <p>
     * Maps INT8 values to floating-point using:
     * {@code real = (int - zeroPoint) × scale}
     * <p>
     * Common in post-training quantization (PTQ) workflows.
     */
    QINT8(1, "qint8"),

    /**
     * Quantized 4-bit integer with per-group scale.
     * <p>
     * Aggressive quantization using 4-bit values with group-wise
     * dequantization parameters. Used in GGML Q4_K formats.
     */
    QINT4(1, "qint4");

    /**
     * Size in bytes for a single element of this type.
     * <p>
     * For sub-byte types (INT4), this is the packed size (1 byte stores 2 elements).
     */
    private final int elementBytes;

    /**
     * Human-readable label for this data type.
     */
    private final String label;

    DType(int elementBytes, String label) {
        this.elementBytes = elementBytes;
        this.label = label;
    }

    /**
     * Returns the size in bytes for a single element of this type.
     * <p>
     * For sub-byte quantized types (e.g., INT4, QINT4), this returns the
     * packed size. INT4 uses 1 byte to store 2 elements, so this method
     * returns 1 (not 0.5).
     * </p>
     * <p>
     * <strong>Example:</strong>
     * </p>
     * <pre>
     * FLOAT32.elementBytes() → 4
     * FLOAT16.elementBytes() → 2
     * INT8.elementBytes()    → 1
     * INT4.elementBytes()    → 1  (packed: 2 elements per byte)
     * </pre>
     *
     * @return bytes per element (packed size for sub-byte types)
     */
    public int elementBytes() {
        return elementBytes;
    }

    /**
     * Returns whether this data type represents a quantized format.
     * <p>
     * Quantized types ({@link #QINT8}, {@link #QINT4}) store integer
     * values that must be dequantized using scale and zero-point
     * parameters before certain operations.
     * </p>
     *
     * @return true if this is a quantized type
     * @see #quantParams()
     */
    public boolean isQuantized() {
        return this == QINT8 || this == QINT4;
    }

    /**
     * Returns whether this data type is a floating-point format.
     * <p>
     * Floating-point types ({@link #FLOAT32}, {@link #FLOAT16}, {@link #BFLOAT16})
     * support the full range of arithmetic operations without dequantization.
     * </p>
     *
     * @return true if this is a floating-point type
     */
    public boolean isFloatingPoint() {
        return this == FLOAT32 || this == FLOAT16 || this == BFLOAT16;
    }

    @Override
    public String toString() {
        return label;
    }
}
