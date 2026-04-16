package tech.kayys.gollek.spi.tensor;

import java.lang.foreign.MemorySegment;


/**
 * Service Provider Interface for hardware-accelerated tensor computation.
 *
 * <p>
 * Implementations of this interface bridge the SDK's autograd engine to
 * hardware-specific kernel plugins (CUDA, Metal, ROCm, etc.).
 * The default implementation ({@link CpuBackend}) runs all operations
 * in pure Java on the CPU.
 *
 * <p>
 * Custom backends are discovered via {@link java.util.ServiceLoader}.
 * Register implementations in
 * {@code META-INF/services/tech.kayys.gollek.spi.tensor.ComputeBackend}.
 *
 * @see CpuBackend
 * @see ComputeBackendRegistry
 */
public interface ComputeBackend {

    /**
     * Matrix multiplication: C = A @ B.
     *
     * @param a      flat data for matrix A in row-major order
     * @param shapeA shape of A (e.g., [m, k])
     * @param b      flat data for matrix B in row-major order
     * @param shapeB shape of B (e.g., [k, n])
     * @return flat result array in row-major order with shape [m, n]
     */
    float[] matmul(float[] a, long[] shapeA, float[] b, long[] shapeB);

    /** Zero-copy MemorySegment matmul. */
    default void matmul(MemorySegment a, long[] shapeA, MemorySegment b, long[] shapeB, MemorySegment out) {
        throw new UnsupportedOperationException("MemorySegment matmul not implemented by " + getClass().getName());
    }

    /**
     * Element-wise addition: C = A + B.
     *
     * @param a     flat data for tensor A
     * @param b     flat data for tensor B
     * @param shape shared shape of A and B
     * @return element-wise sum
     */
    float[] add(float[] a, float[] b, long[] shape);

    /** Zero-copy MemorySegment addition. */
    default void add(MemorySegment a, MemorySegment b, MemorySegment out, long[] shape) {
        throw new UnsupportedOperationException("MemorySegment add not implemented by " + getClass().getName());
    }

    /**
     * Element-wise subtraction: C = A - B.
     */
    float[] sub(float[] a, float[] b, long[] shape);

    /** Zero-copy MemorySegment subtraction. */
    default void sub(MemorySegment a, MemorySegment b, MemorySegment out, long[] shape) {
        throw new UnsupportedOperationException("MemorySegment sub not implemented by " + getClass().getName());
    }

    /**
     * Element-wise multiplication (Hadamard product): C = A ⊙ B.
     */
    float[] mul(float[] a, float[] b, long[] shape);

    /** Zero-copy MemorySegment multiplication. */
    default void mul(MemorySegment a, MemorySegment b, MemorySegment out, long[] shape) {
        throw new UnsupportedOperationException("MemorySegment mul not implemented by " + getClass().getName());
    }

    /**
     * Element-wise division: C = A / B.
     */
    float[] div(float[] a, float[] b, long[] shape);

    /** Zero-copy MemorySegment division. */
    default void div(MemorySegment a, MemorySegment b, MemorySegment out, long[] shape) {
        throw new UnsupportedOperationException("MemorySegment div not implemented by " + getClass().getName());
    }

    /**
     * ReLU activation: max(0, x) element-wise.
     */
    float[] relu(float[] data, long[] shape);

    /** Zero-copy MemorySegment ReLU. */
    default void relu(MemorySegment data, MemorySegment out, long[] shape) {
        throw new UnsupportedOperationException("MemorySegment relu not implemented by " + getClass().getName());
    }

    /**
     * Sigmoid activation: 1 / (1 + exp(-x)) element-wise.
     */
    float[] sigmoid(float[] data, long[] shape);

    /**
     * Tanh activation element-wise.
     */
    float[] tanh(float[] data, long[] shape);

    /**
     * Exponential: exp(x) element-wise.
     */
    float[] exp(float[] data, long[] shape);

    /**
     * Natural logarithm: ln(x) element-wise.
     */
    float[] log(float[] data, long[] shape);

    /**
     * Sum all elements to a single scalar.
     *
     * @return array of length 1 containing the sum
     */
    float[] sum(float[] data, long[] shape);

    /** Zero-copy MemorySegment sum. */
    default void sum(MemorySegment data, MemorySegment out, long[] shape) {
        throw new UnsupportedOperationException("MemorySegment sum not implemented by " + getClass().getName());
    }

    /**
     * Mean of all elements.
     *
     * @return array of length 1 containing the mean
     */
    float[] mean(float[] data, long[] shape);

    /**
     * Transpose a 2D matrix.
     */
    float[] transpose2d(float[] data, long rows, long cols);

    /**
     * Power: x^p element-wise.
     */
    float[] pow(float[] data, long[] shape, float p);

    /**
     * Returns the device this backend targets.
     */
    String deviceName();

    /**
     * Priority for backend selection. Higher = preferred.
     * CPU backend returns 0; GPU backends should return higher values.
     */
    default int priority() {
        return 0;
    }
}
