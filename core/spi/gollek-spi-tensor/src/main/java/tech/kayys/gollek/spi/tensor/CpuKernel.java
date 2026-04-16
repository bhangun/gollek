package tech.kayys.gollek.spi.tensor;

import tech.kayys.gollek.spi.model.DeviceType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * CPU implementation of ComputeKernel using pure Java.
 * <p>
 * This serves as the guaranteed fallback when no hardware accelerators
 * are available. While slower than GPU implementations, it ensures
 * correctness and enables development/testing without GPU hardware.
 * 
 * @since 0.1.0
 */
public class CpuKernel implements ComputeKernel {

    private static final String DEVICE_NAME = detectDeviceName();

    @Override
    public DeviceType deviceType() {
        return DeviceType.CPU;
    }

    @Override
    public String deviceName() {
        return DEVICE_NAME;
    }

    @Override
    public long totalMemory() {
        return Runtime.getRuntime().maxMemory();
    }

    @Override
    public long availableMemory() {
        return Runtime.getRuntime().freeMemory();
    }

    @Override
    public boolean isAvailable() {
        return true;  // CPU is always available
    }

    // ── Memory Management ───────────────────────────────────────────────

    @Override
    public MemorySegment allocate(long bytes) {
        return MemorySegment.ofArray(new byte[(int) bytes]);
    }

    @Override
    public void free(MemorySegment ptr) {
        // No-op for CPU (GC handles it)
    }

    @Override
    public void copyHostToDevice(MemorySegment dst, MemorySegment src, long bytes) {
        dst.copyFrom(src.asSlice(0, bytes));
    }

    @Override
    public void copyDeviceToHost(MemorySegment dst, MemorySegment src, long bytes) {
        dst.copyFrom(src.asSlice(0, bytes));
    }

    @Override
    public void copyDeviceToDevice(MemorySegment dst, MemorySegment src, long bytes) {
        dst.copyFrom(src.asSlice(0, bytes));
    }

    // ── Matrix Multiplication ───────────────────────────────────────────

    @Override
    public void matmul(MemorySegment C, MemorySegment A, MemorySegment B, int M, int K, int N) {
        // Naive O(M×K×N) implementation
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, (long) m * K + k);
                    float b = B.getAtIndex(ValueLayout.JAVA_FLOAT, (long) k * N + n);
                    sum += a * b;
                }
                C.setAtIndex(ValueLayout.JAVA_FLOAT, (long) m * N + n, sum);
            }
        }
    }

    // ── Attention Computation ───────────────────────────────────────────

    @Override
    public void attention(MemorySegment output, MemorySegment query, MemorySegment key,
                         MemorySegment value, int seqLen, int numHeads, int headDim) {
        // Simplified attention implementation
        float scale = 1.0f / (float) Math.sqrt(headDim);
        int dModel = numHeads * headDim;

        for (int i = 0; i < seqLen; i++) {
            // Compute attention scores
            float[] scores = new float[seqLen];
            float maxScore = Float.NEGATIVE_INFINITY;

            for (int j = 0; j < seqLen; j++) {
                float dot = 0.0f;
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float q = query.getAtIndex(ValueLayout.JAVA_FLOAT, 
                            (long) i * dModel + h * headDim + d);
                        float k = key.getAtIndex(ValueLayout.JAVA_FLOAT, 
                            (long) j * dModel + h * headDim + d);
                        dot += q * k;
                    }
                }
                scores[j] = dot * scale;
                maxScore = Math.max(maxScore, scores[j]);
            }

            // Softmax
            float sumExp = 0.0f;
            for (int j = 0; j < seqLen; j++) {
                scores[j] = (float) Math.exp(scores[j] - maxScore);
                sumExp += scores[j];
            }
            for (int j = 0; j < seqLen; j++) {
                scores[j] /= sumExp;
            }

            // Weighted sum
            for (int h = 0; h < numHeads; h++) {
                for (int d = 0; d < headDim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seqLen; j++) {
                        float v = value.getAtIndex(ValueLayout.JAVA_FLOAT, 
                            (long) j * dModel + h * headDim + d);
                        sum += scores[j] * v;
                    }
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, 
                        (long) i * dModel + h * headDim + d, sum);
                }
            }
        }
    }

    // ── Normalization ───────────────────────────────────────────────────

    @Override
    public void rmsNorm(MemorySegment output, MemorySegment input, MemorySegment weight,
                       int hiddenSize, float eps) {
        // Compute RMS
        float sumSq = 0.0f;
        for (int i = 0; i < hiddenSize; i++) {
            float x = input.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            sumSq += x * x;
        }
        float rms = (float) Math.sqrt(sumSq / hiddenSize + eps);

        // Normalize and scale
        for (int i = 0; i < hiddenSize; i++) {
            float x = input.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float w = weight.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            output.setAtIndex(ValueLayout.JAVA_FLOAT, i, (x / rms) * w);
        }
    }

    // ── Activation Functions ────────────────────────────────────────────

    @Override
    public void silu(MemorySegment output, MemorySegment input, long size) {
        for (long i = 0; i < size; i++) {
            float x = input.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-x));
            output.setAtIndex(ValueLayout.JAVA_FLOAT, i, x * sigmoid);
        }
    }

    // ── Element-wise Operations ─────────────────────────────────────────

    @Override
    public void elementwiseAdd(MemorySegment C, MemorySegment A, MemorySegment B, long size) {
        for (long i = 0; i < size; i++) {
            float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float b = B.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            C.setAtIndex(ValueLayout.JAVA_FLOAT, i, a + b);
        }
    }

    @Override
    public void elementwiseMul(MemorySegment C, MemorySegment A, MemorySegment B, long size) {
        for (long i = 0; i < size; i++) {
            float a = A.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float b = B.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            C.setAtIndex(ValueLayout.JAVA_FLOAT, i, a * b);
        }
    }

    @Override
    public void elementwiseScale(MemorySegment A, float scale, long size) {
        for (long i = 0; i < size; i++) {
            float x = A.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            A.setAtIndex(ValueLayout.JAVA_FLOAT, i, x * scale);
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    private static String detectDeviceName() {
        String arch = System.getProperty("os.arch", "unknown");
        int cores = Runtime.getRuntime().availableProcessors();
        return String.format("CPU (%s, %d cores)", arch, cores);
    }
}
