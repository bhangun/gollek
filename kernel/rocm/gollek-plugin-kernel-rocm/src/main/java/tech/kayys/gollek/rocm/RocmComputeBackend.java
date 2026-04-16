package tech.kayys.gollek.rocm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
// import tech.kayys.gollek.rocm.binding.RocmBinding; // TODO: Import native binding when built downstream
import tech.kayys.gollek.spi.tensor.ComputeBackend;
import tech.kayys.gollek.spi.tensor.CpuBackend;

import java.lang.foreign.MemorySegment;

/**
 * ROCm hardware-accelerated computation backend.
 *
 * Provides batched N-Dimensional Matrix Multiplications natively bypassing array
 * fragmentation via Foreign Function offset iterations over the unified ROCm Binding.
 * 
 * Note: Scaffolding deployed to mirror batched Cuda/Metal bindings. Requires 
 * native libgollek_rocm.so JNI linkage upstream.
 */
public class RocmComputeBackend implements ComputeBackend {

    private static final Logger LOG = LoggerFactory.getLogger(RocmComputeBackend.class);
    private final CpuBackend cpuFallback;

    // TODO: Define binding and arenas identically to Metal/CUDA
    private final boolean isNative = false;

    public RocmComputeBackend() {
        this.cpuFallback = new CpuBackend();
    }

    @Override
    public float[] matmul(float[] a, long[] shapeA, float[] b, long[] shapeB) {
        if (!isNative) return cpuFallback.matmul(a, shapeA, b, shapeB);

        int m = (int) shapeA[shapeA.length - 2];
        int k = (int) shapeA[shapeA.length - 1];
        int n = (int) shapeB[shapeB.length - 1];

        int batchA = 1, batchB = 1;
        for (int i = 0; i < shapeA.length - 2; i++) batchA *= (int) shapeA[i];
        for (int i = 0; i < shapeB.length - 2; i++) batchB *= (int) shapeB[i];
        int batch = Math.max(batchA, batchB);

        long offsetA = m * k * 4L;
        long offsetB = k * n * 4L;
        long offsetC = m * n * 4L;

        // --- PLATFORM KERNEL FFM SLICING BLOCK --- //
        /*
        ScratchBuffers buffers = ensureBuffers(bytesA, bytesB, bytesC);
        copyToMem(a, b, buffers.memA, buffers.memB);

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            long offA = (bIdx % batchA) * offsetA;
            long offB = (bIdx % batchB) * offsetB;
            long offC = bIdx * offsetC;

            // TODO: Execute FFI / HIP routines asynchronously
            rocmBinding.matmul(
                    buffers.memC.asSlice(offC, offsetC),
                    buffers.memA.asSlice(offA, offsetA),
                    buffers.memB.asSlice(offB, offsetB),
                    m, k, n, 1.0f, 0.0f);
        }
        */
        
        return cpuFallback.matmul(a, shapeA, b, shapeB);
    }

    @Override public float[] add(float[] a, float[] b, long[] shape) { return cpuFallback.add(a, b, shape); }
    @Override public float[] sub(float[] a, float[] b, long[] shape) { return cpuFallback.sub(a, b, shape); }
    @Override public float[] mul(float[] a, float[] b, long[] shape) { return cpuFallback.mul(a, b, shape); }
    @Override public float[] div(float[] a, float[] b, long[] shape) { return cpuFallback.div(a, b, shape); }
    @Override public float[] relu(float[] data, long[] shape) { return cpuFallback.relu(data, shape); }
    @Override public float[] sigmoid(float[] data, long[] shape) { return cpuFallback.sigmoid(data, shape); }
    @Override public float[] tanh(float[] data, long[] shape) { return cpuFallback.tanh(data, shape); }
    @Override public float[] exp(float[] data, long[] shape) { return cpuFallback.exp(data, shape); }
    @Override public float[] log(float[] data, long[] shape) { return cpuFallback.log(data, shape); }
    @Override public float[] sum(float[] data, long[] shape) { return cpuFallback.sum(data, shape); }
    @Override public float[] mean(float[] data, long[] shape) { return cpuFallback.mean(data, shape); }
    @Override public float[] transpose2d(float[] data, long rows, long cols) { return cpuFallback.transpose2d(data, rows, cols); }
    @Override public float[] pow(float[] data, long[] shape, float p) { return cpuFallback.pow(data, shape, p); }

    @Override
    public String deviceName() {
        return "CPU (ROCm Fallback)";
    }

    @Override
    public int priority() {
        return 90; 
    }
}
