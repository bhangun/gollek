package tech.kayys.gollek.cuda;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.cuda.binding.CudaBinding;
import tech.kayys.gollek.spi.tensor.ComputeBackend;
import tech.kayys.gollek.spi.tensor.CpuBackend;

import java.lang.foreign.MemorySegment;

/**
 * CUDA hardware-accelerated computation backend.
 *
 * Provides batched N-Dimensional Matrix Multiplications natively bypassing array
 * fragmentation via Foreign Function offset iterations over the unified CUDA Binding.
 */
public class CudaComputeBackend implements ComputeBackend {

    private static final Logger LOG = LoggerFactory.getLogger(CudaComputeBackend.class);

    private final CudaBinding cudaBinding;
    private final CpuBackend cpuFallback;
    private final boolean isNative;

    private static class ScratchBuffers implements AutoCloseable {
        public MemorySegment memA;
        public MemorySegment memB;
        public MemorySegment memC;
        private final CudaBinding binding;

        public ScratchBuffers(CudaBinding binding, long sizeA, long sizeB, long sizeC) {
            this.binding = binding;
            this.memA = binding.malloc(sizeA);
            this.memB = binding.malloc(sizeB);
            this.memC = binding.malloc(sizeC);
        }

        @Override
        public void close() {
            if (binding != null) {
                binding.free(memA);
                binding.free(memB);
                binding.free(memC);
            }
        }
    }

    private final ThreadLocal<ScratchBuffers> threadLocalBuffers = new ThreadLocal<>();

    public CudaComputeBackend() {
        CudaBinding initBinding = null;
        try {
            initBinding = CudaBinding.getInstance();
        } catch (IllegalStateException e) {
            LOG.debug("CudaBinding not initialized upstream, operating strictly in CPU fallback.");
        }
        this.cudaBinding = initBinding;
        this.isNative = (cudaBinding != null) && cudaBinding.isNativeAvailable();
        this.cpuFallback = new CpuBackend();
    }

    private ScratchBuffers ensureBuffers(long reqA, long reqB, long reqC) {
        ScratchBuffers buffers = threadLocalBuffers.get();
        if (buffers == null || buffers.memA.byteSize() < reqA || buffers.memB.byteSize() < reqB || buffers.memC.byteSize() < reqC) {
            if (buffers != null) buffers.close();
            long allocA = Math.max(reqA + (reqA / 5), 4L * 1024 * 1024);
            long allocB = Math.max(reqB + (reqB / 5), 4L * 1024 * 1024);
            long allocC = Math.max(reqC + (reqC / 5), 4L * 1024 * 1024);
            buffers = new ScratchBuffers(cudaBinding, allocA, allocB, allocC);
            threadLocalBuffers.set(buffers);
            LOG.debug("Reallocated CUDA scratch buffers [A:{} B:{} C:{}] bytes", allocA, allocB, allocC);
        }
        return buffers;
    }

    private void copyToMem(float[] a, float[] b, MemorySegment memA, MemorySegment memB) {
        if (a != null) {
            cudaBinding.memcpyH2D(memA, MemorySegment.ofArray(a), (long) a.length * 4L);
        }
        if (b != null) {
            cudaBinding.memcpyH2D(memB, MemorySegment.ofArray(b), (long) b.length * 4L);
        }
    }

    private float[] copyFromMem(MemorySegment mem, int length) {
        float[] result = new float[length];
        cudaBinding.memcpyD2H(MemorySegment.ofArray(result), mem, (long) length * 4L);
        return result;
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

        long bytesA = (long) a.length * 4L;
        long bytesB = (long) b.length * 4L;
        long bytesC = (long) batch * m * n * 4L;

        ScratchBuffers buffers = ensureBuffers(bytesA, bytesB, bytesC);
        copyToMem(a, b, buffers.memA, buffers.memB);

        long offsetA = m * k * 4L;
        long offsetB = k * n * 4L;
        long offsetC = m * n * 4L;

        boolean fallback = false;
        
        // Parallel batched slicing leveraging CudaBinding JNI pointers natively!
        for (int bIdx = 0; bIdx < batch; bIdx++) {
            long offA = (bIdx % batchA) * offsetA;
            long offB = (bIdx % batchB) * offsetB;
            long offC = bIdx * offsetC;

            if (cudaBinding.matmul(
                    buffers.memC.asSlice(offC, offsetC),
                    buffers.memA.asSlice(offA, offsetA),
                    buffers.memB.asSlice(offB, offsetB),
                    m, k, n, 1.0f, 0.0f) != 0) {
                fallback = true;
                break;
            }
        }

        if (fallback) {
            return cpuFallback.matmul(a, shapeA, b, shapeB);
        }
        return copyFromMem(buffers.memC, batch * m * n);
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
        return isNative ? cudaBinding.deviceName(0) : "CPU (CUDA Fallback)";
    }

    @Override
    public int priority() {
        return 100; // Prefer Hardware Accelerators
    }
}
