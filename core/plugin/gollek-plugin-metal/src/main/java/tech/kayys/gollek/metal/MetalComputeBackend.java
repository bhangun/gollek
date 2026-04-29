package tech.kayys.gollek.metal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.spi.tensor.ComputeBackend;
import tech.kayys.gollek.spi.tensor.CpuBackend;
import tech.kayys.gollek.runtime.tensor.Device;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import jakarta.enterprise.context.ApplicationScoped;

/**
 * Metal hardware-accelerated computation backend.
 *
 * <p>This adapter implements the generic {@link ComputeBackend} SPI and delegates
 * intensive Math operations to Apple's Metal Performance Shaders (MPS) via JNI/FFM
 * using {@link MetalBinding}. 
 *
 * <p>For mathematical operations not supported natively by the Metal bridge yet (like
 * exp, log, sigmoid), it gracefully falls back to the {@link CpuBackend}.
 */
@ApplicationScoped
public class MetalComputeBackend implements ComputeBackend, AdvancedMetalOps {

    private static final Logger LOG = LoggerFactory.getLogger(MetalComputeBackend.class);

    private final MetalBinding metalBinding;
    private final CpuBackend cpuFallback;
    private final boolean isNative;

    /**
     * Reusable thread-local memory buffers to prevent FFM allocation on every operations.
     * Memory is only expanded, never shrunk, during sequential inferences.
     */
    private static class ScratchBuffers implements AutoCloseable {
        private Arena arena;
        public MemorySegment memA;
        public MemorySegment memB;
        public MemorySegment memC;
        
        public ScratchBuffers(long sizeA, long sizeB, long sizeC) {
            this.arena = Arena.ofConfined();
            this.memA = arena.allocate(sizeA, 16); // 16-byte aligned
            this.memB = arena.allocate(sizeB, 16);
            this.memC = arena.allocate(sizeC, 16);
        }

        @Override
        public void close() {
            if (arena != null) {
                try {
                    arena.close();
                } catch (Exception e) {
                    LOG.error("Failed to close Memory Arena", e);
                }
            }
        }
    }

    private final ThreadLocal<ScratchBuffers> threadLocalBuffers = new ThreadLocal<>();

    public MetalComputeBackend() {
        // Attempt to load the dylib using robust discovery (standard paths, java.library.path, etc.)
        boolean loaded = MetalBinding.initialize();
        
        if (!loaded) {
            LOG.warn("Failed to initialize libgollek_metal.dylib. MetalComputeBackend will operate in full CPU fallback mode.");
            MetalBinding.initializeFallback();
        }

        this.metalBinding = MetalBinding.getInstance();
        this.metalBinding.init();
        this.isNative = metalBinding.isNativeAvailable();

        // Fallback for ops not implemented in Metal (like log, exp, pow)
        this.cpuFallback = new CpuBackend();
        
        LOG.info("Initialized MetalComputeBackend [Device: {}, Unified Memory: {}]", 
                metalBinding.deviceName(), metalBinding.isUnifiedMemory());
    }

    private ScratchBuffers ensureBuffers(long reqA, long reqB, long reqC) {
        ScratchBuffers buffers = threadLocalBuffers.get();
        if (buffers == null || buffers.memA.byteSize() < reqA || buffers.memB.byteSize() < reqB || buffers.memC.byteSize() < reqC) {
            if (buffers != null) buffers.close();
            long allocA = Math.max(reqA + (reqA / 5), 4L * 1024 * 1024);
            long allocB = Math.max(reqB + (reqB / 5), 4L * 1024 * 1024); 
            long allocC = Math.max(reqC + (reqC / 5), 4L * 1024 * 1024);
            buffers = new ScratchBuffers(allocA, allocB, allocC);
            threadLocalBuffers.set(buffers);
            LOG.debug("Reallocated Metal scratch buffers [A:{} B:{} C:{}] bytes", allocA, allocB, allocC);
        }
        return buffers;
    }

    private void copyToMem(float[] a, float[] b, MemorySegment memA, MemorySegment memB) {
        if (a != null) {
            MemorySegment heapA = MemorySegment.ofArray(a);
            MemorySegment.copy(heapA, 0, memA, 0, heapA.byteSize());
        }
        if (b != null) {
            MemorySegment heapB = MemorySegment.ofArray(b);
            MemorySegment.copy(heapB, 0, memB, 0, heapB.byteSize());
        }
    }

    private float[] copyFromMem(MemorySegment mem, int length) {
        float[] result = new float[length];
        MemorySegment heapR = MemorySegment.ofArray(result);
        MemorySegment.copy(mem, 0, heapR, 0, (long) length * 4L);
        return result;
    }

    @Override
    public float[] matmul(float[] a, long[] shapeA, float[] b, long[] shapeB) {
        int M = (int) shapeA[shapeA.length - 2];
        int K = (int) shapeA[shapeA.length - 1];
        int N = (int) shapeB[shapeB.length - 1];

        int batchA = 1, batchB = 1;
        for (int i = 0; i < shapeA.length - 2; i++) batchA *= (int) shapeA[i];
        for (int i = 0; i < shapeB.length - 2; i++) batchB *= (int) shapeB[i];
        int batch = Math.max(batchA, batchB);
        int outSize = batch * M * N;

        ScratchBuffers buffers = ensureBuffers((long)a.length * 4L, (long)b.length * 4L, (long)outSize * 4L);
        copyToMem(a, b, buffers.memA, buffers.memB);

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            long offA = (long) (bIdx % batchA) * M * K * 4L;
            long offB = (long) (bIdx % batchB) * K * N * 4L;
            long offC = (long) bIdx * M * N * 4L;

            MemorySegment sliceA = buffers.memA.asSlice(offA, (long) M * K * 4L);
            MemorySegment sliceB = buffers.memB.asSlice(offB, (long) K * N * 4L);
            MemorySegment sliceC = buffers.memC.asSlice(offC, (long) M * N * 4L);

            metalBinding.matmul(sliceC, sliceA, sliceB, M, K, N, 1.0f, 0.0f);
        }

        return copyFromMem(buffers.memC, outSize);
    }

    @Override
    public void matmul(MemorySegment a, long[] shapeA, MemorySegment b, long[] shapeB, MemorySegment out) {
        int M = (int) shapeA[shapeA.length - 2];
        int K = (int) shapeA[shapeA.length - 1];
        int N = (int) shapeB[shapeB.length - 1];

        int batchA = 1, batchB = 1;
        for (int i = 0; i < shapeA.length - 2; i++) batchA *= (int) shapeA[i];
        for (int i = 0; i < shapeB.length - 2; i++) batchB *= (int) shapeB[i];
        int batch = Math.max(batchA, batchB);

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            long offA = (long) (bIdx % batchA) * M * K * 4L;
            long offB = (long) (bIdx % batchB) * K * N * 4L;
            long offC = (long) bIdx * M * N * 4L;

            MemorySegment sliceA = a.asSlice(offA, (long) M * K * 4L);
            MemorySegment sliceB = b.asSlice(offB, (long) K * N * 4L);
            MemorySegment sliceC = out.asSlice(offC, (long) M * N * 4L);

            metalBinding.matmul(sliceC, sliceA, sliceB, M, K, N, 1.0f, 0.0f);
        }
    }

    @Override
    public float[] add(float[] a, float[] b, long[] shape) {
        int N = a.length;
        ScratchBuffers buffers = ensureBuffers((long)N * 4L, (long)N * 4L, (long)N * 4L);
        copyToMem(a, b, buffers.memA, buffers.memB);
        metalBinding.add(buffers.memC, buffers.memA, buffers.memB, N);
        return copyFromMem(buffers.memC, N);
    }

    @Override
    public void add(MemorySegment a, MemorySegment b, MemorySegment out, long[] shape) {
        long n = 1;
        for (long s : shape) n *= s;
        metalBinding.add(out, a, b, (int)n);
    }

    @Override
    public float[] sub(float[] a, float[] b, long[] shape) {
        if (!isNative) return cpuFallback.sub(a, b, shape);
        int n = a.length; ScratchBuffers bfr = ensureBuffers(n * 4L, n * 4L, n * 4L);
        copyToMem(a, b, bfr.memA, bfr.memB);
        if (metalBinding.sub(bfr.memC, bfr.memA, bfr.memB, n) != 0) return cpuFallback.sub(a, b, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] mul(float[] a, float[] b, long[] shape) {
        if (!isNative) return cpuFallback.mul(a, b, shape);
        int n = a.length; ScratchBuffers bfr = ensureBuffers(n * 4L, n * 4L, n * 4L);
        copyToMem(a, b, bfr.memA, bfr.memB);
        if (metalBinding.mul(bfr.memC, bfr.memA, bfr.memB, n) != 0) return cpuFallback.mul(a, b, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] div(float[] a, float[] b, long[] shape) {
        if (!isNative) return cpuFallback.div(a, b, shape);
        int n = a.length; ScratchBuffers bfr = ensureBuffers(n * 4L, n * 4L, n * 4L);
        copyToMem(a, b, bfr.memA, bfr.memB);
        if (metalBinding.div(bfr.memC, bfr.memA, bfr.memB, n) != 0) return cpuFallback.div(a, b, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] relu(float[] data, long[] shape) {
        if (!isNative) return cpuFallback.relu(data, shape);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, n * 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.relu(bfr.memC, bfr.memA, n) != 0) return cpuFallback.relu(data, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] sigmoid(float[] data, long[] shape) {
        if (!isNative) return cpuFallback.sigmoid(data, shape);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, n * 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.sigmoid(bfr.memC, bfr.memA, n) != 0) return cpuFallback.sigmoid(data, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] tanh(float[] data, long[] shape) {
        if (!isNative) return cpuFallback.tanh(data, shape);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, n * 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.tanh(bfr.memC, bfr.memA, n) != 0) return cpuFallback.tanh(data, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] exp(float[] data, long[] shape) {
        if (!isNative) return cpuFallback.exp(data, shape);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, n * 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.exp(bfr.memC, bfr.memA, n) != 0) return cpuFallback.exp(data, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] log(float[] data, long[] shape) {
        if (!isNative) return cpuFallback.log(data, shape);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, n * 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.log(bfr.memC, bfr.memA, n) != 0) return cpuFallback.log(data, shape);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] sum(float[] data, long[] shape) {
        if (!isNative) return cpuFallback.sum(data, shape);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.sum(bfr.memC, bfr.memA, n) != 0) return cpuFallback.sum(data, shape);
        return copyFromMem(bfr.memC, 1);
    }

    @Override
    public float[] mean(float[] data, long[] shape) {
        if (!isNative) return cpuFallback.mean(data, shape);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.mean(bfr.memC, bfr.memA, n) != 0) return cpuFallback.mean(data, shape);
        return copyFromMem(bfr.memC, 1);
    }

    @Override
    public float[] transpose2d(float[] data, long rows, long cols) {
        if (!isNative) return cpuFallback.transpose2d(data, rows, cols);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, n * 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.transpose2d(bfr.memC, bfr.memA, (int)rows, (int)cols) != 0) return cpuFallback.transpose2d(data, rows, cols);
        return copyFromMem(bfr.memC, n);
    }

    @Override
    public float[] pow(float[] data, long[] shape, float p) {
        if (!isNative) return cpuFallback.pow(data, shape, p);
        int n = data.length; ScratchBuffers bfr = ensureBuffers(n * 4L, 0, n * 4L);
        copyToMem(data, null, bfr.memA, null);
        if (metalBinding.pow(bfr.memC, bfr.memA, n, p) != 0) return cpuFallback.pow(data, shape, p);
        return copyFromMem(bfr.memC, n);
    }

    // ── Advanced Fused Operations (AdvancedMetalOps) ──────────────────────

    @Override
    public void rmsNorm(float[] out, float[] x, float[] weight, int n, float eps, boolean addOne) {
        if (!isNative) {
            // Usually falling back isn't fully 1:1 if a pure CPU node didn't cast to AdvancedMetalOps.
            // But if triggered, we simulate simple iteration.
            float ss = 0.f; for(int i=0; i<n; i++) ss += x[i]*x[i];
            float rms = (float) Math.sqrt(ss / n + eps);
            for(int i=0; i<n; i++) {
                float w = weight[i];
                if (addOne) w += 1.0f;
                out[i] = x[i] / rms * w;
            }
            return;
        }

        ScratchBuffers bfr = ensureBuffers(n * 4L, n * 4L, n * 4L);
        copyToMem(x, weight, bfr.memA, bfr.memB);
        metalBinding.rmsNorm(bfr.memC, bfr.memA, bfr.memB, n, eps, addOne);
        // In-place or output
        MemorySegment heapOut = MemorySegment.ofArray(out);
        MemorySegment.copy(bfr.memC, 0, heapOut, 0, (long) n * 4L);
    }

    @Override
    public void rmsNorm(MemorySegment out, MemorySegment x, MemorySegment weight, int n, float eps, boolean addOne) {
        if (!isNative) {
            float[] xArr = x.toArray(ValueLayout.JAVA_FLOAT);
            float[] wArr = weight.toArray(ValueLayout.JAVA_FLOAT);
            float[] oArr = new float[n];
            rmsNorm(oArr, xArr, wArr, n, eps, addOne);
            out.copyFrom(MemorySegment.ofArray(oArr));
            return;
        }
        metalBinding.rmsNorm(out, x, weight, n, eps, addOne);
    }

    @Override
    public void siluFfn(float[] out, float[] gate, float[] up, int n) {
        if (!isNative) {
            for(int i=0; i<n; i++) {
                float g = gate[i];
                out[i] = (float) ((g / (1.0f + Math.exp(-g))) * up[i]);
            }
            return;
        }

        ScratchBuffers bfr = ensureBuffers(n * 4L, n * 4L, n * 4L);
        copyToMem(gate, up, bfr.memA, bfr.memB);
        metalBinding.siluFfn(bfr.memC, bfr.memA, bfr.memB, n);
        MemorySegment heapOut = MemorySegment.ofArray(out);
        MemorySegment.copy(bfr.memC, 0, heapOut, 0, (long) n * 4L);
    }

    @Override
    public void siluFfn(MemorySegment out, MemorySegment gate, MemorySegment up, int n) {
        if (!isNative) {
            float[] gArr = gate.toArray(ValueLayout.JAVA_FLOAT);
            float[] uArr = up.toArray(ValueLayout.JAVA_FLOAT);
            float[] oArr = new float[n];
            siluFfn(oArr, gArr, uArr, n);
            out.copyFrom(MemorySegment.ofArray(oArr));
            return;
        }
        metalBinding.siluFfn(out, gate, up, n);
    }

    @Override
    public void pagedAttention(
            MemorySegment out, 
            MemorySegment q,
            MemorySegment kCache, 
            MemorySegment vCache,
            MemorySegment blockTable, 
            MemorySegment contextLens,
            int b, int t, int h, int d,
            int blockSize, int maxBlocks,
            float scale, boolean isCausal, float softCap) {
        
        if (!isNative) {
            throw new UnsupportedOperationException("CPU Fallback for Paged Attention MemorySegments is not supported directly in the Metal adapter. Caller must handle.");
        }

        metalBinding.attention(
                out, q, kCache, vCache, blockTable, contextLens,
                b, t, h, d, blockSize, maxBlocks, scale, isCausal ? 1 : 0, softCap
        );
    }

    @Override
    public String deviceName() {
        return isNative ? metalBinding.deviceName() : "CPU (Metal Fallback)";
    }

    @Override
    public int priority() {
        // Highly prioritised. Registry will pick 100 over CpuBackend's 0.
        return 100;
    }
}
