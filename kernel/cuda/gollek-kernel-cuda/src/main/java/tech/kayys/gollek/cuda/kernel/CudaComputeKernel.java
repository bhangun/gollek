package tech.kayys.gollek.cuda.kernel;

import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.tensor.ComputeKernel;
import tech.kayys.gollek.spi.tensor.ComputeKernelRegistry;
import tech.kayys.gollek.cuda.binding.CudaBinding;

import java.lang.foreign.MemorySegment;

/**
 * Adapter that makes CudaBinding implement the unified ComputeKernel SPI.
 * <p>
 * This adapter translates between the ComputeKernel interface and the
 * existing CudaBinding implementation, handling:
 * <ul>
 *   <li>Method signature normalization (alpha/beta defaults, return codes)</li>
 *   <li>Device type enumeration mapping</li>
 *   <li>Stream type wrapping</li>
 *   <li>Error code to exception conversion</li>
 * </ul>
 * 
 * @since 0.1.0
 */
public class CudaComputeKernel implements ComputeKernel {

    private final CudaBinding binding;
    private final int deviceId;
    private volatile boolean initialized = false;

    /**
     * Creates a new CUDA compute kernel adapter.
     */
    public CudaComputeKernel() {
        this(0);  // Default device
    }

    /**
     * Creates a new CUDA compute kernel adapter for a specific device.
     *
     * @param deviceId CUDA device ID
     */
    public CudaComputeKernel(int deviceId) {
        this.binding = CudaBinding.getInstance();
        this.deviceId = deviceId;
    }

    // ── Device Information ──────────────────────────────────────────────

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public String deviceName() {
        return binding.deviceName(deviceId);
    }

    @Override
    public long totalMemory() {
        // CUDA doesn't expose total memory directly, estimate from free + typical usage
        long freeMem = binding.freeMemory();
        // Assume 80% of free memory is usable (reserve 20% for system)
        return (long) (freeMem / 0.8);
    }

    @Override
    public long availableMemory() {
        return binding.freeMemory();
    }

    @Override
    public boolean isAvailable() {
        return binding.isNativeAvailable();
    }

    // ── Memory Management ───────────────────────────────────────────────

    @Override
    public MemorySegment allocate(long bytes) {
        MemorySegment ptr = binding.malloc(bytes);
        if (ptr == null || ptr.address() == 0) {
            throw new OutOfMemoryError("CUDA malloc failed: requested " + bytes + " bytes");
        }
        return ptr;
    }

    @Override
    public MemorySegment allocateUnified(long bytes) {
        // CUDA unified memory with flags=0 (default)
        MemorySegment ptr = binding.mallocManaged(bytes, 0);
        if (ptr == null || ptr.address() == 0) {
            throw new OutOfMemoryError("CUDA unified malloc failed: requested " + bytes + " bytes");
        }
        return ptr;
    }

    @Override
    public void free(MemorySegment ptr) {
        if (ptr != null && ptr.address() != 0) {
            binding.free(ptr);
        }
    }

    @Override
    public void copyHostToDevice(MemorySegment dst, MemorySegment src, long bytes) {
        binding.memcpyH2D(dst, src, bytes);
    }

    @Override
    public void copyDeviceToHost(MemorySegment dst, MemorySegment src, long bytes) {
        binding.memcpyD2H(dst, src, bytes);
    }

    @Override
    public void copyDeviceToDevice(MemorySegment dst, MemorySegment src, long bytes) {
        // Device-to-device copy uses D2H then H2D as intermediate
        // (CUDA has cudaMemcpyDeviceToDevice but binding doesn't expose it directly)
        MemorySegment tempHost = MemorySegment.ofArray(new byte[(int) bytes]);
        binding.memcpyD2H(tempHost, src, bytes);
        binding.memcpyH2D(dst, tempHost, bytes);
    }

    // ── Matrix Multiplication ───────────────────────────────────────────

    @Override
    public void matmul(MemorySegment C, MemorySegment A, MemorySegment B, int M, int K, int N) {
        // Default alpha=1.0, beta=0.0 (C = A×B)
        int result = binding.matmul(C, A, B, M, K, N, 1.0f, 0.0f);
        if (result != 0) {
            throw new RuntimeException("CUDA matmul failed with error code: " + result);
        }
    }

    // ── Attention Computation ───────────────────────────────────────────

    @Override
    public void attention(MemorySegment output, MemorySegment query, MemorySegment key,
                         MemorySegment value, int seqLen, int numHeads, int headDim) {
        // Simplified attention without paged KV cache
        // Create dummy block table for single sequence
        int blockSize = 16;
        int numBlocks = (seqLen + blockSize - 1) / blockSize;
        int[] blockTable = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++) blockTable[i] = i;
        
        MemorySegment blockTableSeg = MemorySegment.ofArray(blockTable);
        int[] contextLens = {seqLen};
        MemorySegment contextLensSeg = MemorySegment.ofArray(contextLens);
        
        float scale = 1.0f / (float) Math.sqrt(headDim);
        
        int result = binding.attention(
            output, query, key, value,
            blockTableSeg, contextLensSeg,
            1,          // batch size
            seqLen,     // sequence length
            numHeads,   // num heads
            headDim,    // head dim
            blockSize,  // block size
            numBlocks,  // max blocks
            scale,      // scale
            0           // not causal (0=false, 1=true)
        );
        
        if (result != 0) {
            throw new RuntimeException("CUDA attention failed with error code: " + result);
        }
    }

    @Override
    public void flashAttention(MemorySegment output, MemorySegment query, MemorySegment key,
                              MemorySegment value, int seqLen, int numHeads, int headDim) {
        // Try FlashAttention-3 first (H100+), then V2 (A100+), then fallback
        int computeCap = binding.computeCapability(deviceId);
        float scale = 1.0f / (float) Math.sqrt(headDim);
        int result;
        
        if (computeCap >= 90) {
            // H100+ - use FlashAttention-3
            result = binding.flashAttnV3(
                output, query, key, value,
                1, seqLen, seqLen, numHeads, headDim,
                scale, 0, 0  // isCausal=0, useFp8=0
            );
        } else if (computeCap >= 80) {
            // A100+ - use FlashAttention-2
            result = binding.flashAttnV2(
                output, query, key, value,
                1, seqLen, seqLen, numHeads, headDim,
                scale, 0
            );
        } else {
            // Fallback to standard attention
            attention(output, query, key, value, seqLen, numHeads, headDim);
            return;
        }
        
        if (result != 0) {
            // FlashAttention failed, fallback to standard
            attention(output, query, key, value, seqLen, numHeads, headDim);
        }
    }

    // ── Normalization ───────────────────────────────────────────────────

    @Override
    public void rmsNorm(MemorySegment output, MemorySegment input, MemorySegment weight,
                       int hiddenSize, float eps) {
        int result = binding.rmsNorm(output, input, weight, hiddenSize, eps);
        if (result != 0) {
            throw new RuntimeException("CUDA rmsNorm failed with error code: " + result);
        }
    }

    // ── Activation Functions ────────────────────────────────────────────

    @Override
    public void silu(MemorySegment output, MemorySegment input, long size) {
        // CUDA doesn't have direct silu, use siluFfn with gate=up=input
        int result = binding.siluFfn(output, input, input, (int) size);
        if (result != 0) {
            throw new RuntimeException("CUDA silu failed with error code: " + result);
        }
    }

    @Override
    public void siluFfn(MemorySegment output, MemorySegment input, long size) {
        // siluFfn expects separate gate and up projections
        // For simple silu, use same input for both
        int result = binding.siluFfn(output, input, input, (int) size);
        if (result != 0) {
            throw new RuntimeException("CUDA siluFfn failed with error code: " + result);
        }
    }

    // ── Element-wise Operations ─────────────────────────────────────────

    @Override
    public void elementwiseAdd(MemorySegment C, MemorySegment A, MemorySegment B, long size) {
        // CUDA doesn't have elementwise add in binding, would need custom kernel
        // For now, throw unsupported
        throw new UnsupportedOperationException(
            "Elementwise add not available in CUDA binding. Use Metal or CPU kernel.");
    }

    @Override
    public void elementwiseMul(MemorySegment C, MemorySegment A, MemorySegment B, long size) {
        throw new UnsupportedOperationException(
            "Elementwise mul not available in CUDA binding. Use Metal or CPU kernel.");
    }

    @Override
    public void elementwiseScale(MemorySegment A, float scale, long size) {
        throw new UnsupportedOperationException(
            "Elementwise scale not available in CUDA binding. Use Metal or CPU kernel.");
    }

    // ── Stream Management ───────────────────────────────────────────────

    @Override
    public KernelStream createStream() {
        MemorySegment stream = binding.streamCreate();
        return new CudaKernelStream(stream, binding);
    }

    @Override
    public void synchronizeStream(KernelStream stream) {
        if (stream instanceof CudaKernelStream cudaStream) {
            binding.streamSynchronize(cudaStream.getHandle());
        } else {
            throw new IllegalArgumentException("Stream must be CudaKernelStream");
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    @Override
    public void initialize() {
        if (!initialized) {
            int result = binding.init(deviceId);
            if (result != 0) {
                throw new RuntimeException("CUDA initialization failed with error code: " + result);
            }
            initialized = true;
        }
    }

    @Override
    public void shutdown() {
        // CUDA runtime doesn't require explicit shutdown
        initialized = false;
    }

    // ── Nested Classes ────────────────────────────────────────────────

    /**
     * CUDA kernel stream implementation.
     */
    static class CudaKernelStream implements KernelStream {
        private final MemorySegment handle;
        private final CudaBinding binding;

        CudaKernelStream(MemorySegment handle, CudaBinding binding) {
            this.handle = handle;
            this.binding = binding;
        }

        MemorySegment getHandle() {
            return handle;
        }

        @Override
        public void close() {
            binding.streamDestroy(handle);
        }
    }
}
