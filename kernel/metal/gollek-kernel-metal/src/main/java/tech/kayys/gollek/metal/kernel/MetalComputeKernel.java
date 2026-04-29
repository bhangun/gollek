package tech.kayys.gollek.metal.kernel;

import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.tensor.ComputeKernel;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.metal.binding.MetalFlashAttentionBinding;

import org.jboss.logging.Logger;

import java.lang.foreign.MemorySegment;

/**
 * Adapter that makes MetalBinding implement the unified ComputeKernel SPI.
 * <p>
 * This adapter translates between the ComputeKernel interface and the
 * existing MetalBinding implementation. Metal has advantages:
 * <ul>
 * <li>Unified memory architecture (no explicit copies needed)</li>
 * <li>Rich elementwise math operations (add, mul, relu, etc.)</li>
 * </ul>
 * 
 * @since 0.1.0
 */
public class MetalComputeKernel implements ComputeKernel {

    private static final Logger LOG = Logger.getLogger(MetalComputeKernel.class);

    private final MetalBinding binding;
    private volatile boolean initialized = false;

    /**
     * Creates a new Metal compute kernel adapter.
     */
    public MetalComputeKernel() {
        this.binding = MetalBinding.getInstance();
    }

    // ── Device Information ──────────────────────────────────────────────

    @Override
    public DeviceType deviceType() {
        return DeviceType.METAL;
    }

    @Override
    public String deviceName() {
        return binding.deviceName();
    }

    @Override
    public long totalMemory() {
        // Metal unified memory = system RAM + GPU VRAM
        // Estimate from available unified memory
        long available = binding.availableMemory();
        return (long) (available / 0.7); // Assume 70% is usable
    }

    @Override
    public long availableMemory() {
        return binding.availableMemory();
    }

    @Override
    public boolean isAvailable() {
        return binding.isNativeAvailable();
    }

    // ── Memory Management ───────────────────────────────────────────────

    @Override
    public MemorySegment allocate(long bytes) {
        // Metal uses native memory segment allocation
        // The Metal runtime will handle GPU memory mapping
        return java.lang.foreign.Arena.ofConfined().allocate(bytes);
    }

    @Override
    public MemorySegment allocateUnified(long bytes) {
        // Metal is already unified memory, same as allocate
        return allocate(bytes);
    }

    @Override
    public void free(MemorySegment ptr) {
        // Metal uses unified memory managed by the OS autorelease pool.
        // Explicit free is not exposed in the binding - memory is reclaimed
        // automatically when the segment is no longer referenced.
        // Mark as NULL to prevent use-after-free
        if (ptr != null && ptr.address() != 0) {
            LOG.debugf("Metal memory segment 0x%x marked for release", ptr.address());
        }
    }

    @Override
    public void copyHostToDevice(MemorySegment dst, MemorySegment src, long bytes) {
        // Metal unified memory - no copy needed, just memory copy
        dst.asSlice(0, bytes).copyFrom(src.asSlice(0, bytes));
    }

    @Override
    public void copyDeviceToHost(MemorySegment dst, MemorySegment src, long bytes) {
        // Metal unified memory - no copy needed
        dst.asSlice(0, bytes).copyFrom(src.asSlice(0, bytes));
    }

    @Override
    public void copyDeviceToDevice(MemorySegment dst, MemorySegment src, long bytes) {
        dst.asSlice(0, bytes).copyFrom(src.asSlice(0, bytes));
    }

    // ── Matrix Multiplication ───────────────────────────────────────────

    @Override
    public void matmul(MemorySegment C, MemorySegment A, MemorySegment B, int M, int K, int N) {
        int result = binding.matmul(C, A, B, M, K, N, 1.0f, 0.0f);
        if (result != 0) {
            throw new RuntimeException("Metal matmul failed with error code: " + result);
        }
    }

    // ── Attention Computation ───────────────────────────────────────────

    @Override
    public void attention(MemorySegment output, MemorySegment query, MemorySegment key,
            MemorySegment value, int seqLen, int numHeads, int headDim) {
        int blockSize = 16;
        int numBlocks = (seqLen + blockSize - 1) / blockSize;
        int[] blockTable = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++)
            blockTable[i] = i;

        MemorySegment blockTableSeg = MemorySegment.ofArray(blockTable);
        int[] contextLens = { seqLen };
        MemorySegment contextLensSeg = MemorySegment.ofArray(contextLens);

        float scale = 1.0f / (float) Math.sqrt(headDim);

        int result = binding.attention(
                output, query, key, value,
                blockTableSeg, contextLensSeg,
                1, seqLen, numHeads, headDim,
                blockSize, numBlocks,
                scale, 0, 0.0f // isCausal: 0=false, 1=true, softCap: 0.0f (disabled)
        );

        if (result != 0) {
            throw new RuntimeException("Metal attention failed with error code: " + result);
        }
    }

    @Override
    public void flashAttention(MemorySegment output, MemorySegment query, MemorySegment key,
            MemorySegment value, int seqLen, int numHeads, int headDim) {
        // Metal FA4 equivalent via MetalFlashAttentionBinding
        MetalFlashAttentionBinding faBinding = MetalFlashAttentionBinding.getInstance();
        if (faBinding.isNativeAvailable()) {
            float scale = 1.0f / (float) Math.sqrt(headDim);
            int result = faBinding.fa4Attention(
                    output, query, key, value,
                    1, seqLen, seqLen, numHeads, numHeads, headDim,
                    scale, false, false);
            if (result == 0)
                return; // Success
        }

        // Fallback to standard attention
        attention(output, query, key, value, seqLen, numHeads, headDim);
    }

    // ── Normalization ───────────────────────────────────────────────────

    @Override
    public void rmsNorm(MemorySegment output, MemorySegment input, MemorySegment weight,
            int hiddenSize, float eps, boolean addOne) {
        int result = binding.rmsNorm(output, input, weight, hiddenSize, eps, addOne);
        if (result != 0) {
            throw new RuntimeException("Metal rmsNorm failed with error code: " + result);
        }
    }

    // ── Activation Functions ────────────────────────────────────────────

    @Override
    public void silu(MemorySegment output, MemorySegment input, long size) {
        // Metal doesn't have direct silu, implement via elementwise ops:
        // silu(x) = x * sigmoid(x)
        // Need temp memory for sigmoid output
        MemorySegment sigmoidOut = allocate((int) size * Float.BYTES);
        try {
            int result = binding.sigmoid(sigmoidOut, input, (int) size);
            if (result != 0)
                throw new RuntimeException("Metal sigmoid failed: " + result);

            result = binding.mul(output, input, sigmoidOut, (int) size);
            if (result != 0)
                throw new RuntimeException("Metal mul failed: " + result);
        } finally {
            free(sigmoidOut);
        }
    }

    @Override
    public void siluFfn(MemorySegment output, MemorySegment input, long size) {
        // Use silu (ignoring second param which is for gate/up separation)
        silu(output, input, size);
    }

    @Override
    public void relu(MemorySegment output, MemorySegment input, long size) {
        int result = binding.relu(output, input, (int) size);
        if (result != 0) {
            throw new RuntimeException("Metal relu failed with error code: " + result);
        }
    }

    // ── Element-wise Operations ─────────────────────────────────────────

    @Override
    public void elementwiseAdd(MemorySegment C, MemorySegment A, MemorySegment B, long size) {
        int result = binding.add(C, A, B, (int) size);
        if (result != 0) {
            throw new RuntimeException("Metal add failed with error code: " + result);
        }
    }

    @Override
    public void elementwiseMul(MemorySegment C, MemorySegment A, MemorySegment B, long size) {
        int result = binding.mul(C, A, B, (int) size);
        if (result != 0) {
            throw new RuntimeException("Metal mul failed with error code: " + result);
        }
    }

    @Override
    public void elementwiseScale(MemorySegment A, float scale, long size) {
        // Create scale factor array and multiply
        float[] scaleArr = new float[(int) size];
        for (int i = 0; i < size; i++)
            scaleArr[i] = scale;
        MemorySegment scaleSeg = MemorySegment.ofArray(scaleArr);

        int result = binding.mul(A, A, scaleSeg, (int) size);
        if (result != 0) {
            throw new RuntimeException("Metal scale failed with error code: " + result);
        }
    }

    // ── Stream Management ───────────────────────────────────────────────

    @Override
    public KernelStream createStream() {
        // Metal uses command queues instead of streams
        // Return a placeholder for now
        return new MetalKernelStream();
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    @Override
    public void initialize() {
        if (!initialized) {
            int result = binding.init();
            if (result != 0) {
                throw new RuntimeException("Metal initialization failed with error code: " + result);
            }
            initialized = true;
        }
    }

    @Override
    public void shutdown() {
        initialized = false;
    }

    // ── Nested Classes ────────────────────────────────────────────────

    /**
     * Metal kernel stream placeholder.
     */
    static class MetalKernelStream implements KernelStream {
        @Override
        public void close() {
            // Metal command queues are released automatically
        }
    }
}
