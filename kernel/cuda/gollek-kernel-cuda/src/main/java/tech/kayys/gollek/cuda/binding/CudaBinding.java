package tech.kayys.gollek.cuda.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM-based binding to the Gollek CUDA bridge ({@code libgollek_cuda.so}).
 *
 * <p>
 * Mirrors the pattern of {@link tech.kayys.gollek.metal.binding.MetalBinding}:
 * a singleton loaded once at startup via {@link SymbolLookup#libraryLookup},
 * falling back to a CPU implementation when the library is absent.
 *
 * <h2>CUDA Memory Model</h2>
 * <p>
 * On NVIDIA GPUs with unified memory (A100, H100), {@link MemorySegment}
 * allocated with {@link Arena#ofShared()} can be accessed by the GPU via
 * CUDA unified virtual addressing. The bridge's {@code cuda_wrap_ptr()}
 * function gives CUDA a direct pointer — <b>zero copy</b> for UMA systems.
 *
 * <h2>Lifecycle</h2>
 *
 * <pre>{@code
 * // At startup (e.g., CudaRunner.initialize()):
 * boolean loaded = CudaBinding.initialize(
 *         Path.of("/usr/local/cuda/lib64/libgollek_cuda.so"));
 * CudaBinding binding = CudaBinding.getInstance();
 *
 * // Use:
 * binding.init();
 * binding.matmul(C, A, B, M, K, N, 1.0f, 0.0f);
 * binding.attention(out, Q, K, V, bt, ctx, B, T, H, D, blockSize, maxBlocks, scale, 1);
 * }</pre>
 */
public class CudaBinding {

    private static final Logger LOG = Logger.getLogger(CudaBinding.class);
    private static volatile CudaBinding instance;

    // ── Function names ────────────────────────────────────────────────────────

    private static final String FN_INIT = "gollek_cuda_init";
    private static final String FN_FREE_MEM = "gollek_cuda_free_memory";
    private static final String FN_ALLOC = "gollek_cuda_malloc";
    private static final String FN_ALLOC_MANAGED = "gollek_cuda_malloc_managed";
    private static final String FN_FREE = "gollek_cuda_free";
    private static final String FN_MEMCPY_H2D = "gollek_cuda_memcpy_h2d";
    private static final String FN_MEMCPY_D2H = "gollek_cuda_memcpy_d2h";
    private static final String FN_MEMCPY_ASYNC = "gollek_cuda_memcpy_async";
    private static final String FN_STREAM_CREATE = "gollek_cuda_stream_create";
    private static final String FN_STREAM_DESTROY = "gollek_cuda_stream_destroy";
    private static final String FN_STREAM_SYNCHRONIZE = "gollek_cuda_stream_synchronize";
    private static final String FN_MATMUL = "gollek_cuda_matmul";
    private static final String FN_ATTENTION = "gollek_cuda_attention";
    private static final String FN_FLASH_ATTN_V2 = "gollek_cuda_flash_attn_v2";
    private static final String FN_FLASH_ATTN_V3 = "gollek_cuda_flash_attn_v3";
    private static final String FN_RMSNORM = "gollek_cuda_rmsnorm";
    private static final String FN_SILU_FFN = "gollek_cuda_silu_ffn";
    private static final String FN_DEVICE_COUNT = "gollek_cuda_device_count";
    private static final String FN_DEVICE_NAME = "gollek_cuda_device_name";
    private static final String FN_DEVICE_GET = "gollek_cuda_device_get";
    private static final String FN_COMPUTE_CAP = "gollek_cuda_compute_capability";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> handles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private CudaBinding(SymbolLookup lookup) {
        this.lookup = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable)
            bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null)
            return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new CudaBinding(lk);
            LOG.infof("CudaBinding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("CudaBinding: library not found at %s (%s) — CPU fallback active",
                    libraryPath, e.getMessage());
            instance = new CudaBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null)
            return;
        instance = new CudaBinding(null);
        LOG.info("CudaBinding: CPU fallback mode");
    }

    public static CudaBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException(
                    "CudaBinding not initialized — call initialize() first");
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Initialize the CUDA runtime and select device.
     *
     * @param deviceId CUDA device ID (0-based)
     * @return 0 on success, negative on error
     */
    public int init(int deviceId) {
        if (!nativeAvailable)
            return 0; // CPU fallback — nothing to init
        return (int) invoke(FN_INIT, deviceId);
    }

    /**
     * Get the number of CUDA devices.
     */
    public int deviceCount() {
        if (!nativeAvailable)
            return 0;
        return (int) invoke(FN_DEVICE_COUNT);
    }

    /**
     * Select the CUDA device for subsequent operations.
     */
    public int deviceGet(int deviceId) {
        if (!nativeAvailable)
            return 0;
        return (int) invoke(FN_DEVICE_GET, deviceId);
    }

    /**
     * Get device name.
     */
    public String deviceName(int deviceId) {
        if (!nativeAvailable)
            return "CPU (no CUDA)";
        try (Arena a = Arena.ofConfined()) {
            MemorySegment buf = a.allocate(256L);
            invoke(FN_DEVICE_NAME, buf, 256, deviceId);
            return buf.getString(0, java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    /**
     * Get compute capability (major * 10 + minor).
     * E.g., 80 = 8.0 (A100), 90 = 9.0 (H100)
     */
    public int computeCapability(int deviceId) {
        if (!nativeAvailable)
            return 0;
        return (int) invoke(FN_COMPUTE_CAP, deviceId);
    }

    /**
     * Get free GPU memory in bytes.
     */
    public long freeMemory() {
        if (!nativeAvailable)
            return Runtime.getRuntime().freeMemory();
        return (long) invoke(FN_FREE_MEM);
    }

    /**
     * Allocate device memory.
     *
     * @param bytes Size in bytes
     * @return MemorySegment pointing to device memory
     */
    public MemorySegment malloc(long bytes) {
        if (!nativeAvailable) {
            try (Arena a = Arena.ofConfined()) {
                return a.allocate(bytes, 64);
            }
        }
        return (MemorySegment) invoke(FN_ALLOC, bytes, 64L);
    }

    /**
     * Allocate managed (unified) memory for zero-copy on A100/H100.
     *
     * @param bytes Size in bytes
     * @param flags Attach flags (typically HIP_MEM_ATTACH_GLOBAL)
     * @return MemorySegment pointing to managed memory
     */
    public MemorySegment mallocManaged(long bytes, int flags) {
        if (!nativeAvailable) {
            return Arena.ofAuto().allocate(bytes, 64);
        }
        return (MemorySegment) invoke(FN_ALLOC_MANAGED, bytes, flags);
    }

    /**
     * Free device memory.
     */
    public void free(MemorySegment ptr) {
        if (!nativeAvailable)
            return;
        invoke(FN_FREE, ptr);
    }

    /**
     * Copy memory from host to device.
     */
    public void memcpyH2D(MemorySegment dst, MemorySegment src, long bytes) {
        if (!nativeAvailable) {
            dst.copyFrom(src);
            return;
        }
        invoke(FN_MEMCPY_H2D, dst, src, bytes);
    }

    /**
     * Copy memory from device to host.
     */
    public void memcpyD2H(MemorySegment dst, MemorySegment src, long bytes) {
        if (!nativeAvailable) {
            dst.copyFrom(src);
            return;
        }
        invoke(FN_MEMCPY_D2H, dst, src, bytes);
    }

    /**
     * Create a CUDA stream for async operations.
     */
    public MemorySegment streamCreate() {
        if (!nativeAvailable)
            return MemorySegment.NULL;
        return (MemorySegment) invoke(FN_STREAM_CREATE);
    }

    /**
     * Destroy a CUDA stream.
     */
    public void streamDestroy(MemorySegment stream) {
        if (!nativeAvailable)
            return;
        invoke(FN_STREAM_DESTROY, stream);
    }

    /**
     * Synchronize a CUDA stream.
     */
    public void streamSynchronize(MemorySegment stream) {
        if (!nativeAvailable)
            return;
        invoke(FN_STREAM_SYNCHRONIZE, stream);
    }

    /**
     * Matrix multiplication via CUDA: C = alpha * A × B + beta * C
     *
     * <p>
     * On NVIDIA GPUs this dispatches to cuBLAS or custom tensor core kernels.
     *
     * @param C     Output [M × N float32]
     * @param A     Left [M × K float32]
     * @param B     Right [K × N float32]
     * @param M,K,N Dimensions
     * @param alpha Scale factor
     * @param beta  Accumulation (0 = overwrite)
     * @return 0 on success
     */
    public int matmul(MemorySegment C, MemorySegment A, MemorySegment B,
                      int M, int K, int N, float alpha, float beta) {
        if (!nativeAvailable) {
            return CudaCpuFallback.matmul(C, A, B, M, K, N, alpha, beta);
        }
        return (int) invoke(FN_MATMUL, C, A, B, M, K, N, alpha, beta);
    }

    /**
     * Paged softmax attention via CUDA.
     *
     * <p>
     * K/V cache is accessed directly from Gollek's
     * {@link tech.kayys.gollek.kvcache.PhysicalBlockPool} off-heap slabs.
     *
     * @param out         [B, T, H, D] float32
     * @param Q           [B, T, H, D] float32
     * @param K_cache     paged K pool
     * @param V_cache     paged V pool
     * @param blockTable  [B, maxBlocks] int32
     * @param contextLens [B] int32
     * @param scale       1/sqrt(D)
     * @param isCausal    1 = causal mask
     * @return 0 on success
     */
    public int attention(MemorySegment out, MemorySegment Q,
                         MemorySegment K_cache, MemorySegment V_cache,
                         MemorySegment blockTable, MemorySegment contextLens,
                         int B, int T, int H, int D,
                         int blockSize, int maxBlocks,
                         float scale, int isCausal) {
        if (!nativeAvailable) {
            return CudaCpuFallback.attention(out, Q, K_cache, V_cache,
                    blockTable, contextLens, B, T, H, D, blockSize, maxBlocks, scale, isCausal);
        }
        return (int) invoke(FN_ATTENTION,
                out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, D, blockSize, maxBlocks, scale, isCausal);
    }

    /**
     * FlashAttention-2 kernel for A100+ (compute cap ≥ 8.0).
     *
     * @param out         Output [B, T, H, D]
     * @param Q           Query [B, T, H, D]
     * @param K           Key [B, S, H, D]
     * @param V           Value [B, S, H, D]
     * @param B           Batch size
     * @param T           Query sequence length
     * @param S           Key/Value sequence length
     * @param H           Number of heads
     * @param D           Head dimension
     * @param scale       Attention scale
     * @param isCausal    1 = causal mask
     * @return 0 on success
     */
    public int flashAttnV2(MemorySegment out, MemorySegment Q,
                           MemorySegment K, MemorySegment V,
                           int B, int T, int S, int H, int D,
                           float scale, int isCausal) {
        if (!nativeAvailable) {
            // Fall back to regular attention
            try (Arena a = Arena.ofConfined()) {
                MemorySegment fakeBt = a.allocate((long) B * 4L, 4);
                MemorySegment fakeCl = a.allocate((long) B * 4L, 4);
                for (int i = 0; i < B; i++) {
                    fakeBt.setAtIndex(ValueLayout.JAVA_INT, i, 0);
                    fakeCl.setAtIndex(ValueLayout.JAVA_INT, i, S);
                }
                return attention(out, Q, K, V, fakeBt, fakeCl,
                        B, T, H, D, 64, 1, scale, isCausal);
            }
        }
        return (int) invoke(FN_FLASH_ATTN_V2,
                out, Q, K, V, B, T, S, H, D, scale, isCausal);
    }

    /**
     * FlashAttention-3 kernel for H100+ (compute cap ≥ 9.0) with FP8 support.
     */
    public int flashAttnV3(MemorySegment out, MemorySegment Q,
                           MemorySegment K, MemorySegment V,
                           int B, int T, int S, int H, int D,
                           float scale, int isCausal, int useFp8) {
        if (!nativeAvailable) {
            return flashAttnV2(out, Q, K, V, B, T, S, H, D, scale, isCausal);
        }
        return (int) invoke(FN_FLASH_ATTN_V3,
                out, Q, K, V, B, T, S, H, D, scale, isCausal, useFp8);
    }

    /**
     * RMS normalisation: out = x / rms(x) * weight.
     *
     * @return 0 on success
     */
    public int rmsNorm(MemorySegment out, MemorySegment x,
                       MemorySegment weight, int N, float eps) {
        if (!nativeAvailable)
            return CudaCpuFallback.rmsNorm(out, x, weight, N, eps);
        return (int) invoke(FN_RMSNORM, out, x, weight, N, eps);
    }

    /**
     * SiLU-gated FFN: out = silu(gate) * up
     *
     * @return 0 on success
     */
    public int siluFfn(MemorySegment out, MemorySegment gate,
                       MemorySegment up, int N) {
        if (!nativeAvailable)
            return CudaCpuFallback.siluFfn(out, gate, up, N);
        return (int) invoke(FN_SILU_FFN, out, gate, up, N);
    }

    /**
     * Generic kernel launch via CUDA (FFM).
     */
    public void launchKernel(long functionHandle,
                            int gridX, int gridY, int gridZ,
                            int blockX, int blockY, int blockZ,
                            int sharedMem, MemorySegment stream,
                            MemorySegment[] params, Object extra) {
        if (!nativeAvailable) {
            throw new UnsupportedOperationException("CUDA launchKernel not available in CPU fallback");
        }
        // This would require a FN_LAUNCH_KERNEL binding in bindAll()
        // For now, let's just log and skip if not bound to avoid RuntimeException in invoke()
        // because we don't have the native function name yet.
        LOG.warn("CudaBinding: launchKernel requested but not yet implemented in native bridge");
    }

    /**
     * Gets maximum grid size for a dimension.
     */
    public int maxGridSize(int dim) {
        return 2147483647; // Default max for CUDA
    }

    /**
     * Gets maximum threads per block.
     */
    public int maxThreadsPerBlock() {
        return 1024; // Default max for CUDA
    }

    // ── FFM binding ───────────────────────────────────────────────────────────

    private void bindAll() {
        // int gollek_cuda_init(int deviceId)
        bind(FN_INIT, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        // int gollek_cuda_device_count()
        bind(FN_DEVICE_COUNT, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // int gollek_cuda_device_get(int deviceId)
        bind(FN_DEVICE_GET, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        // int gollek_cuda_device_name(char* buf, int bufSz, int deviceId)
        bind(FN_DEVICE_NAME, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        // int gollek_cuda_compute_capability(int deviceId)
        bind(FN_COMPUTE_CAP, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        // long gollek_cuda_free_memory()
        bind(FN_FREE_MEM, FunctionDescriptor.of(ValueLayout.JAVA_LONG));

        // void* gollek_cuda_malloc(size_t bytes, size_t align)
        bind(FN_ALLOC, FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        // void* gollek_cuda_malloc_managed(size_t bytes, int flags)
        bind(FN_ALLOC_MANAGED, FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));

        // void gollek_cuda_free(void* ptr)
        bind(FN_FREE, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // void gollek_cuda_memcpy_h2d(void* dst, void* src, size_t bytes)
        bind(FN_MEMCPY_H2D, FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        // void gollek_cuda_memcpy_d2h(void* dst, void* src, size_t bytes)
        bind(FN_MEMCPY_D2H, FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        // void* gollek_cuda_stream_create()
        bind(FN_STREAM_CREATE, FunctionDescriptor.of(ValueLayout.ADDRESS));

        // void gollek_cuda_stream_destroy(void* stream)
        bind(FN_STREAM_DESTROY, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // void gollek_cuda_stream_synchronize(void* stream)
        bind(FN_STREAM_SYNCHRONIZE, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // int gollek_cuda_matmul(C, A, B, M, K, N, alpha, beta)
        bind(FN_MATMUL, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        // int gollek_cuda_attention(out, Q, K, V, bt, ctx, B, T, H, D, bs, mb, scale, causal)
        bind(FN_ATTENTION, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        // int gollek_cuda_flash_attn_v2(out, Q, K, V, B, T, S, H, D, scale, causal)
        bind(FN_FLASH_ATTN_V2, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        // int gollek_cuda_flash_attn_v3(out, Q, K, V, B, T, S, H, D, scale, causal, fp8)
        bind(FN_FLASH_ATTN_V3, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        // int gollek_cuda_rmsnorm(out, x, weight, N, eps)
        bind(FN_RMSNORM, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        // int gollek_cuda_silu_ffn(out, gate, up, N)
        bind(FN_SILU_FFN, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> sym = lookup.find(name);
        if (sym.isPresent()) {
            handles.put(name, Linker.nativeLinker().downcallHandle(sym.get(), descriptor));
            LOG.debugf("CudaBinding: bound %s", name);
        } else {
            LOG.warnf("CudaBinding: symbol not found — %s", name);
        }
    }

    /** Reflective invoke helper — wraps Throwable as RuntimeException. */
    private Object invoke(String name, Object... args) {
        MethodHandle mh = handles.get(name);
        if (mh == null)
            throw new IllegalStateException("Unbound: " + name);
        try {
            return mh.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new RuntimeException("CudaBinding." + name + " failed", t);
        }
    }

    // ── Test helper ───────────────────────────────────────────────────────────

    static void reset() {
        instance = null;
    }
}
