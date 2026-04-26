package tech.kayys.gollek.blackwell.binding;

import org.jboss.logging.Logger;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM-based binding to the Gollek Blackwell CUDA bridge ({@code libgollek_blackwell.so}).
 *
 * <p>
 * Optimized for NVIDIA Blackwell architecture (B100, B200, GB200) with:
 * <ul>
 *   <li>FlashAttention-3 with TMEM (Tensor Memory) accumulators</li>
 *   <li>FP4 tensor core acceleration</li>
 *   <li>Async execution engines for concurrent copy/compute</li>
 *   <li>192 GB HBM3e unified memory support</li>
 * </ul>
 *
 * <h2>Blackwell TMEM Architecture</h2>
 * <p>
 * Blackwell introduces TMEM - a 64 MB on-chip tensor memory accumulator that
 * enables FlashAttention-3 to achieve 2x throughput over FlashAttention-2.
 * The TMEM acts as a high-bandwidth scratchpad for QK^T and attention output
 * accumulation without going through HBM.
 *
 * <h2>Lifecycle</h2>
 *
 * <pre>{@code
 * // At startup (e.g., BlackwellRunner.initialize()):
 * boolean loaded = BlackwellBinding.initialize(
 *         Path.of("/usr/local/cuda/lib64/libgollek_blackwell.so"));
 * BlackwellBinding binding = BlackwellBinding.getInstance();
 *
 * // Use:
 * binding.init(0);
 * binding.flashAttn3(out, Q, K, V, B, T, S, H, D, scale, 1, 1); // FP4 mode
 * }</pre>
 */
public class BlackwellBinding {

    private static final Logger LOG = Logger.getLogger(BlackwellBinding.class);
    private static volatile BlackwellBinding instance;

    // ── Function names ────────────────────────────────────────────────────────

    private static final String FN_INIT = "gollek_blackwell_init";
    private static final String FN_FREE_MEM = "gollek_blackwell_free_memory";
    private static final String FN_ALLOC = "gollek_blackwell_malloc";
    private static final String FN_ALLOC_MANAGED = "gollek_blackwell_malloc_managed";
    private static final String FN_ALLOC_TMEM = "gollek_blackwell_tmem_alloc";
    private static final String FN_FREE = "gollek_blackwell_free";
    private static final String FN_MEMCPY_H2D = "gollek_blackwell_memcpy_h2d";
    private static final String FN_MEMCPY_D2H = "gollek_blackwell_memcpy_d2h";
    private static final String FN_MEMCPY_ASYNC = "gollek_blackwell_memcpy_async";
    private static final String FN_STREAM_CREATE = "gollek_blackwell_stream_create";
    private static final String FN_STREAM_DESTROY = "gollek_blackwell_stream_destroy";
    private static final String FN_STREAM_SYNCHRONIZE = "gollek_blackwell_stream_synchronize";
    private static final String FN_STREAM_WAIT_VALUE = "gollek_blackwell_stream_wait_value";
    private static final String FN_STREAM_WRITE_VALUE = "gollek_blackwell_stream_write_value";
    private static final String FN_MATMUL = "gollek_blackwell_matmul";
    private static final String FN_MATMUL_FP8 = "gollek_blackwell_matmul_fp8";
    private static final String FN_MATMUL_FP4 = "gollek_blackwell_matmul_fp4";
    private static final String FN_ATTENTION = "gollek_blackwell_attention";
    private static final String FN_FLASH_ATTN_V3 = "gollek_blackwell_flash_attn_v3";
    private static final String FN_FLASH_ATTN_V3_TMEM = "gollek_blackwell_flash_attn_v3_tmem";
    private static final String FN_RMSNORM = "gollek_blackwell_rmsnorm";
    private static final String FN_SILU_FFN = "gollek_blackwell_silu_ffn";
    private static final String FN_DEVICE_COUNT = "gollek_blackwell_device_count";
    private static final String FN_DEVICE_NAME = "gollek_blackwell_device_name";
    private static final String FN_DEVICE_GET = "gollek_blackwell_device_get";
    private static final String FN_COMPUTE_CAP = "gollek_blackwell_compute_capability";
    private static final String FN_TMEM_SIZE = "gollek_blackwell_tmem_size";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> handles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private BlackwellBinding(SymbolLookup lookup) {
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
            instance = new BlackwellBinding(lk);
            LOG.infof("BlackwellBinding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("BlackwellBinding: library not found at %s (%s) — CPU fallback active",
                    libraryPath, e.getMessage());
            instance = new BlackwellBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null)
            return;
        instance = new BlackwellBinding(null);
        LOG.info("BlackwellBinding: CPU fallback mode");
    }

    public static BlackwellBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException(
                    "BlackwellBinding not initialized — call initialize() first");
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Initialize the Blackwell CUDA runtime and select device.
     *
     * @param deviceId CUDA device ID (0-based)
     * @return 0 on success, negative on error
     */
    public int init(int deviceId) {
        if (!nativeAvailable)
            return 0;
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
            return "CPU (no Blackwell)";
        try (Arena a = Arena.ofConfined()) {
            MemorySegment buf = a.allocate(256L);
            invoke(FN_DEVICE_NAME, buf, 256, deviceId);
            return buf.getString(0, java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    /**
     * Get compute capability (major * 10 + minor).
     * E.g., 100 = 10.0 (Blackwell B100/B200)
     */
    public int computeCapability(int deviceId) {
        if (!nativeAvailable)
            return 0;
        return (int) invoke(FN_COMPUTE_CAP, deviceId);
    }

    /**
     * Get TMEM size in bytes (Blackwell feature).
     */
    public long tmemSize(int deviceId) {
        if (!nativeAvailable)
            return 0;
        return (long) invoke(FN_TMEM_SIZE, deviceId);
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
     * Allocate managed (unified) memory for zero-copy on B100/B200.
     */
    public MemorySegment mallocManaged(long bytes, int flags) {
        if (!nativeAvailable) {
            return Arena.ofAuto().allocate(bytes, 64);
        }
        return (MemorySegment) invoke(FN_ALLOC_MANAGED, bytes, flags);
    }

    /**
     * Allocate TMEM (Tensor Memory) for FlashAttention-3.
     * Blackwell-only feature for on-chip accumulator.
     *
     * @param bytes Size in bytes (typically 64MB max)
     * @return MemorySegment pointing to TMEM
     */
    public MemorySegment tmemAlloc(long bytes) {
        if (!nativeAvailable) {
            return Arena.ofConfined().allocate(bytes, 64);
        }
        return (MemorySegment) invoke(FN_ALLOC_TMEM, bytes);
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
     * Async memory copy (concurrent with compute on Blackwell).
     */
    public void memcpyAsync(MemorySegment dst, MemorySegment src, long bytes, MemorySegment stream) {
        if (!nativeAvailable) {
            dst.copyFrom(src);
            return;
        }
        invoke(FN_MEMCPY_ASYNC, dst, src, bytes, stream);
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
     * Wait for a semaphore value on a stream (Blackwell async feature).
     */
    public void streamWaitValue(MemorySegment stream, MemorySegment addr, int value) {
        if (!nativeAvailable)
            return;
        invoke(FN_STREAM_WAIT_VALUE, stream, addr, value);
    }

    /**
     * Write a semaphore value (Blackwell async feature).
     */
    public void streamWriteValue(MemorySegment addr, int value) {
        if (!nativeAvailable)
            return;
        invoke(FN_STREAM_WRITE_VALUE, addr, value);
    }

    /**
     * Matrix multiplication via Blackwell tensor cores.
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
            return BlackwellCpuFallback.matmul(C, A, B, M, K, N, alpha, beta);
        }
        return (int) invoke(FN_MATMUL, C, A, B, M, K, N, alpha, beta);
    }

    /**
     * FP8 matrix multiplication for H100+/Blackwell.
     */
    public int matmulFp8(MemorySegment C, MemorySegment A, MemorySegment B,
                         int M, int K, int N, float alpha, float beta) {
        if (!nativeAvailable) {
            return matmul(C, A, B, M, K, N, alpha, beta);
        }
        return (int) invoke(FN_MATMUL_FP8, C, A, B, M, K, N, alpha, beta);
    }

    /**
     * FP4 matrix multiplication for Blackwell (2x FP8 throughput).
     */
    public int matmulFp4(MemorySegment C, MemorySegment A, MemorySegment B,
                         int M, int K, int N, float alpha, float beta) {
        if (!nativeAvailable) {
            return matmul(C, A, B, M, K, N, alpha, beta);
        }
        return (int) invoke(FN_MATMUL_FP4, C, A, B, M, K, N, alpha, beta);
    }

    /**
     * Paged softmax attention via Blackwell CUDA.
     */
    public int attention(MemorySegment out, MemorySegment Q,
                         MemorySegment K_cache, MemorySegment V_cache,
                         MemorySegment blockTable, MemorySegment contextLens,
                         int B, int T, int H, int D,
                         int blockSize, int maxBlocks,
                         float scale, int isCausal) {
        if (!nativeAvailable) {
            try (Arena a = Arena.ofConfined()) {
                MemorySegment fakeBt = a.allocate((long) B * 4L, 4);
                MemorySegment fakeCl = a.allocate((long) B * 4L, 4);
                for (int i = 0; i < B; i++) {
                    fakeBt.setAtIndex(ValueLayout.JAVA_INT, i, 0);
                    fakeCl.setAtIndex(ValueLayout.JAVA_INT, i, T);
                }
                return BlackwellCpuFallback.flashAttn3(out, Q, K_cache, V_cache,
                        B, T, T, H, D, scale, isCausal, 0);
            }
        }
        return (int) invoke(FN_ATTENTION,
                out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, D, blockSize, maxBlocks, scale, isCausal);
    }

    /**
     * FlashAttention-3 kernel for Blackwell with TMEM acceleration.
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
     * @param useFp4      1 = use FP4 tensor cores
     * @return 0 on success
     */
    public int flashAttnV3(MemorySegment out, MemorySegment Q,
                           MemorySegment K, MemorySegment V,
                           int B, int T, int S, int H, int D,
                           float scale, int isCausal, int useFp4) {
        if (!nativeAvailable) {
            return BlackwellCpuFallback.flashAttn3(out, Q, K, V, B, T, S, H, D, scale, isCausal, useFp4);
        }
        return (int) invoke(FN_FLASH_ATTN_V3,
                out, Q, K, V, B, T, S, H, D, scale, isCausal, useFp4);
    }

    /**
     * FlashAttention-3 with explicit TMEM management.
     * Uses Blackwell's 64MB TMEM for QK^T accumulation.
     */
    public int flashAttnV3Tmem(MemorySegment out, MemorySegment Q,
                                MemorySegment K, MemorySegment V,
                                MemorySegment tmem,
                                int B, int T, int S, int H, int D,
                                float scale, int isCausal, int useFp4) {
        if (!nativeAvailable) {
            return flashAttnV3(out, Q, K, V, B, T, S, H, D, scale, isCausal, useFp4);
        }
        return (int) invoke(FN_FLASH_ATTN_V3_TMEM,
                out, Q, K, V, tmem, B, T, S, H, D, scale, isCausal, useFp4);
    }

    /**
     * RMS normalisation: out = x / rms(x) * weight.
     */
    public int rmsNorm(MemorySegment out, MemorySegment x,
                       MemorySegment weight, int N, float eps) {
        if (!nativeAvailable)
            return BlackwellCpuFallback.rmsNorm(out, x, weight, N, eps);
        return (int) invoke(FN_RMSNORM, out, x, weight, N, eps);
    }

    /**
     * SiLU-gated FFN: out = silu(gate) * up
     */
    public int siluFfn(MemorySegment out, MemorySegment gate,
                       MemorySegment up, int N) {
        if (!nativeAvailable)
            return BlackwellCpuFallback.siluFfn(out, gate, up, N);
        return (int) invoke(FN_SILU_FFN, out, gate, up, N);
    }

    // ── FFM binding ───────────────────────────────────────────────────────────

    private void bindAll() {
        // int gollek_blackwell_init(int deviceId)
        bind(FN_INIT, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        // int gollek_blackwell_device_count()
        bind(FN_DEVICE_COUNT, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // int gollek_blackwell_device_get(int deviceId)
        bind(FN_DEVICE_GET, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        // int gollek_blackwell_device_name(char* buf, int bufSz, int deviceId)
        bind(FN_DEVICE_NAME, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        // int gollek_blackwell_compute_capability(int deviceId)
        bind(FN_COMPUTE_CAP, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT));

        // long gollek_blackwell_tmem_size(int deviceId)
        bind(FN_TMEM_SIZE, FunctionDescriptor.of(ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_INT));

        // long gollek_blackwell_free_memory()
        bind(FN_FREE_MEM, FunctionDescriptor.of(ValueLayout.JAVA_LONG));

        // void* gollek_blackwell_malloc(size_t bytes, size_t align)
        bind(FN_ALLOC, FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        // void* gollek_blackwell_malloc_managed(size_t bytes, int flags)
        bind(FN_ALLOC_MANAGED, FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));

        // void* gollek_blackwell_tmem_alloc(size_t bytes)
        bind(FN_ALLOC_TMEM, FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG));

        // void gollek_blackwell_free(void* ptr)
        bind(FN_FREE, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // void gollek_blackwell_memcpy_h2d(void* dst, void* src, size_t bytes)
        bind(FN_MEMCPY_H2D, FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        // void gollek_blackwell_memcpy_d2h(void* dst, void* src, size_t bytes)
        bind(FN_MEMCPY_D2H, FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));

        // void gollek_blackwell_memcpy_async(void* dst, void* src, size_t bytes, void* stream)
        bind(FN_MEMCPY_ASYNC, FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

        // void* gollek_blackwell_stream_create()
        bind(FN_STREAM_CREATE, FunctionDescriptor.of(ValueLayout.ADDRESS));

        // void gollek_blackwell_stream_destroy(void* stream)
        bind(FN_STREAM_DESTROY, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // void gollek_blackwell_stream_synchronize(void* stream)
        bind(FN_STREAM_SYNCHRONIZE, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

        // void gollek_blackwell_stream_wait_value(void* stream, void* addr, int value)
        bind(FN_STREAM_WAIT_VALUE, FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        // void gollek_blackwell_stream_write_value(void* addr, int value)
        bind(FN_STREAM_WRITE_VALUE, FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        // int gollek_blackwell_matmul(C, A, B, M, K, N, alpha, beta)
        bind(FN_MATMUL, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        // int gollek_blackwell_matmul_fp8(C, A, B, M, K, N, alpha, beta)
        bind(FN_MATMUL_FP8, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        // int gollek_blackwell_matmul_fp4(C, A, B, M, K, N, alpha, beta)
        bind(FN_MATMUL_FP4, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        // int gollek_blackwell_attention(out, Q, K, V, bt, ctx, B, T, H, D, bs, mb, scale, causal)
        bind(FN_ATTENTION, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        // int gollek_blackwell_flash_attn_v3(out, Q, K, V, B, T, S, H, D, scale, causal, fp4)
        bind(FN_FLASH_ATTN_V3, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        // int gollek_blackwell_flash_attn_v3_tmem(out, Q, K, V, tmem, B, T, S, H, D, scale, causal, fp4)
        bind(FN_FLASH_ATTN_V3_TMEM, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

        // int gollek_blackwell_rmsnorm(out, x, weight, N, eps)
        bind(FN_RMSNORM, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        // int gollek_blackwell_silu_ffn(out, gate, up, N)
        bind(FN_SILU_FFN, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> sym = lookup.find(name);
        if (sym.isPresent()) {
            handles.put(name, Linker.nativeLinker().downcallHandle(sym.get(), descriptor));
            LOG.debugf("BlackwellBinding: bound %s", name);
        } else {
            LOG.warnf("BlackwellBinding: symbol not found — %s", name);
        }
    }

    private Object invoke(String name, Object... args) {
        MethodHandle mh = handles.get(name);
        if (mh == null)
            throw new IllegalStateException("Unbound: " + name);
        try {
            return mh.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new RuntimeException("BlackwellBinding." + name + " failed", t);
        }
    }

    static void reset() {
        instance = null;
    }
}
