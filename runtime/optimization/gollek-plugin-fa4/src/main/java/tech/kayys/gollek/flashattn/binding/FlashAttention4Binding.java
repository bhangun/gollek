package tech.kayys.gollek.flashattn.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the FlashAttention-4 CUDA kernel for NVIDIA Blackwell (sm_100a).
 *
 * <p>Loads {@code libgollek_fa4_kernels.so} at runtime via JDK FFM API.
 * Mirrors the structure of {@code FlashAttention3Binding} exactly.
 *
 * <h2>C ABI exposed by libgollek_fa4_kernels.so</h2>
 * <pre>
 * int flash_attention_4_launch(
 *     float*       output,       // [B, T, H, D] output
 *     const float* query,        // [B, T, H, D]
 *     const float* key_cache,    // paged K pool
 *     const float* value_cache,  // paged V pool
 *     int          batch_size,
 *     int          seq_len,
 *     int          num_heads,
 *     int          num_heads_k,  // GQA KV heads
 *     int          head_dim,
 *     float        softmax_scale,
 *     int          is_causal,
 *     int          use_fp8
 * );
 * </pre>
 *
 * <h2>Build</h2>
 * <pre>
 *   make -C src/main/cpp/fa4   # requires CUDA 12.8+ sm_100a
 * </pre>
 */
public class FlashAttention4Binding {

    private static final Logger LOG = Logger.getLogger(FlashAttention4Binding.class);
    private static volatile FlashAttention4Binding instance;

    private static final String FN_FA4_LAUNCH = "flash_attention_4_launch";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private FlashAttention4Binding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation (mirrors FA3Binding exactly) ───────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new FlashAttention4Binding(lk);
            LOG.infof("FlashAttention-4 native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load FA4 native library from %s: %s. Falling back to CPU.",
                    libraryPath, e.getMessage());
            instance = new FlashAttention4Binding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new FlashAttention4Binding(null);
        LOG.info("FlashAttention-4 initialized in CPU fallback mode");
    }

    public static FlashAttention4Binding getInstance() {
        if (instance == null)
            throw new IllegalStateException("FlashAttention4Binding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Kernel invocation ─────────────────────────────────────────────────────

    /**
     * Launch the FA4 CUDA kernel.
     *
     * @param output       [B, T, H, D] fp32/fp16/fp8 output
     * @param query        [B, T, H, D] query tensor
     * @param keyCache     paged K-cache pool (from PhysicalBlockPool.rawKPool())
     * @param valueCache   paged V-cache pool (from PhysicalBlockPool.rawVPool())
     * @param batchSize    B
     * @param seqLen       T (full context length, not just decode query)
     * @param numHeads     H (query heads)
     * @param numHeadsK    H_kv (key/value heads for GQA)
     * @param headDim      D
     * @param softmaxScale 1/sqrt(D)
     * @param isCausal     apply causal mask
     * @param useFp8       use FP8 precision (Blackwell native)
     * @return 0 on success, CUDA error code on failure
     */
    public int flashAttention4Launch(
            MemorySegment output,
            MemorySegment query,
            MemorySegment keyCache,
            MemorySegment valueCache,
            int batchSize,
            int seqLen,
            int numHeads,
            int numHeadsK,
            int headDim,
            float softmaxScale,
            boolean isCausal,
            boolean useFp8) {

        if (!nativeAvailable) {
            return FlashAttention4CpuFallback.execute(
                    output, query, keyCache, valueCache,
                    batchSize, seqLen, numHeads, numHeadsK, headDim,
                    softmaxScale, isCausal, useFp8);
        }

        MethodHandle mh = methodHandles.get(FN_FA4_LAUNCH);
        if (mh == null) throw new IllegalStateException(FN_FA4_LAUNCH + " not bound");

        try {
            return (int) mh.invokeExact(
                    output, query, keyCache, valueCache,
                    batchSize, seqLen, numHeads, numHeadsK, headDim,
                    softmaxScale,
                    isCausal ? 1 : 0,
                    useFp8   ? 1 : 0);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_FA4_LAUNCH, e);
        }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int flash_attention_4_launch(float*, float*, float*, float*,
        //                              int, int, int, int, int,
        //                              float, int, int) -> int
        bind(FN_FA4_LAUNCH, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,    // return: error code
                ValueLayout.ADDRESS,     // output
                ValueLayout.ADDRESS,     // query
                ValueLayout.ADDRESS,     // key_cache
                ValueLayout.ADDRESS,     // value_cache
                ValueLayout.JAVA_INT,    // batch_size
                ValueLayout.JAVA_INT,    // seq_len
                ValueLayout.JAVA_INT,    // num_heads
                ValueLayout.JAVA_INT,    // num_heads_k
                ValueLayout.JAVA_INT,    // head_dim
                ValueLayout.JAVA_FLOAT,  // softmax_scale
                ValueLayout.JAVA_INT,    // is_causal
                ValueLayout.JAVA_INT     // use_fp8
        ));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            MethodHandle mh = Linker.nativeLinker().downcallHandle(symbol.get(), descriptor);
            methodHandles.put(name, mh);
            LOG.debugf("Bound native symbol: %s", name);
        } else {
            LOG.warnf("Native symbol not found: %s", name);
        }
    }

    static void reset() { instance = null; }
}
