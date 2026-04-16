package tech.kayys.gollek.qlora.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the QLoRA NF4+INT4 fused GEMM kernel.
 *
 * <p>Loads {@code libgollek_qlora.so} at runtime.
 *
 * <h2>C ABI exposed by libgollek_qlora.so</h2>
 * <pre>
 * // Fused matmul: out = (W_nf4 + scale * B × A) × input
 * // W_nf4 is the base model weight in NormalFloat4 quantisation.
 * // B [out_dim, rank] and A [rank, in_dim] are the LoRA adapter in INT4.
 * int qlora_fused_matmul(
 *     float*       out,        // [M, N]  output (fp32)
 *     const void*  W_nf4,      // [N, K]  base weight (NF4-packed, 2 values/byte)
 *     const void*  lora_B,     // [N, R]  adapter B matrix (INT4-packed)
 *     const void*  lora_A,     // [R, K]  adapter A matrix (INT4-packed)
 *     const float* input,      // [M, K]  input activations (fp32)
 *     float        lora_scale, // alpha / rank scaling factor
 *     int          M,          // batch × seq tokens
 *     int          N,          // output features
 *     int          K,          // input features
 *     int          R           // LoRA rank
 * );
 *
 * // Load an NF4-quantised weight tensor from a flat byte buffer
 * int qlora_load_nf4(
 *     void*         dst,       // pre-allocated output buffer (NF4-packed)
 *     const float*  src,       // [N, K] fp32 source weights
 *     int           N, int K
 * );
 *
 * // Load an INT4 LoRA adapter from a flat byte buffer
 * int qlora_load_adapter(
 *     void*         dst,       // pre-allocated output buffer (INT4-packed)
 *     const float*  src,       // fp32 adapter weights
 *     int           rows, int cols
 * );
 * </pre>
 *
 * <h2>Build</h2>
 * <pre>
 *   make -C src/main/cpp/qlora   # requires CUDA 12.x
 * </pre>
 */
public class QLoraBinding {

    private static final Logger LOG = Logger.getLogger(QLoraBinding.class);
    private static volatile QLoraBinding instance;

    private static final String FN_FUSED_MATMUL   = "qlora_fused_matmul";
    private static final String FN_LOAD_NF4       = "qlora_load_nf4";
    private static final String FN_LOAD_ADAPTER   = "qlora_load_adapter";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private QLoraBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new QLoraBinding(lk);
            LOG.infof("QLoRA native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load QLoRA native library from %s: %s. Falling back to CPU.",
                    libraryPath, e.getMessage());
            instance = new QLoraBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new QLoraBinding(null);
        LOG.info("QLoRA initialized in CPU fallback mode");
    }

    public static QLoraBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("QLoraBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Kernel invocations ────────────────────────────────────────────────────

    /**
     * Execute the fused NF4 base + INT4 LoRA GEMM.
     *
     * @param out        output [M, N] fp32
     * @param wNf4       base weight [N, K] NF4-packed
     * @param loraB      LoRA B matrix [N, R] INT4-packed
     * @param loraA      LoRA A matrix [R, K] INT4-packed
     * @param input      input activations [M, K] fp32
     * @param loraScale  alpha/rank scaling factor
     * @param M          tokens (batch × seq)
     * @param N          output features
     * @param K          input features
     * @param R          LoRA rank
     * @return 0 on success
     */
    public int fusedMatmul(
            MemorySegment out,
            MemorySegment wNf4,
            MemorySegment loraB,
            MemorySegment loraA,
            MemorySegment input,
            float loraScale,
            int M, int N, int K, int R) {

        if (!nativeAvailable) {
            return QLoraCpuFallback.fusedMatmul(
                    out, wNf4, loraB, loraA, input, loraScale, M, N, K, R);
        }
        MethodHandle mh = methodHandles.get(FN_FUSED_MATMUL);
        if (mh == null) throw new IllegalStateException(FN_FUSED_MATMUL + " not bound");
        try {
            return (int) mh.invokeExact(
                    out, wNf4, loraB, loraA, input, loraScale, M, N, K, R);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_FUSED_MATMUL, e);
        }
    }

    /**
     * Quantise fp32 weight tensor to NF4 layout in-place.
     */
    public int loadNf4(MemorySegment dst, MemorySegment src, int N, int K) {
        if (!nativeAvailable) return QLoraCpuFallback.loadNf4(dst, src, N, K);
        MethodHandle mh = methodHandles.get(FN_LOAD_NF4);
        if (mh == null) throw new IllegalStateException(FN_LOAD_NF4 + " not bound");
        try { return (int) mh.invokeExact(dst, src, N, K); }
        catch (Throwable e) { throw new RuntimeException("Failed to invoke " + FN_LOAD_NF4, e); }
    }

    /**
     * Pack fp32 adapter matrix to INT4 layout.
     */
    public int loadAdapter(MemorySegment dst, MemorySegment src, int rows, int cols) {
        if (!nativeAvailable) return QLoraCpuFallback.loadAdapter(dst, src, rows, cols);
        MethodHandle mh = methodHandles.get(FN_LOAD_ADAPTER);
        if (mh == null) throw new IllegalStateException(FN_LOAD_ADAPTER + " not bound");
        try { return (int) mh.invokeExact(dst, src, rows, cols); }
        catch (Throwable e) { throw new RuntimeException("Failed to invoke " + FN_LOAD_ADAPTER, e); }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int qlora_fused_matmul(float*, void*, void*, void*, float*,
        //                        float, int, int, int, int) -> int
        bind(FN_FUSED_MATMUL, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,    // return
                ValueLayout.ADDRESS,     // out
                ValueLayout.ADDRESS,     // W_nf4
                ValueLayout.ADDRESS,     // lora_B
                ValueLayout.ADDRESS,     // lora_A
                ValueLayout.ADDRESS,     // input
                ValueLayout.JAVA_FLOAT,  // lora_scale
                ValueLayout.JAVA_INT,    // M
                ValueLayout.JAVA_INT,    // N
                ValueLayout.JAVA_INT,    // K
                ValueLayout.JAVA_INT     // R
        ));

        // int qlora_load_nf4(void*, float*, int, int) -> int
        bind(FN_LOAD_NF4, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,     // dst
                ValueLayout.ADDRESS,     // src
                ValueLayout.JAVA_INT,    // N
                ValueLayout.JAVA_INT     // K
        ));

        // int qlora_load_adapter(void*, float*, int, int) -> int
        bind(FN_LOAD_ADAPTER, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,     // dst
                ValueLayout.ADDRESS,     // src
                ValueLayout.JAVA_INT,    // rows
                ValueLayout.JAVA_INT     // cols
        ));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            methodHandles.put(name,
                    Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
            LOG.debugf("Bound native symbol: %s", name);
        } else {
            LOG.warnf("Native symbol not found: %s", name);
        }
    }

    static void reset() { instance = null; }
}
