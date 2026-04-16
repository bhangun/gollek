package tech.kayys.gollek.hybridattn.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the Gated Delta Network CUDA kernel.
 *
 * <p>Loads {@code libgollek_gdn_kernels.so} at runtime.
 * Mirrors the structure of {@link tech.kayys.gollek.kernel.fa3.FlashAttention3Binding}.
 *
 * <h2>C ABI exposed by libgollek_gdn_kernels.so</h2>
 * <pre>
 * // Forward recurrence: S_t = α_t ⊙ S_{t-1} + β_t (v_t − S_{t-1}k_t) k_t^T
 * int gdn_layer_forward(
 *     float*       out,        // [B, T, model_dim]   output activations
 *     float*       state,      // [B, model_dim, S]   recurrent state (read+write)
 *     const float* input,      // [B, T, model_dim]   input activations
 *     const float* alpha,      // [B, T, model_dim]   output gate
 *     const float* beta,       // [B, T, model_dim]   update gate
 *     int          batch,
 *     int          seq_len,
 *     int          model_dim,
 *     int          state_dim
 * );
 *
 * // Populate gate tensors from input projection weights
 * int gdn_gate_project(
 *     float*       alpha_out,  // [B, T, D]
 *     float*       beta_out,   // [B, T, D]
 *     const float* input,      // [B, T, D]
 *     const float* w_alpha,    // [D, D]
 *     const float* w_beta,     // [D, D]
 *     int          B, int T, int D
 * );
 * </pre>
 *
 * <h2>Build</h2>
 * <pre>
 *   make -C src/main/cpp/gdn   # requires CUDA 12.x
 * </pre>
 *
 * <h2>Paper</h2>
 * Gated Delta Networks (arXiv:2412.06464, ICLR 2025) — Yang, Kautz, Hatamizadeh (NVIDIA)
 */
public class GdnBinding {

    private static final Logger LOG = Logger.getLogger(GdnBinding.class);
    private static volatile GdnBinding instance;

    private static final String FN_GDN_FORWARD      = "gdn_layer_forward";
    private static final String FN_GDN_GATE_PROJECT = "gdn_gate_project";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private GdnBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new GdnBinding(lk);
            LOG.infof("GDN native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load GDN native library from %s: %s. Falling back to CPU.",
                    libraryPath, e.getMessage());
            instance = new GdnBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new GdnBinding(null);
        LOG.info("GDN initialized in CPU fallback mode");
    }

    public static GdnBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("GdnBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Kernel invocations ────────────────────────────────────────────────────

    /**
     * Run one GDN recurrent layer forward pass.
     *
     * @param out      output activations [B, T, modelDim]
     * @param state    recurrent state [B, modelDim, stateDim] — updated in-place
     * @param input    input activations [B, T, modelDim]
     * @param alpha    output gate [B, T, modelDim]
     * @param beta     update gate [B, T, modelDim]
     * @param batch    B
     * @param seqLen   T
     * @param modelDim D
     * @param stateDim S
     * @return 0 on success, error code on failure
     */
    public int gdnLayerForward(
            MemorySegment out,
            MemorySegment state,
            MemorySegment input,
            MemorySegment alpha,
            MemorySegment beta,
            int batch, int seqLen, int modelDim, int stateDim) {

        if (!nativeAvailable) {
            return GdnCpuFallback.gdnLayerForward(
                    out, state, input, alpha, beta,
                    batch, seqLen, modelDim, stateDim);
        }
        MethodHandle mh = methodHandles.get(FN_GDN_FORWARD);
        if (mh == null) throw new IllegalStateException(FN_GDN_FORWARD + " not bound");
        try {
            return (int) mh.invokeExact(
                    out, state, input, alpha, beta,
                    batch, seqLen, modelDim, stateDim);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_GDN_FORWARD, e);
        }
    }

    /**
     * Compute alpha and beta gate tensors via projection matrices.
     */
    public int gdnGateProject(
            MemorySegment alphaOut,
            MemorySegment betaOut,
            MemorySegment input,
            MemorySegment wAlpha,
            MemorySegment wBeta,
            int B, int T, int D) {

        if (!nativeAvailable) {
            return GdnCpuFallback.gdnGateProject(
                    alphaOut, betaOut, input, wAlpha, wBeta, B, T, D);
        }
        MethodHandle mh = methodHandles.get(FN_GDN_GATE_PROJECT);
        if (mh == null) throw new IllegalStateException(FN_GDN_GATE_PROJECT + " not bound");
        try {
            return (int) mh.invokeExact(
                    alphaOut, betaOut, input, wAlpha, wBeta, B, T, D);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_GDN_GATE_PROJECT, e);
        }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int gdn_layer_forward(float*, float*, float*, float*, float*,
        //                       int, int, int, int) -> int
        bind(FN_GDN_FORWARD, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,   // return
                ValueLayout.ADDRESS,    // out
                ValueLayout.ADDRESS,    // state
                ValueLayout.ADDRESS,    // input
                ValueLayout.ADDRESS,    // alpha
                ValueLayout.ADDRESS,    // beta
                ValueLayout.JAVA_INT,   // batch
                ValueLayout.JAVA_INT,   // seq_len
                ValueLayout.JAVA_INT,   // model_dim
                ValueLayout.JAVA_INT    // state_dim
        ));

        // int gdn_gate_project(float*, float*, float*, float*, float*,
        //                      int, int, int) -> int
        bind(FN_GDN_GATE_PROJECT, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // alpha_out
                ValueLayout.ADDRESS,    // beta_out
                ValueLayout.ADDRESS,    // input
                ValueLayout.ADDRESS,    // w_alpha
                ValueLayout.ADDRESS,    // w_beta
                ValueLayout.JAVA_INT,   // B
                ValueLayout.JAVA_INT,   // T
                ValueLayout.JAVA_INT    // D
        ));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            methodHandles.put(name,
                    Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
            LOG.debugf("Bound native symbol: %s", name);
        } else {
            LOG.warnf("Native symbol not found: %s (some features may be unavailable)", name);
        }
    }

    static void reset() { instance = null; }
}
