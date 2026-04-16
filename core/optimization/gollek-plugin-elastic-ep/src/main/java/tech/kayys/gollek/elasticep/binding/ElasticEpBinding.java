package tech.kayys.gollek.elasticep.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the Elastic Expert Parallelism kernels.
 *
 * <p>Loads {@code libgollek_ep.so} at runtime.
 *
 * <h2>C ABI exposed by libgollek_ep.so</h2>
 * <pre>
 * // Route tokens to topK experts and combine outputs
 * int ep_dispatch(
 *     float*       out,          // [B, T, dim]     combined expert outputs
 *     const float* input,        // [B, T, dim]     input hidden states
 *     const int*   expert_ids,   // [B, T, top_k]   expert assignments per token
 *     int          B,            // batch size
 *     int          T,            // sequence length
 *     int          top_k,        // experts activated per token
 *     int          num_experts,  // total expert count
 *     int          dim           // hidden dimension
 * );
 *
 * // Solve min-cost assignment and update expert→GPU mapping
 * int ep_rebalance(
 *     int*         new_assignment,  // [num_experts]  new GPU index per expert (out)
 *     const float* load_histogram,  // [num_experts]  token count per expert
 *     int          num_experts,
 *     int          num_gpus
 * );
 *
 * // HMM zero-copy expert remap: update VA→PA mapping without data movement
 * int ep_hmm_remap(
 *     int   expert_id,
 *     int   src_gpu,
 *     int   dst_gpu
 * );
 * </pre>
 *
 * <h2>Build</h2>
 * <pre>
 *   make -C src/main/cpp/elasticep   # requires CUDA 12.x + HCCL
 * </pre>
 *
 * <h2>Paper</h2>
 * Elastic Expert Parallelism (arXiv:2510.02613, October 2025)
 */
public class ElasticEpBinding {

    private static final Logger LOG = Logger.getLogger(ElasticEpBinding.class);
    private static volatile ElasticEpBinding instance;

    private static final String FN_EP_DISPATCH   = "ep_dispatch";
    private static final String FN_EP_REBALANCE  = "ep_rebalance";
    private static final String FN_EP_HMM_REMAP  = "ep_hmm_remap";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private ElasticEpBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new ElasticEpBinding(lk);
            LOG.infof("ElasticEP native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load ElasticEP native library from %s: %s. Falling back to CPU.",
                    libraryPath, e.getMessage());
            instance = new ElasticEpBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new ElasticEpBinding(null);
        LOG.info("ElasticEP initialized in CPU fallback mode");
    }

    public static ElasticEpBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("ElasticEpBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Kernel invocations ────────────────────────────────────────────────────

    /**
     * Dispatch tokens to their assigned experts and combine outputs.
     *
     * @param out       combined output [B, T, dim]
     * @param input     input hidden states [B, T, dim]
     * @param expertIds expert assignments [B, T, topK] int32
     * @param B         batch size
     * @param T         sequence length
     * @param topK      experts activated per token
     * @param numExperts total expert count
     * @param dim       hidden dimension
     * @return 0 on success
     */
    public int epDispatch(
            MemorySegment out,
            MemorySegment input,
            MemorySegment expertIds,
            int B, int T, int topK, int numExperts, int dim) {

        if (!nativeAvailable) {
            return ElasticEpCpuFallback.epDispatch(
                    out, input, expertIds, B, T, topK, numExperts, dim);
        }
        MethodHandle mh = methodHandles.get(FN_EP_DISPATCH);
        if (mh == null) throw new IllegalStateException(FN_EP_DISPATCH + " not bound");
        try {
            return (int) mh.invokeExact(out, input, expertIds, B, T, topK, numExperts, dim);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_EP_DISPATCH, e);
        }
    }

    /**
     * Solve min-cost assignment and update expert-to-GPU mapping.
     *
     * @param newAssignment output: new GPU index per expert [numExperts] int32
     * @param loadHistogram tokens routed per expert in last window [numExperts] float32
     * @param numExperts    total expert count
     * @param numGpus       GPU count
     * @return 0 on success
     */
    public int epRebalance(
            MemorySegment newAssignment,
            MemorySegment loadHistogram,
            int numExperts, int numGpus) {

        if (!nativeAvailable) {
            return ElasticEpCpuFallback.epRebalance(
                    newAssignment, loadHistogram, numExperts, numGpus);
        }
        MethodHandle mh = methodHandles.get(FN_EP_REBALANCE);
        if (mh == null) throw new IllegalStateException(FN_EP_REBALANCE + " not bound");
        try {
            return (int) mh.invokeExact(newAssignment, loadHistogram, numExperts, numGpus);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_EP_REBALANCE, e);
        }
    }

    /**
     * HMM zero-copy expert remap — updates VA→PA without data movement.
     *
     * @param expertId expert to migrate
     * @param srcGpu   source GPU rank
     * @param dstGpu   destination GPU rank
     * @return 0 on success, -1 if HMM not available
     */
    public int epHmmRemap(int expertId, int srcGpu, int dstGpu) {
        if (!nativeAvailable) return 0; // no-op in CPU fallback
        MethodHandle mh = methodHandles.get(FN_EP_HMM_REMAP);
        if (mh == null) return -1;
        try {
            return (int) mh.invokeExact(expertId, srcGpu, dstGpu);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_EP_HMM_REMAP, e);
        }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int ep_dispatch(float*, float*, int*, int, int, int, int, int) -> int
        bind(FN_EP_DISPATCH, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,    // return
                ValueLayout.ADDRESS,     // out
                ValueLayout.ADDRESS,     // input
                ValueLayout.ADDRESS,     // expert_ids
                ValueLayout.JAVA_INT,    // B
                ValueLayout.JAVA_INT,    // T
                ValueLayout.JAVA_INT,    // top_k
                ValueLayout.JAVA_INT,    // num_experts
                ValueLayout.JAVA_INT     // dim
        ));

        // int ep_rebalance(int*, float*, int, int) -> int
        bind(FN_EP_REBALANCE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,     // new_assignment
                ValueLayout.ADDRESS,     // load_histogram
                ValueLayout.JAVA_INT,    // num_experts
                ValueLayout.JAVA_INT     // num_gpus
        ));

        // int ep_hmm_remap(int, int, int) -> int
        bind(FN_EP_HMM_REMAP, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT,    // expert_id
                ValueLayout.JAVA_INT,    // src_gpu
                ValueLayout.JAVA_INT     // dst_gpu
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
