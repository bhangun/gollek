package tech.kayys.gollek.metal.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the Metal FlashAttention-4 equivalence layer.
 *
 * <p>
 * Loads the FA4 symbols ({@code gollek_metal_fa4_attention}, etc.) from
 * {@code libgollek_metal.dylib} — the same dylib as {@link MetalBinding},
 * compiled from both {@code gollek_metal_bridge.m} <em>and</em>
 * {@code gollek_metal_fa4.m}.
 *
 * <h2>What "FA4 on Metal" means</h2>
 * <p>
 * FlashAttention-4 targets NVIDIA Blackwell (sm_100a). Apple Silicon has no
 * Blackwell hardware. However, the <em>algorithmic ideas</em> of FA4 map
 * directly
 * onto Apple Silicon:
 *
 * <table border="1" cellpadding="4">
 * <tr>
 * <th>FA4 (Blackwell)</th>
 * <th>Metal / Apple Silicon</th>
 * </tr>
 * <tr>
 * <td>TMEM tile accumulator (256 KB/SM)</td>
 * <td>GPU tile cache / threadgroup shared memory (on-chip SRAM)</td>
 * </tr>
 * <tr>
 * <td>Async UMMA pipelines</td>
 * <td>MPSCommandBuffer concurrent dispatch</td>
 * </tr>
 * <tr>
 * <td>Software exp() on FMA units</td>
 * <td>FP16 exp intrinsic on Apple M-series FMU</td>
 * </tr>
 * <tr>
 * <td>Fused Q×K^T/softmax/×V in one pass</td>
 * <td>MPSGraph.scaledDotProductAttention (macOS 14+, M3 optimised)</td>
 * </tr>
 * <tr>
 * <td>No full attention matrix on DRAM</td>
 * <td>Same guarantee — MPS SDPA tiles internally</td>
 * </tr>
 * </table>
 *
 * <p>
 * On <b>M3/M4</b> (MPSGraph SDPA path, macOS 14+), this is a true single-pass
 * fused kernel. On <b>M1/M2</b> (macOS 13, fallback path), it runs separate
 * MPS matmuls for QK^T and ×V.
 *
 * <h2>Lifecycle</h2>
 * 
 * <pre>{@code
 * // Initialize once (shares dylib with MetalBinding):
 * boolean loaded = MetalFlashAttentionBinding.initialize(
 *         Path.of("/opt/gollek/lib/libgollek_metal.dylib"));
 * MetalFlashAttentionBinding fa4 = MetalFlashAttentionBinding.getInstance();
 *
 * // Check hardware support:
 * boolean fusedSdpa = fa4.isSdpaAvailable(); // macOS 14+ fused path
 * boolean bf16 = fa4.isBf16Available(); // M2+ BF16
 *
 * // Run attention (K/V must be gathered from paged cache before calling):
 * fa4.fa4Attention(out, Q, K_gathered, V_gathered,
 *         B, T, S, H, H_kv, D, scale, true, false);
 * }</pre>
 */
public class MetalFlashAttentionBinding {

    private static final Logger LOG = Logger.getLogger(MetalFlashAttentionBinding.class);
    private static volatile MetalFlashAttentionBinding instance;

    private static final String FN_FA4_ATTENTION = "gollek_metal_fa4_attention";
    private static final String FN_SDPA_AVAILABLE = "gollek_metal_fa4_sdpa_available";
    private static final String FN_BF16_AVAILABLE = "gollek_metal_fa4_bf16_available";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    // Cached capability flags (read once after initialize)
    private final boolean sdpaAvailable;
    private final boolean bf16Available;

    private MetalFlashAttentionBinding(SymbolLookup lookup) {
        this.lookup = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) {
            bindAll();
            this.sdpaAvailable = querySdpaAvailable();
            this.bf16Available = queryBf16Available();
        } else {
            this.sdpaAvailable = false;
            this.bf16Available = false;
        }
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    /**
     * Load FA4 symbols from {@code libgollek_metal.dylib}.
     * The dylib must be compiled from both {@code gollek_metal_bridge.m}
     * and {@code gollek_metal_fa4.m}.
     */
    /**
     * Automatically discover and initialize the Metal FA4 native bridge.
     *
     * @return true if successfully loaded.
     */
    public static boolean initialize() {
        return initialize(MetalLibraryDiscovery.findLibrary());
    }

    public static boolean initialize(Path libraryPath) {
        if (instance != null)
            return instance.nativeAvailable;
        if (libraryPath == null || !Files.exists(libraryPath)) {
            LOG.warnf("MetalFlashAttentionBinding: dylib not found at %s — CPU fallback active", libraryPath);
            instance = new MetalFlashAttentionBinding(null);
            return false;
        }
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new MetalFlashAttentionBinding(lk);
            LOG.infof("MetalFlashAttentionBinding loaded from %s (SDPA=%s, BF16=%s)",
                    libraryPath, instance.sdpaAvailable, instance.bf16Available);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load MetalFA4 library from %s: %s. CPU fallback active.",
                    libraryPath, e.getMessage());
            instance = new MetalFlashAttentionBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null)
            return;
        instance = new MetalFlashAttentionBinding(null);
        LOG.info("MetalFlashAttentionBinding initialized in CPU fallback mode");
    }

    public static MetalFlashAttentionBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("MetalFlashAttentionBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    /** True when running on macOS 14+ — MPSGraph SDPA fused single-pass path. */
    public boolean isSdpaAvailable() {
        return sdpaAvailable;
    }

    /** True on M2/M3/M4+ — BF16 compute supported. */
    public boolean isBf16Available() {
        return bf16Available;
    }

    // ── Kernel invocation ─────────────────────────────────────────────────────

    /**
     * Fused FlashAttention-4-equivalent on Metal.
     *
     * <p>
     * K and V must already be <em>gathered</em> from the paged KV-cache
     * into contiguous buffers before calling. Use
     * {@link tech.kayys.gollek.kvcache.PagedKVCacheManager#getBlockPool()}
     * to obtain the raw K/V pool and gather the relevant blocks.
     *
     * <p>
     * On macOS 14+ (M3 optimised): dispatches to
     * {@code MPSGraph.scaledDotProductAttentionWithQuery} — a single-pass
     * kernel that tiles S=QK^T in on-chip L1 memory, matching FA4's
     * TMEM accumulator strategy.
     *
     * <p>
     * On macOS 13 (M1/M2 fallback): separate MPS QK^T matmul →
     * in-place softmax → MPS ×V matmul. Correct but materialises the
     * full attention matrix.
     *
     * @param output   [B, T, H, D] output (float16 in unified DRAM)
     * @param query    [B, T, H, D] query
     * @param key      [B, S, H_kv, D] gathered key (contiguous, not paged)
     * @param value    [B, S, H_kv, D] gathered value
     * @param B        batch size
     * @param T        query length
     * @param S        key/value context length
     * @param H        query heads
     * @param H_kv     key/value heads (GQA)
     * @param D        head dimension
     * @param scale    softmax scale (1/sqrt(D))
     * @param isCausal apply causal mask
     * @param useBf16  use BF16 arithmetic (ignored on M1)
     * @return 0 on success, -1 on error
     */
    public int fa4Attention(
            MemorySegment output,
            MemorySegment query,
            MemorySegment key,
            MemorySegment value,
            int B, int T, int S, int H, int H_kv, int D,
            float scale, boolean isCausal, boolean useBf16) {

        if (!nativeAvailable) {
            return MetalFlashAttentionCpuFallback.execute(
                    output, query, key, value,
                    B, T, S, H, H_kv, D, scale, isCausal);
        }

        MethodHandle mh = methodHandles.get(FN_FA4_ATTENTION);
        if (mh == null)
            throw new IllegalStateException(FN_FA4_ATTENTION + " not bound");
        try {
            return (int) mh.invokeExact(
                    output, query, key, value,
                    B, T, S, H, H_kv, D,
                    scale,
                    isCausal ? 1 : 0,
                    useBf16 ? 1 : 0);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_FA4_ATTENTION, e);
        }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int gollek_metal_fa4_attention(void*, void*, void*, void*,
        // int, int, int, int, int, int, float, int, int) -> int
        bind(FN_FA4_ATTENTION, FunctionDescriptor.of(
                ValueLayout.JAVA_INT, // return
                ValueLayout.ADDRESS, // output
                ValueLayout.ADDRESS, // query
                ValueLayout.ADDRESS, // key
                ValueLayout.ADDRESS, // value
                ValueLayout.JAVA_INT, // B
                ValueLayout.JAVA_INT, // T
                ValueLayout.JAVA_INT, // S
                ValueLayout.JAVA_INT, // H
                ValueLayout.JAVA_INT, // H_kv
                ValueLayout.JAVA_INT, // D
                ValueLayout.JAVA_FLOAT, // scale
                ValueLayout.JAVA_INT, // is_causal
                ValueLayout.JAVA_INT // use_bf16
        ));

        // int gollek_metal_fa4_sdpa_available() -> int
        bind(FN_SDPA_AVAILABLE, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // int gollek_metal_fa4_bf16_available() -> int
        bind(FN_BF16_AVAILABLE, FunctionDescriptor.of(ValueLayout.JAVA_INT));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            methodHandles.put(name,
                    Linker.nativeLinker().downcallHandle(symbol.get(), descriptor));
            LOG.debugf("Bound Metal FA4 symbol: %s", name);
        } else {
            LOG.warnf("Metal FA4 symbol not found: %s", name);
        }
    }

    private boolean querySdpaAvailable() {
        MethodHandle mh = methodHandles.get(FN_SDPA_AVAILABLE);
        if (mh == null)
            return false;
        try {
            return ((int) mh.invokeExact()) == 1;
        } catch (Throwable e) {
            return false;
        }
    }

    private boolean queryBf16Available() {
        MethodHandle mh = methodHandles.get(FN_BF16_AVAILABLE);
        if (mh == null)
            return false;
        try {
            return ((int) mh.invokeExact()) == 1;
        } catch (Throwable e) {
            return false;
        }
    }

    static void reset() {
        instance = null;
    }
}
