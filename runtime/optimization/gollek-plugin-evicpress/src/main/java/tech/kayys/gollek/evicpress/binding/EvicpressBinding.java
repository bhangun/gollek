package tech.kayys.gollek.evicpress.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the EVICPRESS KV-cache eviction/compression kernels.
 *
 * <p>Loads {@code libgollek_evicpress.so} at runtime.
 *
 * <h2>C ABI exposed by libgollek_evicpress.so</h2>
 * <pre>
 * // Score each block using cumulative attention, leverage, recency, head-diversity.
 * // Optionally runs GVote: samples K queries from N(μ,σ²) and votes on positions.
 * int evicpress_score(
 *     float*       scores,       // [num_blocks]  output importance score per block
 *     const float* attn_weights, // [num_blocks * block_size] attention accumulation
 *     int          num_blocks,
 *     int          block_size,
 *     int          num_layers,
 *     int          gvote_samples  // 0 = disabled, >0 = GVote sample count
 * );
 *
 * // Quantise a KV block in-place to the target dtype.
 * // dtype: "int4" | "int8" | "fp8"
 * int evicpress_compress(
 *     void*       block_data,  // KV block in GPU memory (read+write)
 *     long        bytes,
 *     const char* dtype        // null-terminated dtype string
 * );
 *
 * // Token-Route: mark blocks on critical attention chains as protected.
 * // Returns bitmask of protected block indices in protected_mask[num_blocks/64].
 * int evicpress_route_protect(
 *     uint64_t*    protected_mask, // [ceil(num_blocks/64)] output bitmask
 *     const float* attn_weights,
 *     int          num_blocks,
 *     int          block_size,
 *     int          num_layers
 * );
 * </pre>
 *
 * <h2>Build</h2>
 * <pre>
 *   make -C src/main/cpp/evicpress   # requires CUDA 12.x
 * </pre>
 *
 * <h2>Papers</h2>
 * EVICPRESS (arXiv:2512.14946), GVote (arXiv:2509.03136), Token-Route (arXiv:2603.01426)
 */
public class EvicpressBinding {

    private static final Logger LOG = Logger.getLogger(EvicpressBinding.class);
    private static volatile EvicpressBinding instance;

    private static final String FN_SCORE         = "evicpress_score";
    private static final String FN_COMPRESS      = "evicpress_compress";
    private static final String FN_ROUTE_PROTECT = "evicpress_route_protect";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private EvicpressBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new EvicpressBinding(lk);
            LOG.infof("EVICPRESS native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load EVICPRESS library from %s: %s. CPU fallback active.",
                    libraryPath, e.getMessage());
            instance = new EvicpressBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new EvicpressBinding(null);
        LOG.info("EVICPRESS initialized in CPU fallback mode");
    }

    public static EvicpressBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("EvicpressBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Kernel invocations ────────────────────────────────────────────────────

    /**
     * Score KV blocks by importance using four signals + optional GVote.
     *
     * @param scores       output [numBlocks] float32 — higher = more important
     * @param attnWeights  cumulative attention mass [numBlocks * blockSize]
     * @param numBlocks    number of blocks to score
     * @param blockSize    tokens per block
     * @param numLayers    transformer layer count
     * @param gvoteSamples GVote sample count (0 = disabled)
     * @return 0 on success
     */
    public int score(
            MemorySegment scores,
            MemorySegment attnWeights,
            int numBlocks, int blockSize, int numLayers, int gvoteSamples) {

        if (!nativeAvailable) {
            return EvicpressCpuFallback.score(
                    scores, attnWeights, numBlocks, blockSize, numLayers, gvoteSamples);
        }
        MethodHandle mh = methodHandles.get(FN_SCORE);
        if (mh == null) throw new IllegalStateException(FN_SCORE + " not bound");
        try {
            return (int) mh.invokeExact(
                    scores, attnWeights, numBlocks, blockSize, numLayers, gvoteSamples);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_SCORE, e);
        }
    }

    /**
     * Quantise a KV block in-place to reduce its VRAM footprint.
     *
     * @param blockData  block memory segment (modified in-place)
     * @param bytes      segment size
     * @param dtype      target quantisation: "int4", "int8", or "fp8"
     * @return 0 on success
     */
    public int compress(MemorySegment blockData, long bytes, String dtype) {
        if (!nativeAvailable) {
            return EvicpressCpuFallback.compress(blockData, bytes, dtype);
        }
        MethodHandle mh = methodHandles.get(FN_COMPRESS);
        if (mh == null) throw new IllegalStateException(FN_COMPRESS + " not bound");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment dtypeSeg = a.allocateFrom(dtype);
            return (int) mh.invokeExact(blockData, bytes, dtypeSeg);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_COMPRESS, e);
        }
    }

    /**
     * Token-Route: compute attention-chain connectivity and mark protected blocks.
     *
     * @param protectedMask output bitmask [ceil(numBlocks/64)] long array
     * @param attnWeights   attention weights [numBlocks * blockSize]
     * @param numBlocks     block count
     * @param blockSize     tokens per block
     * @param numLayers     layer count
     * @return 0 on success
     */
    public int routeProtect(
            MemorySegment protectedMask,
            MemorySegment attnWeights,
            int numBlocks, int blockSize, int numLayers) {

        if (!nativeAvailable) {
            return EvicpressCpuFallback.routeProtect(
                    protectedMask, attnWeights, numBlocks, blockSize, numLayers);
        }
        MethodHandle mh = methodHandles.get(FN_ROUTE_PROTECT);
        if (mh == null) throw new IllegalStateException(FN_ROUTE_PROTECT + " not bound");
        try {
            return (int) mh.invokeExact(
                    protectedMask, attnWeights, numBlocks, blockSize, numLayers);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_ROUTE_PROTECT, e);
        }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int evicpress_score(float*, float*, int, int, int, int) -> int
        bind(FN_SCORE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // scores
                ValueLayout.ADDRESS,    // attn_weights
                ValueLayout.JAVA_INT,   // num_blocks
                ValueLayout.JAVA_INT,   // block_size
                ValueLayout.JAVA_INT,   // num_layers
                ValueLayout.JAVA_INT    // gvote_samples
        ));

        // int evicpress_compress(void*, long, char*) -> int
        bind(FN_COMPRESS, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // block_data
                ValueLayout.JAVA_LONG,  // bytes
                ValueLayout.ADDRESS     // dtype (char*)
        ));

        // int evicpress_route_protect(uint64_t*, float*, int, int, int) -> int
        bind(FN_ROUTE_PROTECT, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // protected_mask
                ValueLayout.ADDRESS,    // attn_weights
                ValueLayout.JAVA_INT,   // num_blocks
                ValueLayout.JAVA_INT,   // block_size
                ValueLayout.JAVA_INT    // num_layers
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
