package tech.kayys.gollek.kernel.paged;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

import org.jboss.logging.Logger;

/**
 * FFM-based binding to the PagedAttention CUDA kernel.
 * <p>
 * Loads {@code libgollek_kernels.so} at runtime and exposes the
 * {@code paged_attention_launch} function to Java. Uses JDK 25's
 * Foreign Function & Memory API for zero-copy, zero-overhead native calls.
 * <p>
 * <b>Initialization:</b>
 * <pre>{@code
 * PagedAttentionBinding.initialize(Path.of("/path/to/libgollek_kernels.so"));
 * PagedAttentionBinding binding = PagedAttentionBinding.getInstance();
 *
 * // Launch kernel
 * int result = binding.pagedAttentionLaunch(
 *     output, query, kCache, vCache,
 *     blockTables, contextLens,
 *     numSeqs, numHeads, headDim, blockSize, maxBlocksPerSeq, scale
 * );
 * }</pre>
 * <p>
 * <b>Graceful degradation:</b> If the native library is not available (e.g.,
 * no GPU), the binding falls back to {@link PagedAttentionCpuFallback}.
 *
 * @see PagedAttentionCpuFallback
 */
public class PagedAttentionBinding {

    private static final Logger LOG = Logger.getLogger(PagedAttentionBinding.class);

    private static volatile PagedAttentionBinding instance;

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    // --- Native function names ---
    private static final String FN_PAGED_ATTENTION = "paged_attention_launch";
    private static final String FN_CUDA_DEVICE_COUNT = "gollek_cuda_device_count";
    private static final String FN_CUDA_DEVICE_NAME = "gollek_cuda_device_name";

    private PagedAttentionBinding(SymbolLookup lookup) {
        this.lookup = lookup;
        this.nativeAvailable = (lookup != null);

        if (nativeAvailable) {
            bindAll();
        }
    }

    // ---- Initialization ----

    /**
     * Initialize the binding by loading the native library.
     *
     * @param libraryPath path to {@code libgollek_kernels.so}
     * @return true if the native library was loaded successfully
     */
    public static boolean initialize(Path libraryPath) {
        if (instance != null) {
            LOG.warn("PagedAttentionBinding already initialized");
            return instance.nativeAvailable;
        }

        try {
            SymbolLookup lookup = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new PagedAttentionBinding(lookup);
            LOG.infof("PagedAttention native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load PagedAttention native library from %s: %s. " +
                      "Falling back to CPU implementation.", libraryPath, e.getMessage());
            instance = new PagedAttentionBinding(null);
            return false;
        }
    }

    /**
     * Initialize with fallback mode (no native library).
     */
    public static void initializeFallback() {
        if (instance != null) return;
        instance = new PagedAttentionBinding(null);
        LOG.info("PagedAttention initialized in CPU fallback mode");
    }

    /**
     * Get the singleton instance.
     *
     * @return the binding instance
     * @throws IllegalStateException if not initialized
     */
    public static PagedAttentionBinding getInstance() {
        if (instance == null) {
            throw new IllegalStateException(
                    "PagedAttentionBinding not initialized. Call initialize() first.");
        }
        return instance;
    }

    /**
     * Whether the native CUDA kernel is available.
     */
    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    // ---- Kernel Invocation ----

    /**
     * Launch the PagedAttention kernel.
     * <p>
     * All {@link MemorySegment} arguments must point to device memory
     * (GPU pointers). The binding passes raw addresses to the CUDA kernel.
     *
     * @param output          Output tensor segment [numSeqs, numHeads, headDim]
     * @param query           Query tensor segment  [numSeqs, numHeads, headDim]
     * @param keyCache        K-cache pool segment   [totalBlocks, numHeads, blockSize, headDim]
     * @param valueCache      V-cache pool segment   [totalBlocks, numHeads, blockSize, headDim]
     * @param blockTables     Block table segment    [numSeqs, maxBlocksPerSeq]
     * @param contextLens     Context length segment [numSeqs]
     * @param numSeqs         Number of sequences in batch
     * @param numHeads        Number of attention heads
     * @param headDim         Dimension per head
     * @param blockSize       Tokens per cache block
     * @param maxBlocksPerSeq Maximum blocks per sequence
     * @param scale           Attention scale factor (1/sqrt(headDim))
     * @return 0 on success, CUDA error code on failure
     */
    public int pagedAttentionLaunch(
            MemorySegment output,
            MemorySegment query,
            MemorySegment keyCache,
            MemorySegment valueCache,
            MemorySegment blockTables,
            MemorySegment contextLens,
            int numSeqs,
            int numHeads,
            int headDim,
            int blockSize,
            int maxBlocksPerSeq,
            float scale
    ) {
        if (!nativeAvailable) {
            LOG.trace("Using CPU fallback for PagedAttention");
            return PagedAttentionCpuFallback.execute(
                    output, query, keyCache, valueCache,
                    blockTables, contextLens,
                    numSeqs, numHeads, headDim, blockSize, maxBlocksPerSeq, scale
            );
        }

        MethodHandle mh = methodHandles.get(FN_PAGED_ATTENTION);
        if (mh == null) {
            throw new IllegalStateException("paged_attention_launch not bound");
        }

        try {
            return (int) mh.invokeExact(
                    output, query, keyCache, valueCache,
                    blockTables, contextLens,
                    numSeqs, numHeads, headDim, blockSize, maxBlocksPerSeq, scale
            );
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke paged_attention_launch", e);
        }
    }

    /**
     * Get the number of available CUDA devices.
     *
     * @return number of CUDA devices, or 0 if native not available
     */
    public int getCudaDeviceCount() {
        if (!nativeAvailable) return 0;

        MethodHandle mh = methodHandles.get(FN_CUDA_DEVICE_COUNT);
        if (mh == null) return 0;

        try {
            return (int) mh.invokeExact();
        } catch (Throwable e) {
            LOG.warnf("Failed to query CUDA device count: %s", e.getMessage());
            return 0;
        }
    }

    // ---- Internal Binding ----

    private void bindAll() {
        // paged_attention_launch(float*, float*, float*, float*, int*, int*,
        //                        int, int, int, int, int, float) -> int
        bind(FN_PAGED_ATTENTION, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,       // return: error code
                ValueLayout.ADDRESS,        // output
                ValueLayout.ADDRESS,        // query
                ValueLayout.ADDRESS,        // key_cache
                ValueLayout.ADDRESS,        // value_cache
                ValueLayout.ADDRESS,        // block_tables
                ValueLayout.ADDRESS,        // context_lens
                ValueLayout.JAVA_INT,       // num_seqs
                ValueLayout.JAVA_INT,       // num_heads
                ValueLayout.JAVA_INT,       // head_dim
                ValueLayout.JAVA_INT,       // block_size
                ValueLayout.JAVA_INT,       // max_blocks_per_seq
                ValueLayout.JAVA_FLOAT      // scale
        ));

        // gollek_cuda_device_count() -> int
        bind(FN_CUDA_DEVICE_COUNT, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // gollek_cuda_device_name(char*, int) -> int
        bind(FN_CUDA_DEVICE_NAME, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT
        ));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            Linker linker = Linker.nativeLinker();
            MethodHandle mh = linker.downcallHandle(symbol.get(), descriptor);
            methodHandles.put(name, mh);
            LOG.debugf("Bound native function: %s", name);
        } else {
            LOG.warnf("Native symbol not found: %s (some features may be unavailable)", name);
        }
    }

    /**
     * Reset the binding (for testing).
     */
    static void reset() {
        instance = null;
    }
}
