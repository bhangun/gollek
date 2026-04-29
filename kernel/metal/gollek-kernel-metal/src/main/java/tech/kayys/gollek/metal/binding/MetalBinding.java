package tech.kayys.gollek.metal.binding;

import org.jboss.logging.Logger;

import java.io.File;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFM-based binding to the Gollek Metal bridge ({@code libgollek_metal.dylib}).
 *
 * <p>
 * Mirrors the pattern of the existing {@code FlashAttention3Binding} and
 * {@code PagedAttentionBinding}: a singleton loaded once at startup via
 * {@link SymbolLookup#libraryLookup}, falling back to a CPU implementation
 * when the dylib is absent (e.g., on non-Apple hardware).
 *
 * <h2>Unified Memory optimisation</h2>
 * <p>
 * On Apple Silicon the CPU and GPU share the same physical DRAM.
 * A {@link MemorySegment} allocated with {@link Arena#ofShared()} has a CPU
 * virtual address that Metal can access directly as a
 * {@code MTLStorageModeShared} buffer. The bridge's {@code wrap_ptr()}
 * function (in {@code gollek_metal_bridge.m}) calls
 * {@code newBufferWithBytesNoCopy} to hand the existing pointer to Metal —
 * <b>zero copy, zero extra allocation</b>.
 *
 * <h2>Lifecycle</h2>
 * 
 * <pre>{@code
 * // At startup (e.g., MetalRunner.initialize()):
 * boolean loaded = MetalBinding.initialize(
 *         Path.of("/opt/gollek/lib/libgollek_metal.dylib"));
 * MetalBinding binding = MetalBinding.getInstance();
 *
 * // Use:
 * binding.init();
 * binding.matmul(C, A, B, M, K, N, 1.0f, 0.0f);
 * binding.attention(out, Q, K, V, bt, ctx, B, T, H, D, blockSize, maxBlocks, scale, 1);
 * }</pre>
 */
public class MetalBinding {

    private static final Logger LOG = Logger.getLogger(MetalBinding.class);
    private static volatile MetalBinding instance;

    // ── Function names ────────────────────────────────────────────────────────

    private static final String FN_INIT = "gollek_metal_init";
    private static final String FN_AVAIL_MEM = "gollek_metal_available_memory";
    private static final String FN_ALLOC = "gollek_metal_alloc";
    private static final String FN_MATMUL = "gollek_metal_matmul";
    private static final String FN_ATTENTION = "gollek_metal_attention";
    private static final String FN_RMSNORM = "gollek_metal_rmsnorm";
    private static final String FN_SILU_FFN = "gollek_metal_silu_ffn";
    private static final String FN_UNIFIED_MEM = "gollek_metal_is_unified_memory";
    private static final String FN_DEVICE_NAME = "gollek_metal_device_name";

    // Basic Math Operations
    private static final String FN_ADD = "gollek_metal_add";
    private static final String FN_SUB = "gollek_metal_sub";
    private static final String FN_MUL = "gollek_metal_mul";
    private static final String FN_DIV = "gollek_metal_div";
    private static final String FN_RELU = "gollek_metal_relu";
    private static final String FN_SIGMOID = "gollek_metal_sigmoid";
    private static final String FN_TANH = "gollek_metal_tanh";
    private static final String FN_EXP = "gollek_metal_exp";
    private static final String FN_LOG = "gollek_metal_log";
    private static final String FN_SUM = "gollek_metal_sum";
    private static final String FN_MEAN = "gollek_metal_mean";
    private static final String FN_POW = "gollek_metal_pow";
    private static final String FN_TRANSPOSE2D = "gollek_metal_transpose2d";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> handles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private MetalBinding(SymbolLookup lookup) {
        this.lookup = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable)
            bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    /**
     * Automatically discover and initialize the Metal native bridge.
     * Searches in standard locations and java.library.path.
     *
     * @return true if successfully loaded.
     */
    public static boolean initialize() {
        return initialize(findLibrary());
    }

    public static boolean initialize(Path libraryPath) {
        if (instance != null)
            return instance.nativeAvailable;
        if (libraryPath == null || !Files.exists(libraryPath)) {
            LOG.warnf("MetalBinding: dylib not found at %s — CPU fallback active", libraryPath);
            instance = new MetalBinding(null);
            return false;
        }
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new MetalBinding(lk);
            LOG.infof("MetalBinding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("MetalBinding: failed to load %s (%s) — CPU fallback active",
                    libraryPath, e.getMessage());
            instance = new MetalBinding(null);
            return false;
        }
    }

    private static Path findLibrary() {
        String libName = "libgollek_metal.dylib";

        // 1. Explicit override
        String override = System.getProperty("gollek.metal.dylib");
        if (override != null) {
            Path p = Path.of(override);
            if (Files.exists(p))
                return p;
        }

        // 2. Standard Gollek installation path (~/.gollek/libs)
        String home = System.getProperty("user.home");
        if (home != null) {
            Path p = Path.of(home, ".gollek", "libs", libName);
            if (Files.exists(p))
                return p;
        }

        // 3. Search in java.library.path
        String libPath = System.getProperty("java.library.path");
        if (libPath != null) {
            for (String dir : libPath.split(File.pathSeparator)) {
                Path p = Path.of(dir, libName);
                if (Files.exists(p))
                    return p;
            }
        }

        // 4. Current directory
        Path p = Path.of(libName);
        if (Files.exists(p))
            return p;

        return p; // Return the path anyway; initialize(Path) will handle the Files.exists()
                  // check.
    }

    public static void initializeFallback() {
        if (instance != null)
            return;
        instance = new MetalBinding(null);
        LOG.info("MetalBinding: CPU fallback mode");
    }

    public static MetalBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException(
                    "MetalBinding not initialised — call initialize() first");
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Initialise the Metal device and command queue on the native side.
     * 
     * @return 0 on success, negative on error.
     */
    public int init() {
        if (!nativeAvailable)
            return 0; // CPU fallback — nothing to init
        return (int) invoke(FN_INIT);
    }

    /**
     * Available GPU memory in bytes (shared DRAM on Apple Silicon).
     */
    public long availableMemory() {
        if (!nativeAvailable)
            return Runtime.getRuntime().freeMemory();
        return (long) invoke(FN_AVAIL_MEM);
    }

    /**
     * Whether the device uses unified memory (always true on Apple Silicon
     * M-series).
     */
    public boolean isUnifiedMemory() {
        if (!nativeAvailable)
            return false;
        return ((int) invoke(FN_UNIFIED_MEM)) == 1;
    }

    /**
     * Metal device display name (e.g., "Apple M3 Pro").
     */
    public String deviceName() {
        if (!nativeAvailable)
            return "CPU (no Metal)";
        try (Arena a = Arena.ofConfined()) {
            MemorySegment buf = a.allocate(256L);
            invoke(FN_DEVICE_NAME, buf, 256);
            return buf.getString(0, java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    /**
     * Matrix multiplication via MPS: C = alpha · A × B + beta · C
     *
     * <p>
     * On Apple Silicon this dispatches to the AMX blocks via
     * MPSMatrixMultiplication — significantly faster than BLAS on CPU.
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
            return MetalCpuFallback.matmul(C, A, B, M, K, N, alpha, beta);
        }
        return (int) invoke(FN_MATMUL, C, A, B, M, K, N, alpha, beta);
    }

    /**
     * Paged softmax attention via Metal MPS.
     *
     * <p>
     * K/V cache is accessed directly from Gollek's
     * {@link tech.kayys.gollek.kvcache.PhysicalBlockPool} off-heap slabs —
     * no copy on Apple Silicon because the slabs are already in shared DRAM.
     *
     * @param out         [B, T, H, D] float32
     * @param Q           [B, T, H, D] float32
     * @param K_cache     paged K pool (from PhysicalBlockPool.rawKPool())
     * @param V_cache     paged V pool (from PhysicalBlockPool.rawVPool())
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
            float scale, int isCausal, float softCap) {
        if (!nativeAvailable) {
            return MetalCpuFallback.attention(out, Q, K_cache, V_cache,
                    blockTable, contextLens, B, T, H, D, blockSize, maxBlocks, scale, isCausal);
        }
        return (int) invoke(FN_ATTENTION,
                out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, D, blockSize, maxBlocks, scale, isCausal, softCap);
    }

    /**
     * RMS normalisation: out = x / rms(x) * weight.
     * 
     * @return 0 on success
     */
    public int rmsNorm(MemorySegment out, MemorySegment x,
            MemorySegment weight, int N, float eps, boolean addOne) {
        if (!nativeAvailable)
            return MetalCpuFallback.rmsNorm(out, x, weight, N, eps, addOne);
        return (int) invoke(FN_RMSNORM, out, x, weight, N, eps, addOne ? 1 : 0);
    }

    /**
     * SiLU-gated FFN: out = silu(gate) * up
     * 
     * @return 0 on success
     */
    public int siluFfn(MemorySegment out, MemorySegment gate,
            MemorySegment up, int N) {
        if (!nativeAvailable)
            return MetalCpuFallback.siluFfn(out, gate, up, N);
        return (int) invoke(FN_SILU_FFN, out, gate, up, N);
    }

    // ── Basic Math API ────────────────────────────────────────────────────────

    public int add(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_ADD, C, A, B, N);
    }

    public int sub(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_SUB, C, A, B, N);
    }

    public int mul(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_MUL, C, A, B, N);
    }

    public int div(MemorySegment C, MemorySegment A, MemorySegment B, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_DIV, C, A, B, N);
    }

    public int relu(MemorySegment C, MemorySegment A, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_RELU, C, A, N);
    }

    public int sigmoid(MemorySegment C, MemorySegment A, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_SIGMOID, C, A, N);
    }

    public int tanh(MemorySegment C, MemorySegment A, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_TANH, C, A, N);
    }

    public int exp(MemorySegment C, MemorySegment A, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_EXP, C, A, N);
    }

    public int log(MemorySegment C, MemorySegment A, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_LOG, C, A, N);
    }

    public int sum(MemorySegment out, MemorySegment A, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_SUM, out, A, N);
    }

    public int mean(MemorySegment out, MemorySegment A, int N) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_MEAN, out, A, N);
    }

    public int pow(MemorySegment C, MemorySegment A, int N, float p) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_POW, C, A, N, p);
    }

    public int transpose2d(MemorySegment C, MemorySegment A, int rows, int cols) {
        if (!nativeAvailable)
            return -1;
        return (int) invoke(FN_TRANSPOSE2D, C, A, rows, cols);
    }

    // ── FFM binding ───────────────────────────────────────────────────────────

    private void bindAll() {
        // int gollek_metal_init()
        bind(FN_INIT, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // long gollek_metal_available_memory()
        bind(FN_AVAIL_MEM, FunctionDescriptor.of(ValueLayout.JAVA_LONG));

        // void* gollek_metal_alloc(size_t bytes, size_t align)
        bind(FN_ALLOC, FunctionDescriptor.of(ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));

        // int gollek_metal_matmul(C, A, B, M, K, N, alpha, beta)
        bind(FN_MATMUL, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_FLOAT));

        // causal, soft_cap)
        bind(FN_ATTENTION, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        // int gollek_metal_rmsnorm(out, x, weight, N, eps, add_one)
        bind(FN_RMSNORM, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT, ValueLayout.JAVA_INT));

        // int gollek_metal_silu_ffn(out, gate, up, N)
        bind(FN_SILU_FFN, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT));

        // int gollek_metal_is_unified_memory()
        bind(FN_UNIFIED_MEM, FunctionDescriptor.of(ValueLayout.JAVA_INT));

        // int gollek_metal_device_name(char* buf, int bufSz)
        bind(FN_DEVICE_NAME, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

        // Basic Math ops
        FunctionDescriptor binaryDesc = FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);
        FunctionDescriptor unaryDesc = FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);

        bind(FN_ADD, binaryDesc);
        bind(FN_SUB, binaryDesc);
        bind(FN_MUL, binaryDesc);
        bind(FN_DIV, binaryDesc);

        bind(FN_RELU, unaryDesc);
        bind(FN_SIGMOID, unaryDesc);
        bind(FN_TANH, unaryDesc);
        bind(FN_EXP, unaryDesc);
        bind(FN_LOG, unaryDesc);
        bind(FN_SUM, unaryDesc);
        bind(FN_MEAN, unaryDesc);

        bind(FN_POW, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT));

        bind(FN_TRANSPOSE2D, FunctionDescriptor.of(ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> sym = lookup.find(name);
        if (sym.isPresent()) {
            handles.put(name, Linker.nativeLinker().downcallHandle(sym.get(), descriptor));
            LOG.debugf("MetalBinding: bound %s", name);
        } else {
            LOG.warnf("MetalBinding: symbol not found — %s", name);
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
            throw new RuntimeException("MetalBinding." + name + " failed", t);
        }
    }

    // ── Test helper ───────────────────────────────────────────────────────────

    static void reset() {
        instance = null;
    }
}
