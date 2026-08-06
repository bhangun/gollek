package tech.kayys.gollek.kernel.fa3;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM-based binding to the FlashAttention-3 CUDA kernel targeting NVIDIA Hopper
 * architectures (sm_90) with TMA and FP8 support.
 * <p>
 * Loads {@code libgollek_fa3_kernels.so} at runtime.
 */
public class FlashAttention3Binding {

    private static final Logger LOG = Logger.getLogger(FlashAttention3Binding.class);
    private static volatile FlashAttention3Binding instance;

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private static final String FN_FA3_LAUNCH = "flash_attention_3_launch";

    private FlashAttention3Binding(SymbolLookup lookup) {
        this.lookup = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) {
            bindAll();
        }
    }

    public static boolean initialize(Path libraryPath) {
        if (instance != null) {
            return instance.nativeAvailable;
        }

        try {
            SymbolLookup lookup = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new FlashAttention3Binding(lookup);
            LOG.infof("FlashAttention-3 native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load FlashAttention-3 native library from %s: %s. " +
                      "Falling back to CPU standard attention.", libraryPath, e.getMessage());
            instance = new FlashAttention3Binding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new FlashAttention3Binding(null);
        LOG.info("FlashAttention-3 initialized in CPU fallback mode");
    }

    public static FlashAttention3Binding getInstance() {
        if (instance == null) {
            throw new IllegalStateException("FlashAttention3Binding not initialized.");
        }
        return instance;
    }

    public boolean isNativeAvailable() {
        return nativeAvailable;
    }

    public int flashAttention3Launch(
            MemorySegment output,
            MemorySegment query,
            MemorySegment key,
            MemorySegment value,
            int batchSize,
            int seqLen,
            int numHeads,
            int numHeadsK,
            int headDim,
            float softmaxScale,
            boolean isCausal,
            boolean useFp8
    ) {
        if (!nativeAvailable) {
            return FlashAttention3CpuFallback.execute(
                    output, query, key, value,
                    batchSize, seqLen, numHeads, numHeadsK, headDim,
                    softmaxScale, isCausal, useFp8
            );
        }

        MethodHandle mh = methodHandles.get(FN_FA3_LAUNCH);
        if (mh == null) {
            throw new IllegalStateException(FN_FA3_LAUNCH + " not bound");
        }

        try {
            return (int) mh.invokeExact(
                    output, query, key, value,
                    batchSize, seqLen, numHeads, numHeadsK, headDim,
                    softmaxScale, (isCausal ? 1 : 0), (useFp8 ? 1 : 0)
            );
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_FA3_LAUNCH, e);
        }
    }

    private void bindAll() {
        bind(FN_FA3_LAUNCH, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,       // return: error code
                ValueLayout.ADDRESS,        // output
                ValueLayout.ADDRESS,        // query
                ValueLayout.ADDRESS,        // key
                ValueLayout.ADDRESS,        // value
                ValueLayout.JAVA_INT,       // batch_size
                ValueLayout.JAVA_INT,       // seq_len
                ValueLayout.JAVA_INT,       // num_heads
                ValueLayout.JAVA_INT,       // num_heads_k
                ValueLayout.JAVA_INT,       // head_dim
                ValueLayout.JAVA_FLOAT,     // softmax_scale
                ValueLayout.JAVA_INT,       // is_causal
                ValueLayout.JAVA_INT        // use_fp8
        ));
    }

    private void bind(String name, FunctionDescriptor descriptor) {
        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isPresent()) {
            Linker linker = Linker.nativeLinker();
            MethodHandle mh = linker.downcallHandle(symbol.get(), descriptor);
            methodHandles.put(name, mh);
        } else {
            LOG.warnf("Native symbol not found: %s", name);
        }
    }
}
