package tech.kayys.gollek.waitscheduler.binding;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.jboss.logging.Logger;

/**
 * FFM binding to the ForeLen output-length predictor.
 *
 * <p>Loads {@code libgollek_forelen.so} at runtime.
 *
 * <h2>What ForeLen is</h2>
 * <p>ForeLen (ICLR 2026) is a 2-layer MLP trained on (prompt_tokens → output_tokens)
 * pairs. It runs on CPU in ~0.5 ms and provides an output-length estimate used
 * by the WAIT fluid admission scheduler to compute per-request KV block demand.
 *
 * <h2>C ABI exposed by libgollek_forelen.so</h2>
 * <pre>
 * // Predict output token count for a prompt of length prompt_tokens.
 * int forelen_predict(int prompt_tokens);
 *
 * // Predict for a batch of prompts in one call.
 * // out_predictions: [batch_size] int32 — predicted output token count per prompt
 * int forelen_predict_batch(
 *     int*       out_predictions,  // [batch_size] output
 *     const int* prompt_lengths,   // [batch_size] input
 *     int        batch_size
 * );
 *
 * // Update the model with new (prompt, actual_output) pairs for online learning.
 * int forelen_update(
 *     const int* prompt_lengths,   // [n]
 *     const int* actual_outputs,   // [n]
 *     int        n
 * );
 *
 * // Return current mean absolute error (MAE) on recent predictions.
 * float forelen_mae(void);
 * </pre>
 *
 * <h2>Build</h2>
 * <pre>
 *   python -m forelen.server --export /opt/gollek/lib/libgollek_forelen.so
 *   # Or via CMake:
 *   make -C src/main/cpp/forelen
 * </pre>
 *
 * <h2>Paper</h2>
 * ForeLen: Predicting Output Length for Efficient LLM Scheduling (ICLR 2026)
 */
public class ForelenBinding {

    private static final Logger LOG = Logger.getLogger(ForelenBinding.class);
    private static volatile ForelenBinding instance;

    private static final String FN_PREDICT       = "forelen_predict";
    private static final String FN_PREDICT_BATCH = "forelen_predict_batch";
    private static final String FN_UPDATE        = "forelen_update";
    private static final String FN_MAE           = "forelen_mae";

    private final SymbolLookup lookup;
    private final Map<String, MethodHandle> methodHandles = new ConcurrentHashMap<>();
    private final boolean nativeAvailable;

    private ForelenBinding(SymbolLookup lookup) {
        this.lookup          = lookup;
        this.nativeAvailable = (lookup != null);
        if (nativeAvailable) bindAll();
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    public static boolean initialize(Path libraryPath) {
        if (instance != null) return instance.nativeAvailable;
        try {
            SymbolLookup lk = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            instance = new ForelenBinding(lk);
            LOG.infof("ForeLen native binding loaded from %s", libraryPath);
            return true;
        } catch (Exception e) {
            LOG.warnf("Failed to load ForeLen library from %s: %s. Using heuristic fallback.",
                    libraryPath, e.getMessage());
            instance = new ForelenBinding(null);
            return false;
        }
    }

    public static void initializeFallback() {
        if (instance != null) return;
        instance = new ForelenBinding(null);
        LOG.info("ForeLen initialized in heuristic fallback mode");
    }

    public static ForelenBinding getInstance() {
        if (instance == null)
            throw new IllegalStateException("ForelenBinding not initialized.");
        return instance;
    }

    public boolean isNativeAvailable() { return nativeAvailable; }

    // ── Predictions ───────────────────────────────────────────────────────────

    /**
     * Predict output token count for a single prompt.
     *
     * @param promptTokens number of tokens in the prompt
     * @return predicted output token count, or heuristic estimate if native unavailable
     */
    public int predict(int promptTokens) {
        if (!nativeAvailable) {
            return ForelenHeuristic.predict(promptTokens);
        }
        MethodHandle mh = methodHandles.get(FN_PREDICT);
        if (mh == null) return ForelenHeuristic.predict(promptTokens);
        try {
            return (int) mh.invokeExact(promptTokens);
        } catch (Throwable e) {
            LOG.warnf("ForeLen predict failed: %s — using heuristic", e.getMessage());
            return ForelenHeuristic.predict(promptTokens);
        }
    }

    /**
     * Predict output lengths for a batch of prompts.
     *
     * @param outPredictions output array [batchSize] int32
     * @param promptLengths  input array  [batchSize] int32
     * @param batchSize      batch size
     * @return 0 on success
     */
    public int predictBatch(MemorySegment outPredictions,
                            MemorySegment promptLengths,
                            int batchSize) {
        if (!nativeAvailable) {
            return ForelenHeuristic.predictBatch(outPredictions, promptLengths, batchSize);
        }
        MethodHandle mh = methodHandles.get(FN_PREDICT_BATCH);
        if (mh == null) return ForelenHeuristic.predictBatch(outPredictions, promptLengths, batchSize);
        try {
            return (int) mh.invokeExact(outPredictions, promptLengths, batchSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_PREDICT_BATCH, e);
        }
    }

    /**
     * Online model update: provide actual (prompt, output) pairs for retraining.
     *
     * @param promptLengths  [n] int32 prompt lengths
     * @param actualOutputs  [n] int32 actual output lengths
     * @param n              sample count
     * @return 0 on success
     */
    public int update(MemorySegment promptLengths, MemorySegment actualOutputs, int n) {
        if (!nativeAvailable) return 0; // no-op in heuristic mode
        MethodHandle mh = methodHandles.get(FN_UPDATE);
        if (mh == null) return 0;
        try {
            return (int) mh.invokeExact(promptLengths, actualOutputs, n);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to invoke " + FN_UPDATE, e);
        }
    }

    /** Current mean absolute error over recent predictions. */
    public float mae() {
        if (!nativeAvailable) return Float.NaN;
        MethodHandle mh = methodHandles.get(FN_MAE);
        if (mh == null) return Float.NaN;
        try { return (float) mh.invokeExact(); }
        catch (Throwable e) { return Float.NaN; }
    }

    // ── FFM wiring ────────────────────────────────────────────────────────────

    private void bindAll() {
        // int forelen_predict(int) -> int
        bind(FN_PREDICT, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT    // prompt_tokens
        ));

        // int forelen_predict_batch(int*, int*, int) -> int
        bind(FN_PREDICT_BATCH, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // out_predictions
                ValueLayout.ADDRESS,    // prompt_lengths
                ValueLayout.JAVA_INT    // batch_size
        ));

        // int forelen_update(int*, int*, int) -> int
        bind(FN_UPDATE, FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,    // prompt_lengths
                ValueLayout.ADDRESS,    // actual_outputs
                ValueLayout.JAVA_INT    // n
        ));

        // float forelen_mae() -> float
        bind(FN_MAE, FunctionDescriptor.of(ValueLayout.JAVA_FLOAT));
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
