package tech.kayys.gollek.provider.litert;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.*;

/**
 * Specialized runner for LLM models (like Gemma) using LiteRT 2.0 CompiledModel API.
 *
 * <p>In LiteRT 2.0, the legacy TfLite Interpreter/SignatureRunner API is replaced by:
 * <ul>
 *   <li>{@code LiteRtCompiledModel} for model compilation and inference</li>
 *   <li>{@code LiteRtTensorBuffer} for I/O data (with lock/unlock)</li>
 * </ul>
 *
 * <p>Note: LLM inference with KV-cache management requires the LiteRT-LM engine
 * for production use. This runner provides basic autoregressive generation
 * using the standard CompiledModel API.
 */
public class LiteRTInferenceRunner implements AutoCloseable {

    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(LiteRTInferenceRunner.class);

    private final LiteRTNativeBindings bindings;
    private final Path modelPath;
    private final LiteRTTokenizer tokenizer;

    private final boolean useGpu;
    private final int numThreads;

    private MemorySegment environment;
    private MemorySegment model;
    private MemorySegment options;
    private MemorySegment compiledModel;

    private int numInputs = 0;
    private int numOutputs = 0;

    private LiteRTGemmaMetalRunner gemmaMetalRunner;

    private final Arena arena = Arena.ofAuto();

    private int maxSeqLen = 2048;

    public LiteRTInferenceRunner(LiteRTNativeBindings bindings, Path modelPath,
                                  LiteRTTokenizer tokenizer, boolean useGpu, int numThreads) {
        this.bindings = bindings;
        this.modelPath = modelPath;
        this.tokenizer = tokenizer;
        this.useGpu = useGpu;
        this.numThreads = numThreads;
    }

    public void initialize() {
        log.info("Initializing LiteRT 2.0 InferenceRunner for: {}", modelPath);

        // Redirect native C++ stderr (LiteRT runtime logs) to file
        redirectNativeStderr();

        try {
            // 1. Parse container format first (before any native calls)
            LiteRTContainerParser.ContainerInfo info = LiteRTContainerParser.parse(modelPath);

            if (info.format() == LiteRTContainerParser.ContainerFormat.LITERTLM) {
                // .litertlm containers use the Gemma inference engine (multiplatform)
                log.info("Detected .litertlm container. Initializing Gemma inference engine.");
                this.gemmaMetalRunner = new LiteRTGemmaMetalRunner(modelPath, tokenizer);
                this.gemmaMetalRunner.initialize();
                return;
            }

            // 2. Create Environment for standard TFLite models
            this.environment = bindings.createEnvironment(arena);


            // Load model from file (standard .tflite or .task)
            this.model = bindings.createModelFromFile(modelPath.toAbsolutePath().toString(), arena);

            // Introspect
            int numSigs = bindings.getNumModelSignatures(model, arena);
            int numSubgraphs = bindings.getNumModelSubgraphs(model, arena);
            log.info("Model loaded: {} subgraphs, {} signatures", numSubgraphs, numSigs);

            if (numSigs > 0) {
                MemorySegment sig = bindings.getModelSignature(model, 0, arena);
                String sigKey = bindings.getSignatureKey(sig, arena);
                log.info("Primary signature: '{}'", sigKey);

                this.numInputs = bindings.getNumSignatureInputs(sig, arena);
                this.numOutputs = bindings.getNumSignatureOutputs(sig, arena);

                for (int i = 0; i < numInputs; i++) {
                    String name = bindings.getSignatureInputName(sig, i, arena);
                    log.info("  Input[{}]: {}", i, name);
                }
                for (int i = 0; i < numOutputs; i++) {
                    String name = bindings.getSignatureOutputName(sig, i, arena);
                    log.info("  Output[{}]: {}", i, name);
                }
            }

            // 3. Create Options with accelerator selection
            this.options = bindings.createOptions(arena);
            int accelerators = LiteRTNativeBindings.kLiteRtHwAcceleratorCpu;
            if (useGpu) {
                accelerators |= LiteRTNativeBindings.kLiteRtHwAcceleratorGpu;
            }
            bindings.setOptionsHardwareAccelerators(options, accelerators);

            // 4. Create CompiledModel
            long t0 = System.currentTimeMillis();
            this.compiledModel = bindings.createCompiledModel(environment, model, options, arena);
            log.info("✓ CompiledModel created in {}ms", System.currentTimeMillis() - t0);

            boolean fullyAccelerated = bindings.isFullyAccelerated(compiledModel, arena);
            log.info("Fully accelerated: {}", fullyAccelerated);

            log.info("LiteRT 2.0 InferenceRunner initialized successfully");

        } catch (Exception e) {
            log.error("Failed to initialize InferenceRunner", e);
            cleanup();
            throw new RuntimeException("LiteRT InferenceRunner init failed: " + e.getMessage(), e);
        }
    }

    /**
     * Generate text from a prompt using autoregressive decoding.
     *
     * <p>This is a simplified implementation using the CompiledModel API.
     * For production LLM inference with KV-cache, use the LiteRT-LM engine.
     */
    public void generate(String prompt, Consumer<String> tokenCallback) {
        if (gemmaMetalRunner != null) {
            gemmaMetalRunner.generate(prompt, tokenCallback);
            return;
        }

        if (compiledModel == null) {
            throw new RuntimeException("InferenceRunner not initialized");
        }

        int[] inputIds = tokenizer.encodeChatPrompt(prompt);
        log.debug("Encoded prompt: {} tokens", inputIds.length);

        try (Arena inferArena = Arena.ofConfined()) {
            // Single-step inference: feed all tokens, get logits
            int nextToken = runInferenceStep(inputIds, inferArena);

            // Autoregressive loop
            int maxNewTokens = 512;
            for (int i = 0; i < maxNewTokens; i++) {
                if (tokenizer.isEosToken(nextToken)) break;

                String tokenStr = tokenizer.decodeToken(nextToken);
                tokenCallback.accept(tokenStr);

                // Feed single token for next step
                nextToken = runInferenceStep(new int[]{nextToken}, inferArena);
            }
        } catch (Exception e) {
            log.error("Generation failed", e);
            tokenCallback.accept("\n[Error: " + e.getMessage() + "]");
        }
    }

    private int runInferenceStep(int[] inputIds, Arena inferArena) {
        // Prepare input buffer: int32 array of token IDs
        int inputByteSize = inputIds.length * 4;
        MemorySegment hostInput = inferArena.allocate(inputByteSize, 64);
        for (int i = 0; i < inputIds.length; i++) {
            hostInput.setAtIndex(JAVA_INT, i, inputIds[i]);
        }

        // Get buffer requirements
        MemorySegment inputBufReq = bindings.getCompiledModelInputBufferRequirements(
                compiledModel, 0, 0, inferArena);
        MemorySegment outputBufReq = bindings.getCompiledModelOutputBufferRequirements(
                compiledModel, 0, 0, inferArena);

        // Create managed buffers
        MemorySegment inputBuf = bindings.createManagedTensorBufferFromRequirements(
                environment, MemorySegment.NULL, inputBufReq, inferArena);
        MemorySegment outputBuf = bindings.createManagedTensorBufferFromRequirements(
                environment, MemorySegment.NULL, outputBufReq, inferArena);

        try {
            // Write input data
            MemorySegment locked = bindings.lockTensorBuffer(inputBuf,
                    LiteRTNativeBindings.kLiteRtTensorBufferLockModeWrite, inferArena);
            MemorySegment.copy(hostInput, 0, locked.reinterpret(inputByteSize), 0, inputByteSize);
            bindings.unlockTensorBuffer(inputBuf);

            // Build buffer arrays
            MemorySegment inputBufArray = inferArena.allocate(ADDRESS, 1);
            MemorySegment outputBufArray = inferArena.allocate(ADDRESS, 1);
            inputBufArray.set(ADDRESS, 0, inputBuf);
            outputBufArray.set(ADDRESS, 0, outputBuf);

            // Run
            bindings.runCompiledModel(compiledModel, 0, inputBufArray, 1, outputBufArray, 1);

            // Read output logits
            long outSize = bindings.getTensorBufferSize(outputBuf, inferArena);
            MemorySegment outLocked = bindings.lockTensorBuffer(outputBuf,
                    LiteRTNativeBindings.kLiteRtTensorBufferLockModeRead, inferArena);

            int numLogits = (int) (outSize / 4);
            float[] logits = new float[numLogits];
            MemorySegment outMem = outLocked.reinterpret(outSize);
            for (int i = 0; i < numLogits; i++) {
                logits[i] = outMem.getAtIndex(JAVA_FLOAT, i);
            }
            bindings.unlockTensorBuffer(outputBuf);

            // Argmax sampling
            return argmax(logits);
        } finally {
            bindings.destroyTensorBuffer(inputBuf);
            bindings.destroyTensorBuffer(outputBuf);
        }
    }

    private int argmax(float[] logits) {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private void cleanup() {
        try {
            if (gemmaMetalRunner != null) {
                gemmaMetalRunner.close();
                gemmaMetalRunner = null;
            }
            if (compiledModel != null) { bindings.destroyCompiledModel(compiledModel); compiledModel = null; }
            if (options != null) { bindings.destroyOptions(options); options = null; }
            if (model != null) { bindings.destroyModel(model); model = null; }
            if (environment != null) { bindings.destroyEnvironment(environment); environment = null; }
        } catch (Exception e) {
            log.error("Cleanup failed", e);
        }
    }

    /**
     * Redirect native C/C++ stderr to a log file.
     * The LiteRT runtime writes INFO/WARNING messages directly to stderr
     * (from environment.cc, auto_registration.cc, etc.) which pollutes the CLI.
     * We redirect fd 2 via Java's System.setErr to capture these.
     */
    private static void redirectNativeStderr() {
        try {
            Path logDir = Path.of(System.getProperty("user.home"), ".gollek", "logs");
            Files.createDirectories(logDir);
            Path logFile = logDir.resolve("litert-native.log");
            FileOutputStream fos = new FileOutputStream(logFile.toFile(), true);
            System.setErr(new PrintStream(fos, true));
            log.debug("Native stderr redirected to {}", logFile);
        } catch (Exception e) {
            log.debug("Could not redirect native stderr: {}", e.getMessage());
        }
    }

    @Override
    public void close() {
        cleanup();
        arena.close();
    }
}
