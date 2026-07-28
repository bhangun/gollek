package tech.kayys.gollek.provider.litert;

import io.smallrye.mutiny.Multi;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.*;

import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

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
    private static final String ALLOW_NON_LITERT_FALLBACKS_PROPERTY =
            "gollek.litert.allow_nonlitert_fallbacks";
    private static final String PREFER_NATIVE_LITERTLM_PROPERTY =
            "gollek.litert.prefer_native_litertlm";
    static final String ENABLE_EXPERIMENTAL_GEMMA4_TASK_RUNNER_PROPERTY =
            "gollek.litert.enable_experimental_gemma4_task_runner";
    private static final Pattern GEMMA4_THOUGHT_CHANNEL =
            Pattern.compile("^<\\|channel>thought\\n.*?<channel\\|>", Pattern.DOTALL);
    private static final Pattern GEMMA4_GENERIC_CHANNEL_OPEN =
            Pattern.compile("^<\\|channel>[^\\n]*\\n", Pattern.DOTALL);
    private static final Pattern GEMMA4_ASSISTANT_TURN_OPEN =
            Pattern.compile("^(?:<\\|turn>|<start_of_turn>)(?:model|assistant)\\s*\\r?\\n?",
                    Pattern.CASE_INSENSITIVE);
    private static final Pattern GEMMA4_PLAIN_ASSISTANT_LINE =
            Pattern.compile("^(?:model|assistant)\\r?\\n+", Pattern.CASE_INSENSITIVE);

    private final LiteRTNativeBindings bindings;
    private final Path modelPath;
    private LiteRTTokenizer tokenizer;
    private boolean ownsTokenizer;

    private final boolean useGpu;
    private final int numThreads;

    private MemorySegment environment;
    private MemorySegment model;
    private MemorySegment options;
    private MemorySegment compiledModel;

    private int numInputs = 0;
    private int numOutputs = 0;

    private LiteRTGemmaMetalRunner gemmaMetalRunner;
    private LiteRTGemmaNativeRunner gemmaNativeRunner;
    private LiteRTLmJvmBridge officialJvmLmBridge;
    private LiteRTLmPythonBridge officialLmBridge;
    private Path safetensorFallbackModelPath;
    private Path litertLmRuntimeModelPath;
    private boolean preferCompanionSafetensorFallback;
    private String strictLiteRtFailureReason;

    private final Arena arena = Arena.ofConfined();

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
            Path runtimeModelPath = resolveRuntimeModelPath();
            this.safetensorFallbackModelPath = findCompanionSafetensorModel(runtimeModelPath);
            if (this.safetensorFallbackModelPath != null) {
                log.info("Found companion safetensor model {}", this.safetensorFallbackModelPath.getFileName());
            }

            // 1. Parse container format first (before any native calls)
            LiteRTContainerParser.ContainerInfo info = LiteRTContainerParser.parse(runtimeModelPath);

            if (info.format() == LiteRTContainerParser.ContainerFormat.LITERTLM) {
                this.litertLmRuntimeModelPath = runtimeModelPath;
                if (shouldPreferCompanionSafetensorFallback(runtimeModelPath)) {
                    if (allowNonLiteRtFallbacks()) {
                        this.preferCompanionSafetensorFallback = true;
                        log.warn("Bypassing native LiteRT-LM path for {} on {} and preferring companion fallback because {}=true",
                                runtimeModelPath.getFileName(),
                                System.getProperty("os.name", "unknown"),
                                ALLOW_NON_LITERT_FALLBACKS_PROPERTY);
                    } else {
                        this.strictLiteRtFailureReason = unsupportedLiteRtRuntimeMessage(runtimeModelPath, null);
                        log.error(this.strictLiteRtFailureReason);
                    }
                    return;
                }
                if (tryInitializeOfficialJvmBridge(runtimeModelPath)) {
                    return;
                }
                if (LiteRTLmPythonBridge.enabled()) {
                    if (LiteRTLmPythonBridge.available()) {
                        this.officialLmBridge = new LiteRTLmPythonBridge(runtimeModelPath, tokenizer(), useGpu);
                        log.warn("Using official LiteRT-LM Python bridge for {} (backend={}). "
                                        + "This path avoids the slow Java KV-cache copy fallback.",
                                runtimeModelPath.getFileName(),
                                LiteRTLmPythonBridge.backendName(useGpu));
                        return;
                    }
                    log.warn("Official LiteRT-LM Python bridge is not available through '{}'; "
                                    + "falling back to the slower manual LiteRT signature runner. "
                                    + "Install Google's litert_lm package or set -D{}=false to suppress this probe.",
                            LiteRTLmPythonBridge.pythonCommand(),
                            LiteRTLmPythonBridge.ENABLE_PROPERTY);
                }
                try {
                    initializeNativeLiteRtLmFallback(runtimeModelPath);
                    return;
                } catch (Exception e) {
                    if (isStrictLiteRtOnlyMode(runtimeModelPath)) {
                        this.strictLiteRtFailureReason = unsupportedLiteRtRuntimeMessage(runtimeModelPath, e);
                        log.error(this.strictLiteRtFailureReason);
                        return;
                    }
                    log.warn("Native LiteRT-LM runner init failed, falling back to Gemma Java runner: {}", e.getMessage());
                    this.gemmaNativeRunner = null;
                }
            }

            if (shouldUseGemmaRunner(runtimeModelPath, info)) {
                if (isGemma4LiteRtTask(runtimeModelPath)
                        && !Boolean.getBoolean(ENABLE_EXPERIMENTAL_GEMMA4_TASK_RUNNER_PROPERTY)) {
                    this.strictLiteRtFailureReason = "Gemma 4 LiteRT .task runner is disabled by default because "
                            + "the current Java/Metal path is slow and produces repeated-token output for this export. "
                            + "Set -D" + ENABLE_EXPERIMENTAL_GEMMA4_TASK_RUNNER_PROPERTY
                            + "=true only for diagnostics.";
                    log.error(this.strictLiteRtFailureReason);
                    return;
                }
                // Gemma LiteRT artifacts are more reliable through the dedicated runner
                // than the generic LiteRT loader path.
                log.info("Detected Gemma LiteRT artifact. Initializing Gemma inference engine.");
                this.gemmaMetalRunner = new LiteRTGemmaMetalRunner(runtimeModelPath, tokenizer());
                this.gemmaMetalRunner.initialize();
                return;
            }

            // 2. Create Environment for standard TFLite models
            this.environment = bindings.createEnvironment(arena);


            // Load model from file (standard .tflite or .task)
            this.model = bindings.createModelFromFile(runtimeModelPath.toAbsolutePath().toString(), arena);

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

    private void initializeNativeLiteRtLmFallback(Path runtimeModelPath) {
        log.info("Detected LiteRT-LM container. Initializing native LiteRT-LM signature fallback.");
        this.gemmaNativeRunner = new LiteRTGemmaNativeRunner(bindings, runtimeModelPath, tokenizer(), useGpu);
        this.gemmaNativeRunner.initialize();
    }

    private boolean tryInitializeOfficialJvmBridge(Path runtimeModelPath) {
        if (!LiteRTLmJvmBridge.enabled()) {
            return false;
        }
        if (!LiteRTLmJvmBridge.available()) {
            log.warn("Official LiteRT-LM JVM engine is not on the runtime classpath; "
                            + "falling back to other LiteRT-LM paths. The installed Gradle build should include "
                            + "com.google.ai.edge.litertlm:litertlm-jvm:0.11.0 from Google Maven.");
            return false;
        }
        try {
            this.officialJvmLmBridge = new LiteRTLmJvmBridge(runtimeModelPath, tokenizer, useGpu, numThreads);
            this.officialJvmLmBridge.initialize();
            return true;
        } catch (Exception gpuFailure) {
            closeOfficialJvmBridge();
            if (useGpu) {
                log.warn("Official LiteRT-LM JVM GPU engine failed; retrying official CPU backend: {}",
                        gpuFailure.getMessage());
                try {
                    this.officialJvmLmBridge = new LiteRTLmJvmBridge(runtimeModelPath, tokenizer, false, numThreads);
                    this.officialJvmLmBridge.initialize();
                    return true;
                } catch (Exception cpuFailure) {
                    closeOfficialJvmBridge();
                    log.warn("Official LiteRT-LM JVM CPU retry failed; falling back to other paths: {}",
                            cpuFailure.getMessage());
                    return false;
                }
            }
            log.warn("Official LiteRT-LM JVM engine failed; falling back to other paths: {}",
                    gpuFailure.getMessage());
            return false;
        }
    }

    private LiteRTTokenizer tokenizer() {
        if (tokenizer == null) {
            Path tokenizerModelPath = litertLmRuntimeModelPath != null ? litertLmRuntimeModelPath : modelPath;
            tokenizer = LiteRTTokenizer.create(tokenizerModelPath);
            ownsTokenizer = true;
        }
        return tokenizer;
    }

    private Path resolveRuntimeModelPath() {
        try {
            Path runtimeModelPath = modelPath;
            String originalName = modelPath.getFileName() != null
                    ? modelPath.getFileName().toString().toLowerCase(Locale.ROOT)
                    : "";
            boolean preferNativeLiteRtLm = Boolean.getBoolean(PREFER_NATIVE_LITERTLM_PROPERTY);
            if (!preferNativeLiteRtLm) {
                runtimeModelPath = LiteRTContainerParser.findBestModelFile(modelPath).orElse(modelPath);
            }
            if (!preferNativeLiteRtLm
                    && isMacOs()
                    && bindings != null
                    && isGemma4LiteRtTask(runtimeModelPath)
                    && !Boolean.getBoolean(ENABLE_EXPERIMENTAL_GEMMA4_TASK_RUNNER_PROPERTY)) {
                Path nativeGemmaPath = findNativeGemmaLitertlm(runtimeModelPath);
                if (nativeGemmaPath != null && !nativeGemmaPath.equals(runtimeModelPath)) {
                    log.warn("Using native LiteRT-LM sibling {} instead of experimental Gemma 4 task runner {} on macOS",
                            nativeGemmaPath.getFileName(),
                            runtimeModelPath.getFileName());
                    return nativeGemmaPath;
                }
            }
            if (preferNativeLiteRtLm) {
                Path nativeGemmaPath = findNativeGemmaLitertlm(runtimeModelPath);
                if (nativeGemmaPath != null && !nativeGemmaPath.equals(runtimeModelPath)) {
                    log.warn("Overriding runnable LiteRT file {} with native LiteRT-LM sibling {} because {}=true",
                            runtimeModelPath.getFileName(),
                            nativeGemmaPath.getFileName(),
                            PREFER_NATIVE_LITERTLM_PROPERTY);
                    return nativeGemmaPath;
                }
            }
            if (!runtimeModelPath.equals(modelPath)) {
                log.info("Using runnable LiteRT file {} instead of {}",
                        runtimeModelPath.getFileName(), modelPath.getFileName());
            }
            return runtimeModelPath;
        } catch (Exception e) {
            log.debug("Failed to normalize LiteRT model path {}: {}", modelPath, e.getMessage());
            return modelPath;
        }
    }

    private boolean shouldUseGemmaRunner(Path runtimeModelPath, LiteRTContainerParser.ContainerInfo info) {
        if (info.format() == LiteRTContainerParser.ContainerFormat.LITERTLM) {
            return false;
        }
        String fileName = runtimeModelPath.getFileName().toString().toLowerCase();
        return fileName.contains("gemma")
                && (fileName.endsWith(".task") || fileName.endsWith(".tflite") || fileName.endsWith(".tfl"));
    }

    private boolean isMacOs() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
    }

    private boolean isGemma4LiteRtTask(Path runtimeModelPath) {
        if (runtimeModelPath == null || runtimeModelPath.getFileName() == null) {
            return false;
        }
        String fileName = runtimeModelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        return (fileName.contains("gemma-4") || fileName.contains("gemma4"))
                && (fileName.endsWith(".task") || fileName.endsWith(".tflite") || fileName.endsWith(".tfl"));
    }

    private Path findNativeGemmaLitertlm(Path sourcePath) {
        try {
            Path searchDir = Files.isDirectory(sourcePath) ? sourcePath : sourcePath.getParent();
            if (searchDir == null || !Files.isDirectory(searchDir)) {
                return null;
            }
            try (var stream = Files.list(searchDir)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(p -> p.getFileName().toString().toLowerCase().endsWith(".litertlm"))
                        .filter(p -> p.getFileName().toString().toLowerCase().contains("gemma"))
                        .sorted((left, right) -> Integer.compare(
                                litertlmPreferenceScore(left),
                                litertlmPreferenceScore(right)))
                        .findFirst()
                        .orElse(null);
            }
        } catch (Exception e) {
            log.debug("Failed to find native Gemma LiteRT-LM sibling for {}: {}", sourcePath, e.getMessage());
            return null;
        }
    }

    private int litertlmPreferenceScore(Path candidate) {
        String fileName = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
        int score = 0;
        if (fileName.contains("qualcomm")) {
            score += 10;
        }
        if (!fileName.contains("gemma-4-e2b-it.litertlm")) {
            score += 1;
        }
        return score;
    }

    private boolean shouldPreferCompanionSafetensorFallback(Path runtimeModelPath) {
        if (runtimeModelPath == null || safetensorFallbackModelPath == null) {
            return false;
        }
        if (Boolean.getBoolean("gollek.litert.force_native_gemma4")) {
            return false;
        }
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (!os.contains("mac")) {
            return false;
        }
        String lower = runtimeModelPath.getFileName().toString().toLowerCase(Locale.ROOT);
        return lower.endsWith(".litertlm")
                && (lower.contains("gemma-4") || lower.contains("gemma4"))
                && lower.contains("qualcomm");
    }

    private boolean allowNonLiteRtFallbacks() {
        return Boolean.getBoolean(ALLOW_NON_LITERT_FALLBACKS_PROPERTY);
    }

    private boolean isStrictLiteRtOnlyMode(Path runtimeModelPath) {
        return shouldPreferCompanionSafetensorFallback(runtimeModelPath) && !allowNonLiteRtFallbacks();
    }

    private void failIfStrictLiteRtUnavailable() {
        if (strictLiteRtFailureReason != null && !strictLiteRtFailureReason.isBlank()) {
            throw new IllegalStateException(strictLiteRtFailureReason);
        }
    }

    private String unsupportedLiteRtRuntimeMessage(Path runtimeModelPath, Exception cause) {
        String fileName = runtimeModelPath != null && runtimeModelPath.getFileName() != null
                ? runtimeModelPath.getFileName().toString()
                : String.valueOf(runtimeModelPath);
        String osName = System.getProperty("os.name", "unknown");
        String base = "LiteRT cannot run " + fileName + " correctly on " + osName
                + " in strict LiteRT mode. Non-LiteRT fallbacks are disabled, so gollek will not switch to safetensor or GGUF automatically.";
        if (cause == null || cause.getMessage() == null || cause.getMessage().isBlank()) {
            return base + " Set -D" + ALLOW_NON_LITERT_FALLBACKS_PROPERTY + "=true only if you explicitly want cross-runtime fallback.";
        }
        return base + " Native LiteRT-LM error: " + cause.getMessage()
                + ". Set -D" + ALLOW_NON_LITERT_FALLBACKS_PROPERTY + "=true only if you explicitly want cross-runtime fallback.";
    }

    /**
     * Generate text from a prompt using autoregressive decoding.
     *
     * <p>This is a simplified implementation using the CompiledModel API.
     * For production LLM inference with KV-cache, use the LiteRT-LM engine.
     */
    public void generate(String prompt, Consumer<String> tokenCallback) {
        failIfStrictLiteRtUnavailable();
        if (preferCompanionSafetensorFallback) {
            InferenceRequest request =
                    InferenceRequest.builder()
                            .requestId("litert-inline")
                            .model(modelPath.toString())
                            .message(Message.user(prompt == null ? "" : prompt))
                            .temperature(0.2d)
                            .topK(40)
                            .topP(0.9d)
                            .maxTokens(512)
                            .repeatPenalty(1.1d)
                            .streaming(false)
                            .build();
            generate(request, tokenCallback);
            return;
        }
        if (gemmaMetalRunner != null) {
            gemmaMetalRunner.generate(prompt == null ? "" : prompt, tokenCallback);
            return;
        }
        generate(tokenizer().encodeChatPrompt(prompt), 512, 1.0d, tokenCallback);
    }

    public void generate(InferenceRequest request, Consumer<String> tokenCallback) {
        failIfStrictLiteRtUnavailable();
        if (preferCompanionSafetensorFallback) {
            if (tryGenerateWithSafetensorProviderFallback(request, tokenCallback)) {
                return;
            }
            if (tryGenerateWithSafetensorDirectFallback(request, tokenCallback)) {
                return;
            }
            log.warn("Preferred companion safetensor fallback paths failed; falling back to Gemma Java runner.");
            initializeGemmaFallbackRunner();
        }
        if (officialJvmLmBridge != null) {
            try {
                officialJvmLmBridge.generate(request, tokenCallback);
                return;
            } catch (Exception e) {
                log.warn("Official LiteRT-LM JVM bridge failed: {}. Falling back to next LiteRT-LM path.",
                        e.getMessage());
                closeOfficialJvmBridge();
                if (litertLmRuntimeModelPath != null && tryInitializeOfficialJvmBridge(litertLmRuntimeModelPath)) {
                    officialJvmLmBridge.generate(request, tokenCallback);
                    return;
                }
            }
        }
        if (officialLmBridge != null) {
            try {
                officialLmBridge.generate(request, tokenCallback);
                return;
            } catch (Exception e) {
                log.warn("Official LiteRT-LM bridge failed: {}. Falling back to native signature runner.",
                        e.getMessage());
                officialLmBridge = null;
                initializeNativeLiteRtLmFallback(litertLmRuntimeModelPath != null ? litertLmRuntimeModelPath : modelPath);
            }
        }
        if (gemmaNativeRunner != null) {
            try {
                gemmaNativeRunner.generate(request, tokenCallback);
                return;
            } catch (Exception e) {
                if (isStrictLiteRtOnlyMode(modelPath)) {
                    throw new IllegalStateException(unsupportedLiteRtRuntimeMessage(modelPath, e), e);
                }
                log.warn("Native LiteRT-LM generation failed: {}. Trying safetensor fallback paths.", e.getMessage());
                if (tryGenerateWithSafetensorProviderFallback(request, tokenCallback)) {
                    return;
                }
                if (tryGenerateWithSafetensorDirectFallback(request, tokenCallback)) {
                    return;
                }
                log.warn("All safetensor fallback paths failed; falling back to Gemma Java runner.");
                initializeGemmaFallbackRunner();
            }
        }
        if (gemmaMetalRunner != null) {
            gemmaMetalRunner.generate(
                    extractPromptTextForGemmaRunner(request),
                    Math.max(1, request.getMaxTokens()),
                    tokenCallback);
            return;
        }

        int[] inputIds = request.getMessages() != null && !request.getMessages().isEmpty()
                ? tokenizer().encodeChatPrompt(request.getMessages())
                : tokenizer().encodeChatPrompt(request.getPrompt() == null ? "" : request.getPrompt());
        generate(inputIds, Math.max(1, request.getMaxTokens()), request.getRepeatPenalty(), tokenCallback);
    }

    private void generate(int[] inputIds, int maxNewTokens, double repeatPenalty, Consumer<String> tokenCallback) {
        failIfStrictLiteRtUnavailable();
        if (gemmaNativeRunner != null) {
            try {
                gemmaNativeRunner.generate(inputIds, maxNewTokens, 0.2d, 40, 0.9d, repeatPenalty, tokenCallback);
                return;
            } catch (Exception e) {
                if (isStrictLiteRtOnlyMode(modelPath)) {
                    throw new IllegalStateException(unsupportedLiteRtRuntimeMessage(modelPath, e), e);
                }
                log.warn("Native LiteRT-LM generation failed, falling back to Gemma Java runner: {}", e.getMessage());
                initializeGemmaFallbackRunner();
            }
        }
        if (gemmaMetalRunner != null) {
            String promptText = requestlessPromptText(inputIds);
            gemmaMetalRunner.generate(promptText, tokenCallback);
            return;
        }

        if (compiledModel == null) {
            throw new RuntimeException("InferenceRunner not initialized");
        }

        log.debug("Encoded prompt: {} tokens", inputIds.length);

        try (Arena inferArena = Arena.ofConfined()) {
            // Single-step inference: feed all tokens, get logits
            int nextToken = runInferenceStep(inputIds, inferArena);

            // Autoregressive loop
            for (int i = 0; i < maxNewTokens; i++) {
                if (tokenizer().isTerminalToken(nextToken)) break;

                String tokenStr = tokenizer().decodeToken(nextToken);
                if (!tokenStr.isEmpty()) {
                    tokenCallback.accept(tokenStr);
                }

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

    private void initializeGemmaFallbackRunner() {
        if (gemmaMetalRunner != null) {
            return;
        }
        try {
            Path fallbackPath = LiteRTContainerParser.findBestModelFile(modelPath).orElse(modelPath);
            gemmaMetalRunner = new LiteRTGemmaMetalRunner(fallbackPath, tokenizer());
            gemmaMetalRunner.initialize();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize Gemma fallback runner", e);
        }
    }

    private String extractPromptTextForGemmaRunner(InferenceRequest request) {
        if (request == null) {
            return "";
        }
        String prompt = request.getPrompt();
        if (prompt != null && !prompt.isBlank()) {
            return prompt;
        }
        List<Message> messages = request.getMessages();
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        StringBuilder joined = new StringBuilder();
        for (Message message : messages) {
            if (message == null || message.getContent() == null || message.getContent().isBlank()) {
                continue;
            }
            if (!joined.isEmpty()) {
                joined.append("\n\n");
            }
            joined.append(message.getContent().trim());
        }
        return joined.toString();
    }

    private String requestlessPromptText(int[] inputIds) {
        if (inputIds == null || inputIds.length == 0) {
            return "";
        }
        return tokenizer().decode(inputIds);
    }

    private boolean tryGenerateWithSafetensorDirectFallback(
            InferenceRequest request,
            Consumer<String> tokenCallback) {
        if (safetensorFallbackModelPath == null) {
            return false;
        }
        try {
            Object directEngine = resolveArcBean("tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine");
            if (directEngine == null) {
                log.warn("Companion safetensor direct fallback unavailable: DirectInferenceEngine bean not available");
                return false;
            }

            String prompt = buildSafetensorPrompt(request);
            Object generationConfig = buildSafetensorGenerationConfig(request);
            Class<?> generationConfigClass = generationConfig.getClass();
            Method generateMethod = directEngine.getClass().getMethod(
                    "generate",
                    String.class,
                    Path.class,
                    generationConfigClass);
            Object uni = generateMethod.invoke(directEngine, prompt, safetensorFallbackModelPath, generationConfig);
            Object awaiter = uni.getClass().getMethod("await").invoke(uni);
            Object response = awaiter.getClass()
                    .getMethod("atMost", Duration.class)
                    .invoke(awaiter, Duration.ofMinutes(5));
            String content = (String) response.getClass().getMethod("getContent").invoke(response);
            String sanitized = sanitizeGemma4Text(content);
            if ((sanitized == null || sanitized.isBlank()) && content != null && !content.isBlank()) {
                log.warn("Companion safetensor direct fallback returned control-only text: {}",
                        abbreviateForLog(content));
            }
            if (sanitized != null && !sanitized.isEmpty()) {
                tokenCallback.accept(sanitized);
            }
            log.info("Generated response via companion safetensor direct fallback");
            return true;
        } catch (Exception e) {
            log.warn("Companion safetensor direct fallback failed: {}", rootCauseMessage(e));
            return false;
        }
    }

    private boolean tryGenerateWithSafetensorProviderFallback(
            InferenceRequest request,
            Consumer<String> tokenCallback) {
        if (safetensorFallbackModelPath == null) {
            return false;
        }
        if (tryGenerateWithCompanionGgufFallback(request, tokenCallback)) {
            return true;
        }
        if (tryGenerateWithNamedSafetensorProviderFallback(
                request,
                tokenCallback,
                "tech.kayys.gollek.inference.safetensor.SafetensorProvider",
                "companion safetensor auto/GGUF provider")) {
            return true;
        }
        return tryGenerateWithNamedSafetensorProviderFallback(
                request,
                tokenCallback,
                "tech.kayys.gollek.safetensor.engine.warmup.SafetensorProvider",
                "companion safetensor direct provider");
    }

        private boolean tryGenerateWithCompanionGgufFallback(
            InferenceRequest request,
            Consumer<String> tokenCallback) {
        return false;
    }

        private boolean tryGenerateWithNamedSafetensorProviderFallback(
            InferenceRequest request,
            Consumer<String> tokenCallback,
            String providerClassName,
            String providerLabel) {
        return false;
    }

    private boolean streamWithProvider(
            Object provider,
            InferenceRequest providerRequest,
            InferenceRequest request,
            Consumer<String> tokenCallback,
            String providerLabel) throws Exception {
        Method inferStreamMethod = provider.getClass().getMethod("inferStream", InferenceRequest.class);
        @SuppressWarnings("unchecked")
        Multi<StreamingInferenceChunk> stream =
                (Multi<StreamingInferenceChunk>) inferStreamMethod.invoke(provider, providerRequest);

        Duration timeout = request.getTimeout().orElse(Duration.ofMinutes(5));
        AtomicBoolean leadingChannelsPending = new AtomicBoolean(true);
        AtomicBoolean emittedText = new AtomicBoolean(false);
        AtomicReference<Throwable> failureRef = new AtomicReference<>();
        CountDownLatch completion = new CountDownLatch(1);

        stream.subscribe().with(
                chunk -> {
                    String delta = sanitizeGemma4StreamingDelta(chunk, leadingChannelsPending);
                    if (delta != null && !delta.isEmpty()) {
                        emittedText.set(true);
                        tokenCallback.accept(delta);
                    }
                },
                failure -> {
                    failureRef.set(failure);
                    completion.countDown();
                },
                completion::countDown);

        if (!completion.await(timeout.toMillis(), TimeUnit.MILLISECONDS)) {
            throw new IllegalStateException("Timed out waiting for " + providerLabel + " completion");
        }
        if (failureRef.get() != null) {
            throw new IllegalStateException(providerLabel + " failed", failureRef.get());
        }
        if (!emittedText.get()) {
            log.warn("{} completed without emitting text deltas", providerLabel);
            return false;
        }

        log.info("Generated response via {}", providerLabel);
        return true;
    }

    private static Object resolveArcBean(String className) throws Exception {
        try {
            Class<?> beanClass = Class.forName(className);
            Class<?> arcClass = Class.forName("io.quarkus.arc.Arc");
            Object container = arcClass.getMethod("container").invoke(null);
            if (container == null) {
                return null;
            }
            Object handle;
            try {
                handle = container.getClass()
                        .getMethod("instance", Class.class, java.lang.annotation.Annotation[].class)
                        .invoke(container, beanClass, new java.lang.annotation.Annotation[0]);
            } catch (NoSuchMethodException ignored) {
                handle = container.getClass().getMethod("instance", Class.class).invoke(container, beanClass);
            }
            try {
                Class<?> handleClass = Class.forName("io.quarkus.arc.InstanceHandle");
                Method isAvailable = handleClass.getMethod("isAvailable");
                Object available = isAvailable.invoke(handle);
                if (available instanceof Boolean b && !b) {
                    return null;
                }
            } catch (NoSuchMethodException ignored) {
                // Older Arc builds may only expose get(); fall through.
            }
            return Class.forName("io.quarkus.arc.InstanceHandle")
                    .getMethod("get")
                    .invoke(handle);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to resolve Arc bean " + className + ": " + rootCauseMessage(e), e);
        }
    }

    private static String rootCauseMessage(Throwable throwable) {
        if (throwable == null) {
            return "unknown";
        }
        Throwable cursor = throwable;
        while (cursor.getCause() != null && cursor.getCause() != cursor) {
            cursor = cursor.getCause();
        }
        String message = cursor.getMessage();
        if (message == null || message.isBlank()) {
            return cursor.getClass().getName();
        }
        return cursor.getClass().getName() + ": " + message;
    }

    private static String abbreviateForLog(String text) {
        if (text == null) {
            return "";
        }
        String singleLine = text.replace("\n", "\\n");
        if (singleLine.length() <= 200) {
            return singleLine;
        }
        return singleLine.substring(0, 200) + "...";
    }

    private void enableGemma4SafetensorFastPath() {
        if (safetensorFallbackModelPath == null) {
            return;
        }
        String lower = safetensorFallbackModelPath.toString().toLowerCase(Locale.ROOT);
        if (!lower.contains("gemma-4") && !lower.contains("gemma4")) {
            return;
        }
        setSystemPropertyIfAbsent("gollek.safetensor.experimental_metal_linear", "true");
        setSystemPropertyIfAbsent("gollek.safetensor.allow_legacy_metal_attention_bridge", "true");
        setSystemPropertyIfAbsent("gollek.safetensor.use_bf16_attention", "true");
    }

    private static void setSystemPropertyIfAbsent(String key, String value) {
        if (System.getProperty(key) == null) {
            System.setProperty(key, value);
        }
    }

    private Object buildSafetensorGenerationConfig(
            InferenceRequest request) throws Exception {
        Class<?> cfgClass = Class.forName("tech.kayys.gollek.safetensor.generation.GenerationConfig");
        Object builder = cfgClass.getMethod("builder").invoke(null);
        Class<?> builderClass = builder.getClass();
        Class<?> strategyClass = Class.forName(
                "tech.kayys.gollek.safetensor.generation.GenerationConfig$SamplingStrategy");

        float temperature = (float) request.getTemperature();
        int topK = request.getTopK();
        float topP = (float) request.getTopP();
        float repetitionPenalty = normalizeSafetensorRepeatPenalty(
                safetensorFallbackModelPath,
                (float) request.getRepeatPenalty());
        String strategyName = resolveSafetensorSamplingStrategy(temperature, topK, topP);
        @SuppressWarnings({"unchecked", "rawtypes"})
        Object strategy = Enum.valueOf((Class<? extends Enum>) strategyClass.asSubclass(Enum.class), strategyName);

        builderClass.getMethod("maxNewTokens", int.class).invoke(builder, Math.max(1, request.getMaxTokens()));
        builderClass.getMethod("strategy", strategyClass).invoke(builder, strategy);
        builderClass.getMethod("temperature", float.class).invoke(builder, temperature);
        builderClass.getMethod("topK", int.class).invoke(builder, topK);
        builderClass.getMethod("topP", float.class).invoke(builder, topP);
        builderClass.getMethod("repetitionPenalty", float.class).invoke(builder, repetitionPenalty);

        return builderClass.getMethod("build").invoke(builder);
    }

        private InferenceRequest buildSafetensorInferenceRequest(
            InferenceRequest request,
            boolean streaming) {
        return request;
    }

    

    private InferenceRequest normalizeSafetensorFallbackRequest(InferenceRequest request) {
        if (request == null) {
            return InferenceRequest.builder()
                    .model(modelPath.toString())
                    .message(Message.user(""))
                    .build();
        }
        if (request.getMessages() != null && !request.getMessages().isEmpty()) {
            return request;
        }
        String prompt = request.getPrompt();
        if (prompt == null || prompt.isBlank()) {
            return request;
        }
        return request.toBuilder()
                .message(Message.user(prompt))
                .build();
    }

    private void maybeEnableSafetensorFallbackQuantization(
            Map<String, Object> parameterOverrides,
            Map<String, Object> metadataOverrides,
            InferenceRequest request) {
        if (!shouldAutoQuantizeGemma4SafetensorFallback(request)) {
            return;
        }

        Map<String, Object> parameters = request.getParameters();
        if (parameters == null || !parameters.containsKey("quantize_strategy")) {
            parameterOverrides.put("quantize_strategy", "int4");
            metadataOverrides.put("quantize_strategy", "int4");
        }
        if (parameters == null || !parameters.containsKey("quantize_bits")) {
            parameterOverrides.put("quantize_bits", 4);
            metadataOverrides.put("quantize_bits", 4);
        }
    }

    private boolean shouldAutoQuantizeGemma4SafetensorFallback(
            InferenceRequest request) {
        if (safetensorFallbackModelPath == null) {
            return false;
        }
        String lower = safetensorFallbackModelPath.toString().toLowerCase(Locale.ROOT);
        if (!lower.contains("gemma-4") && !lower.contains("gemma4")) {
            return false;
        }
        String modelRef = request.getModel();
        if (modelRef == null || modelRef.isBlank()) {
            modelRef = modelPath.toString();
        }
        String modelLower = modelRef.toLowerCase(Locale.ROOT);
        return modelLower.endsWith(".litertlm")
                || modelLower.contains("litert-community")
                || modelLower.contains("litert");
    }

    private Path ensureCompanionGgufModel() throws Exception {
        if (safetensorFallbackModelPath == null) {
            return null;
        }
        Path sourceDir = Files.isDirectory(safetensorFallbackModelPath)
                ? safetensorFallbackModelPath
                : safetensorFallbackModelPath.getParent();
        if (sourceDir == null || !Files.exists(sourceDir)) {
            return null;
        }

        Path ggufDir = Path.of(System.getProperty("user.home"), ".gollek", "models", "gguf");
        Files.createDirectories(ggufDir);
        Path outputPath = ggufDir.resolve(normalizeCompanionGgufName(sourceDir) + "-Q4_K_M.gguf");
        if (Files.exists(outputPath) && Files.size(outputPath) > 0L) {
            return outputPath;
        }

        Object converter = resolveArcBean("tech.kayys.gollek.converter.GGUFConverter");
        if (converter == null) {
            log.info("Companion GGUF conversion unavailable: GGUFConverter bean not present");
            return null;
        }

        Class<?> paramsClass = Class.forName("tech.kayys.gollek.converter.model.GGUFConversionParams");
        Object builder = paramsClass.getMethod("builder").invoke(null);
        Class<?> builderClass = builder.getClass();
        Class<?> quantClass = Class.forName("tech.kayys.gollek.converter.model.QuantizationType");
        @SuppressWarnings({"unchecked", "rawtypes"})
        Object quant = Enum.valueOf((Class<? extends Enum>) quantClass.asSubclass(Enum.class), "Q4_K_M");

        builderClass.getMethod("inputPath", Path.class).invoke(builder, sourceDir);
        builderClass.getMethod("outputPath", Path.class).invoke(builder, outputPath);
        builderClass.getMethod("quantization", quantClass).invoke(builder, quant);
        builderClass.getMethod("overwriteExisting", boolean.class).invoke(builder, false);
        Object params = builderClass.getMethod("build").invoke(builder);

        log.warn("Preparing GGUF fallback cache at {}", outputPath);
        converter.getClass()
                .getMethod("convert", paramsClass, Consumer.class)
                .invoke(converter, params, null);
        return outputPath;
    }


    private String normalizeCompanionGgufName(Path sourceDir) {
        String raw = sourceDir.getFileName() != null ? sourceDir.getFileName().toString() : "model";
        return raw.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private String resolveSafetensorSamplingStrategy(float temperature, int topK, float topP) {
        if (temperature < 1.0e-4f || topK == 1) {
            return "GREEDY";
        }
        boolean hasTopK = topK > 0;
        boolean hasTopP = topP > 0.0f && topP < 1.0f;
        if (hasTopK && hasTopP) {
            return "TOP_K_TOP_P";
        }
        if (hasTopP) {
            return "TOP_P";
        }
        if (hasTopK) {
            return "TOP_K";
        }
        return "GREEDY";
    }

    private float normalizeSafetensorRepeatPenalty(Path modelPath, float requestedRepeatPenalty) {
        if (modelPath == null) {
            return requestedRepeatPenalty;
        }
        String lower = modelPath.toString().toLowerCase(Locale.ROOT);
        if ((lower.contains("gemma-4") || lower.contains("gemma4"))
                && Math.abs(requestedRepeatPenalty - 1.1f) < 1.0e-6f) {
            return 1.0f;
        }
        return requestedRepeatPenalty;
    }

    private String buildSafetensorPrompt(InferenceRequest request) {
        String prompt = request.getPrompt();
        if (prompt != null && !prompt.isBlank()) {
            return prompt;
        }

        List<Message> messages = request.getMessages();
        if (messages == null || messages.isEmpty()) {
            return "";
        }
        if (messages.size() == 1 && messages.getFirst().getContent() != null) {
            return messages.getFirst().getContent().trim();
        }
        if (messages.size() == 2
                && messages.getFirst().getRole() == tech.kayys.gollek.spi.Message.Role.SYSTEM
                && messages.get(1).getRole() == tech.kayys.gollek.spi.Message.Role.USER) {
            String system = messages.getFirst().getContent() == null ? "" : messages.getFirst().getContent().trim();
            String user = messages.get(1).getContent() == null ? "" : messages.get(1).getContent().trim();
            return system.isEmpty() ? user : system + "\n\n" + user;
        }

        // Fall back to Gemma-4 turn formatting for richer conversations.
        StringBuilder formatted = new StringBuilder("<bos>");
        String firstUserPrefix = "";
        int startIdx = 0;
        Message first = messages.getFirst();
        if (first.getRole() == Message.Role.SYSTEM && first.getContent() != null) {
            firstUserPrefix = first.getContent().trim();
            if (!firstUserPrefix.isEmpty()) {
                firstUserPrefix += "\n\n";
            }
            startIdx = 1;
        }

        boolean firstRendered = true;
        for (int i = startIdx; i < messages.size(); i++) {
            Message message = messages.get(i);
            String content = message.getContent();
            if (content == null || content.isBlank()) {
                continue;
            }
            String roleName = message.getRole() == Message.Role.ASSISTANT
                    ? "model"
                    : message.getRole().name().toLowerCase(Locale.ROOT);
            formatted.append("<|turn>").append(roleName).append("\n");
            if (firstRendered && !firstUserPrefix.isEmpty()
                    && message.getRole() == Message.Role.USER) {
                formatted.append(firstUserPrefix);
            }
            formatted.append(content.trim()).append("<turn|>\n");
            firstRendered = false;
        }
        formatted.append("<|turn>model\n");
        return formatted.toString();
    }

    private Path findCompanionSafetensorModel(Path sourcePath) {
        try {
            String fileName = sourcePath.getFileName().toString().toLowerCase(Locale.ROOT);
            String fileHint = fileName
                    .replaceAll("(_qualcomm_[^.]+)?\\.(litertlm|task|tflite|tfl)$", "")
                    .replace("-web", "")
                    .replace("-litert-lm", "")
                    .replace("-litert", "");
            Path parentDir = Files.isDirectory(sourcePath) ? sourcePath : sourcePath.getParent();
            String dirHint = parentDir != null
                    ? parentDir.getFileName().toString().toLowerCase(Locale.ROOT)
                        .replace("-litert-lm", "")
                        .replace("-litert", "")
                    : fileHint;
            List<String> hints = new ArrayList<>();
            if (!fileHint.isBlank()) {
                hints.add(fileHint);
            }
            if (!dirHint.isBlank() && !dirHint.equals(fileHint)) {
                hints.add(dirHint);
            }

            List<Path> roots = Arrays.asList(
                    Path.of(System.getProperty("user.home"), ".gollek", "models", "blobs"),
                    Path.of(System.getProperty("user.home"), ".gollek", "models", "safetensors"));
            for (Path root : roots) {
                if (!Files.isDirectory(root)) {
                    continue;
                }
                try (var stream = Files.walk(root, 5)) {
                    Path found = stream
                            .filter(Files::isRegularFile)
                            .filter(p -> p.getFileName().toString().equals("model.safetensors"))
                            .filter(p -> {
                                String lower = p.toString().toLowerCase(Locale.ROOT);
                                for (String hint : hints) {
                                    if (!hint.isBlank() && lower.contains(hint)) {
                                        return true;
                                    }
                                }
                                return false;
                            })
                            .findFirst()
                            .orElse(null);
                    if (found != null) {
                        return found;
                    }
                }
            }
        } catch (Exception e) {
            log.debug("Failed to locate companion safetensor model for {}: {}", sourcePath, e.getMessage());
        }
        return null;
    }

    private String sanitizeGemma4Text(String content) {
        if (content == null || content.isEmpty()) {
            return content;
        }
        String sanitized = GEMMA4_THOUGHT_CHANNEL.matcher(content).replaceFirst("");
        sanitized = GEMMA4_GENERIC_CHANNEL_OPEN.matcher(sanitized).replaceFirst("");
        sanitized = GEMMA4_ASSISTANT_TURN_OPEN.matcher(sanitized).replaceFirst("");
        sanitized = GEMMA4_PLAIN_ASSISTANT_LINE.matcher(sanitized).replaceFirst("");
        sanitized = sanitized.replace("<channel|>", "");
        sanitized = sanitized.replace("<turn|>", "");
        return sanitized.replace("<|tool_response>", "");
    }

    private String sanitizeGemma4StreamingDelta(
            StreamingInferenceChunk chunk,
            AtomicBoolean leadingChannelsPending) {
        if (chunk == null || chunk.getDelta() == null || chunk.getDelta().isEmpty()) {
            return null;
        }
        String delta = chunk.getDelta();
        if (leadingChannelsPending.get()) {
            String stripped = sanitizeGemma4Text(delta);
            if (stripped.isBlank() || "model".equalsIgnoreCase(stripped.trim())
                    || "assistant".equalsIgnoreCase(stripped.trim())) {
                return "";
            }
            leadingChannelsPending.set(false);
            return stripped;
        }
        return delta.replace("<channel|>", "")
                .replace("<turn|>", "")
                .replace("<|tool_response>", "");
    }

    private void cleanup() {
        try {
            closeOfficialJvmBridge();
            if (gemmaMetalRunner != null) {
                gemmaMetalRunner.close();
                gemmaMetalRunner = null;
            }
            if (gemmaNativeRunner != null) {
                gemmaNativeRunner.close();
                gemmaNativeRunner = null;
            }
            if (compiledModel != null) { bindings.destroyCompiledModel(compiledModel); compiledModel = null; }
            if (options != null) { bindings.destroyOptions(options); options = null; }
            if (model != null) { bindings.destroyModel(model); model = null; }
            if (environment != null) { bindings.destroyEnvironment(environment); environment = null; }
            if (ownsTokenizer && tokenizer != null) {
                tokenizer.close();
                tokenizer = null;
                ownsTokenizer = false;
            }
        } catch (Exception e) {
            log.error("Cleanup failed", e);
        }
    }

    private void closeOfficialJvmBridge() {
        if (officialJvmLmBridge == null) {
            return;
        }
        try {
            officialJvmLmBridge.close();
        } catch (Exception e) {
            log.debug("Ignoring official LiteRT-LM JVM bridge close failure: {}", e.getMessage());
        } finally {
            officialJvmLmBridge = null;
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
