package tech.kayys.gollek.onnx.runner;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;
import tech.kayys.alkhawarizm.spi.model.ModelManifest;
import tech.kayys.alkhawarizm.spi.model.RunnerMetadata;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * ONNX Runtime ModelRunner for Gollek.
 *
 * <p>
 * Runs {@link ModelFormat#ONNX} ({@code .onnx}) models in-process via
 * {@link OnnxRuntimeBinding}, which wraps the ONNX Runtime C API vtable
 * through Java FFM.
 *
 * <h2>Execution Providers</h2>
 * <p>
 * The runner selects hardware acceleration at startup based on the
 * {@code gollek.runners.onnx.execution-provider} config key:
 *
 * <table border="1" cellpadding="4">
 * <tr>
 * <th>EP value</th>
 * <th>Device</th>
 * <th>Library required</th>
 * </tr>
 * <tr>
 * <td>{@code cpu}</td>
 * <td>CPU MLAS</td>
 * <td>always available</td>
 * </tr>
 * <tr>
 * <td>{@code cuda}</td>
 * <td>NVIDIA CUDA</td>
 * <td>{@code libonnxruntime_providers_cuda.so}</td>
 * </tr>
 * <tr>
 * <td>{@code rocm}</td>
 * <td>AMD ROCm/HIP</td>
 * <td>{@code libonnxruntime_providers_rocm.so}</td>
 * </tr>
 * <tr>
 * <td>{@code tensorrt}</td>
 * <td>NVIDIA TensorRT via ORT</td>
 * <td>TensorRT + ORT TRT EP</td>
 * </tr>
 * <tr>
 * <td>{@code coreml}, {@code metal}, {@code mps}</td>
 * <td>Apple ANE/Metal</td>
 * <td>macOS only</td>
 * </tr>
 * <tr>
 * <td>{@code auto}</td>
 * <td>best available</td>
 * <td>tries CoreML → CUDA → ROCm → CPU</td>
 * </tr>
 * </table>
 *
 * <h2>Why ONNX?</h2>
 * <p>
 * ONNX is the universal export format for LLMs. Hugging Face {@code optimum}
 * exports any model to ONNX in one command:
 * 
 * <pre>
 *   optimum-cli export onnx \
 *     --model meta-llama/Llama-3-8B \
 *     --task text-generation \
 *     output/
 * </pre>
 * 
 * Microsoft's Olive toolkit applies INT8/FP16 post-training quantisation and
 * graph optimisation to the resulting {@code .onnx} file.
 *
 * <h2>KV cache in ONNX models</h2>
 * <p>
 * ONNX models exported with {@code --task text-generation-with-past} embed
 * static KV-cache tensors directly in the model graph (Microsoft's
 * {@code past_key_values} pattern). This runner supports both:
 * <ul>
 * <li><b>Stateful (past_key_values)</b> — the model graph carries KV state
 * as named input/output tensors; the runner passes them back on each step.</li>
 * <li><b>Stateless (context window)</b> — full context is fed on every decode
 * step;
 * slower but simpler and compatible with all export tools.</li>
 * </ul>
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.runners.onnx.enabled=false
 *   gollek.runners.onnx.library-path=/usr/lib/libonnxruntime.so
 *   gollek.runners.onnx.execution-provider=auto
 *   gollek.runners.onnx.device-id=0
 *   gollek.runners.onnx.intra-op-threads=4
 *   gollek.runners.onnx.inter-op-threads=1
 *   gollek.runners.onnx.graph-opt-level=99
 *   gollek.runners.onnx.use-past-kv-cache=true
 *   gollek.runners.onnx.vocab-size=32000
 * </pre>
 *
 * <h3>Install ONNX Runtime</h3>
 * 
 * <pre>
 *   # Linux GPU:
 *   pip install onnxruntime-gpu
 *   find ~/.local -name "libonnxruntime.so" 2>/dev/null
 *
 *   # macOS:
 *   brew install onnxruntime
 *
 *   # Or download release from:
 *   # https://github.com/microsoft/onnxruntime/releases
 * </pre>
 */
@ApplicationScoped
public class OnnxRuntimeRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "onnxruntime";
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final OnnxStopTokens DEFAULT_EOS_TOKEN_IDS = OnnxStopTokens.of(2);

    @ConfigProperty(name = "gollek.runners.onnx.enabled", defaultValue = "true")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.onnx.library-path", defaultValue = "/usr/lib/libonnxruntime.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.onnx.execution-provider", defaultValue = "auto")
    String executionProvider; // auto | cpu | cuda | rocm | tensorrt | coreml

    @ConfigProperty(name = "gollek.runners.onnx.device-id", defaultValue = "0")
    int deviceId;

    @ConfigProperty(name = "gollek.runners.onnx.intra-op-threads", defaultValue = "4")
    int intraOpThreads;

    @ConfigProperty(name = "gollek.runners.onnx.inter-op-threads", defaultValue = "1")
    int interOpThreads;

    @ConfigProperty(name = "gollek.runners.onnx.graph-opt-level", defaultValue = "99")
    int graphOptLevel;

    @ConfigProperty(name = "gollek.runners.onnx.use-past-kv-cache", defaultValue = "true")
    boolean usePastKvCache;

    @ConfigProperty(name = "gollek.runners.onnx.vocab-size", defaultValue = "32000")
    int vocabSize;

    @ConfigProperty(name = "gollek.runners.onnx.text-workspace-pool-size", defaultValue = "1")
    int textWorkspacePoolSize;

    @ConfigProperty(name = "gollek.runners.onnx.input-tensor-cache-entries", defaultValue = "32")
    int inputTensorCacheEntries;

    @Inject
    PagedKVCacheManager kvCacheManager;

    private OnnxRuntimeBinding ort;
    private ModelManifest manifest;

    // ORT session objects — created once, reused across requests
    private MemorySegment ortEnv = MemorySegment.NULL;
    private MemorySegment ortSession = MemorySegment.NULL;
    private MemorySegment memInfo = MemorySegment.NULL;

    // Resolved tensor names from the loaded model
    private String[] inputNames = {};
    private String[] outputNames = {};
    private String resolvedEp = "cpu";
    private boolean coreMlAvailable;
    private Tokenizer tokenizer;
    private Path modelDir;
    private int kvLayers = 32;
    private int kvHeads = 2;
    private int kvHeadSize = 128;
    private int kvElementType = OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16;
    private OnnxStopTokens eosTokenIds = DEFAULT_EOS_TOKEN_IDS;
    private Path optimizedModelPath;
    private String optimizedModelCacheState = "disabled";
    private final ThreadLocal<long[]> singleTokenDecodeScratch = ThreadLocal.withInitial(() -> new long[1]);
    private boolean genAiConfigLoaded;
    private String inputIdsName = "input_ids";
    private String attentionMaskName = "attention_mask";
    private String positionIdsName = "position_ids";
    private String logitsName = "logits";
    private String pastKeyNameTemplate = "past_key_values.%d.key";
    private String pastValueNameTemplate = "past_key_values.%d.value";
    private String presentKeyNameTemplate = "present.%d.key";
    private String presentValueNameTemplate = "present.%d.value";
    private OnnxTextSessionResources textSessionResources = OnnxTextSessionResources.empty();
    private OnnxTextWorkspacePool textWorkspacePool;

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "onnxruntime";
    }

    @Override
    public DeviceType deviceType() {
        return resolveDeviceType();
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "1.19.0",
                List.of(ModelFormat.ONNX),
                resolveSupportedDevices(),
                Map.of("execution_provider", executionProvider,
                        "graph_opt_level", String.valueOf(graphOptLevel),
                        "past_kv_cache", String.valueOf(usePastKvCache),
                        "input_tensor_cache_entries", String.valueOf(inputTensorCacheEntries),
                        "project",
                        "https://github.com/microsoft/onnxruntime"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true) // INT8/INT4/FP16 via Olive/ORT quant
                .maxBatchSize(32)
                .supportedDataTypes(new String[] { "fp32", "fp16", "int8", "int4" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        if (config == null) {
            config = new RunnerConfiguration();
        }
        applyStandaloneDefaults(config);
        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "ONNX Runtime runner disabled (gollek.runners.onnx.enabled=false)");

        executionProvider = config.getStringParameter("execution_provider", executionProvider);
        deviceId = config.getIntParameter("device_id", deviceId);
        intraOpThreads = config.getIntParameter("intra_op_threads", intraOpThreads);
        vocabSize = config.getIntParameter("vocab_size", vocabSize);
        textWorkspacePoolSize = config.getIntParameter("text_workspace_pool_size",
                config.getIntParameter("workspace_pool_size", textWorkspacePoolSize));
        textWorkspacePoolSize = Math.max(0, textWorkspacePoolSize);
        inputTensorCacheEntries = config.getIntParameter("input_tensor_cache_entries",
                config.getIntParameter("input_cache_entries", inputTensorCacheEntries));
        inputTensorCacheEntries = Math.max(0, inputTensorCacheEntries);

        // Load the ORT shared library via FFM
        Path resolvedLibraryPath = resolveLibraryPath(libraryPath);
        OnnxRuntimeBinding.initialize(resolvedLibraryPath);
        ort = OnnxRuntimeBinding.getInstance();
        coreMlAvailable = ort.supportsCoreMl();
        if (coreMlAvailable && isAppleSilicon()) {
            log.info("[ONNX] CoreML EP available (Apple Silicon)");
        }

        if (!ort.isNativeAvailable()) {
            log.warn("[ONNX] Native library not available — CPU fallback active");
            this.manifest = modelManifest;
            this.initialized = true;
            return;
        }

        // Resolve model path from manifest
        Path modelPath = modelManifest.artifacts().values().stream()
                .findFirst()
                .map(loc -> {
                    try {
                        String uri = loc.uri();
                        if (uri.startsWith("file:")) {
                            return Path.of(java.net.URI.create(uri));
                        }
                        return Path.of(uri);
                    } catch (Exception e) {
                        return Path.of(loc.uri());
                    }
                })
                .orElseThrow(() -> new RunnerInitializationException(
                        ErrorCode.INIT_NATIVE_LIBRARY_FAILED, "No .onnx artifact in manifest"));

        Path sourceModelPath = modelPath;
        modelDir = Files.isDirectory(sourceModelPath) ? sourceModelPath : sourceModelPath.getParent();
        loadGenAiConfig(modelDir);
        loadTokenizer(modelDir);

        if (!Files.exists(sourceModelPath)) {
            log.errorf("[ONNX] Model file DOES NOT EXIST: %s", sourceModelPath);
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED, "Model file not found: " + sourceModelPath);
        }

        Path sessionModelPath = resolveSessionModelPath(sourceModelPath, config);

        log.infof("[ONNX] Loading model from: %s", sessionModelPath);

        // Create ORT environment
        ortEnv = ort.createEnv("gollek-ort");

        // Configure session options
        MemorySegment opts = ort.createSessionOptions();
        ort.setIntraOpNumThreads(opts, intraOpThreads);
        ort.setInterOpNumThreads(opts, interOpThreads);
        ort.setGraphOptimizationLevel(opts, graphOptLevel);
        if ("write_requested".equals(optimizedModelCacheState) && optimizedModelPath != null) {
            try {
                Files.createDirectories(optimizedModelPath.getParent());
                ort.setOptimizedModelFilePath(opts, optimizedModelPath.toString());
            } catch (Exception e) {
                log.warnf("[ONNX] Optimized model cache disabled for %s: %s", optimizedModelPath, e.getMessage());
                optimizedModelCacheState = "disabled";
                optimizedModelPath = null;
            }
        }

        // Attach execution provider
        resolvedEp = resolveAndAttachEp(opts);

        // Load model
        ortSession = ort.createSession(ortEnv, sessionModelPath.toString(), opts);
        ort.releaseSessionOptions(opts);
        if ("write_requested".equals(optimizedModelCacheState)
                && optimizedModelPath != null
                && Files.isRegularFile(optimizedModelPath)) {
            optimizedModelCacheState = "created";
        }

        // Discover tensor names from loaded model
        int numInputs = (int) ort.getInputCount(ortSession);
        int numOutputs = (int) ort.getOutputCount(ortSession);
        inputNames = new String[numInputs];
        outputNames = new String[numOutputs];
        for (int i = 0; i < numInputs; i++)
            inputNames[i] = ort.getInputName(ortSession, i);
        for (int i = 0; i < numOutputs; i++)
            outputNames[i] = ort.getOutputName(ortSession, i);
        OnnxTextSessionContract textSessionContract = buildTextRuntimePlan();
        OnnxTextSessionResources newTextSessionResources = OnnxTextSessionResources.create(
                ort,
                textSessionContract,
                vocabSize,
                kvHeads,
                kvHeadSize,
                kvElementType);
        // Pre-create CPU memory info (reused for all tensor allocations)
        memInfo = ort.createCpuMemoryInfo();
        OnnxTextWorkspacePool newTextWorkspacePool = OnnxTextWorkspacePool.create(
                ort,
                ortSession,
                memInfo,
                newTextSessionResources,
                textWorkspacePoolSize,
                inputTensorCacheEntries);
        closeTextWorkspacePool();
        closeTextSessionResources();
        textSessionResources = newTextSessionResources;
        textWorkspacePool = newTextWorkspacePool;

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[ONNX] Initialized — model=%s ep=%s inputs=%d outputs=%d vocab=%d",
                modelManifest.modelId(), resolvedEp, numInputs, numOutputs, vocabSize);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode via ONNX Runtime.
     *
     * <p>
     * Two paths based on {@code usePastKvCache}:
     * <ul>
     * <li><b>Stateful (past_key_values)</b> — on each decode step the previous
     * step's output KV-cache tensors are passed back as inputs. Only the new
     * token is fed, giving O(1) memory per step instead of O(ctx_len²).</li>
     * <li><b>Stateless</b> — the full token sequence is fed on every step.
     * Simpler but O(ctx_len²) compute. Works with any ONNX export.</li>
     * </ul>
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        OnnxTextGenerationResult generation = generateText(request, OnnxGeneratedTokenObserver.NOOP, () -> false, null);

        InferenceResponse.Builder response = InferenceResponse.builder()
                .requestId(generation.requestId())
                .content(generation.content())
                .model(manifest.modelId())
                .inputTokens(generation.inputTokens())
                .outputTokens(generation.outputTokens())
                .durationMs(generation.durationMs())
                .finishReason(generation.finishReason().responseReason())
                .metadata("runner", RUNNER_NAME)
                .metadata("finish_reason", generation.finishReason().wireValue());

        if (generation.fallback()) {
            return response
                    .metadata("fallback", true)
                    .metadata(generation.profile().metadata(
                            "fallback",
                            generation.inputTokens(),
                            generation.outputTokens(),
                            generation.durationMs()))
                    .build();
        }

        OnnxTextSessionContract contract = textSessionResources.contract();
        return response
                .metadata("ep", resolvedEp)
                .metadata("past_kv", usePastKvCache)
                .metadata("onnx_genai_config", genAiConfigLoaded)
                .metadata("onnx_logits_name", contract.logitsName())
                .metadata("onnx_kv_layers", contract.kvLayerCount())
                .metadata("onnx_optimized_model_cache_state", optimizedModelCacheState)
                .metadata("onnx_optimized_model_cache_path",
                        optimizedModelPath == null ? null : optimizedModelPath.toString())
                .metadata("prompt_len", generation.inputTokens())
                .metadata("output_len", generation.content().length())
                .metadata(generation.profile().metadata(
                        resolvedEp,
                        generation.inputTokens(),
                        generation.outputTokens(),
                        generation.durationMs()))
                .build();
    }

    private OnnxTextGenerationResult generateText(
            InferenceRequest request,
            OnnxGeneratedTokenObserver tokenObserver,
            BooleanSupplier cancelled,
            Supplier<String> finalContentSupplier) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "ONNX Runtime runner not initialized");

        long t0 = System.currentTimeMillis();
        OnnxInferenceProfile profile = OnnxInferenceProfile.start(request);
        OnnxTextGenerationSetup setup = OnnxTextGenerationSetup.prepare(
                request,
                tokenObserver,
                cancelled,
                profile,
                this::getMaxTokens,
                this::tokenize,
                () -> OnnxStreamingTokenDecoder.create(tokenizer, this::detokenize),
                finalContentSupplier);
        totalRequests.incrementAndGet();

        if (setup.maxTokens() == 0) {
            OnnxTextGenerationResult result = OnnxTextZeroTokenGeneration.finish(setup, t0);
            totalLatencyMs.addAndGet(result.durationMs());
            return result;
        }

        if (!ort.isNativeAvailable()) {
            return OnnxTextCpuFallbackGeneration.create(vocabSize, this::sampleGreedy, this::detokenize)
                    .generate(setup, t0);
        }

        try {
            int[] prompt = setup.prompt();
            int maxTok = setup.maxTokens();
            String content;
            int outputTokenCount;
            OnnxTextGenerationProgress progress;

            OnnxTextWorkspacePool.Lease lease = textWorkspacePool.acquire(prompt.length, maxTok);
            try (lease) {
                OnnxTextRunWorkspace workspace = lease.workspace();
                OnnxTokenHistory tokenIds = workspace.resetTokenHistory(prompt, maxTok);
                OnnxGeneratedTokens generatedTokenIds = workspace.resetGeneratedTokens(maxTok);
                OnnxTextTokenAcceptor tokenAcceptor = OnnxTextTokenAcceptor.create(
                        generatedTokenIds,
                        tokenIds,
                        setup.observer(),
                        setup.stopStringDecoder(),
                        setup.stopStrings(),
                        this::isEos);
                progress = new OnnxTextGenerationProgress();
                for (int step = 0; step < maxTok; step++) {
                    if (setup.cancellation().getAsBoolean()) {
                        progress.cancel();
                        break;
                    }
                    OnnxTextDecodeStep decodeStep = workspace.planDecodeStep(
                            prompt.length,
                            tokenIds.size(),
                            progress.consumedTokens());

                    long prepareStart = profile.mark();
                    long seqLen = decodeStep.sequenceLength();
                    OnnxRunInputValues inputValues = workspace.assembleInputs(
                            tokenIds,
                            prompt.length,
                            decodeStep);

                    int next;
                    try {
                        workspace.execute(inputValues, profile, prepareStart, decodeStep.prefill());

                        // Select from last-token logits without materializing the full
                        // vocabulary row as a Java float array.
                        long logitsStart = profile.mark();
                        next = workspace.selectNextToken(seqLen);
                        profile.recordLogitsSelect(logitsStart);
                    } finally {
                        workspace.releaseOwnedInputs(inputValues);
                    }

                    workspace.capturePresentAndReleaseLogits();
                    progress.advance(seqLen);

                    long samplingStart = profile.mark();
                    profile.markFirstToken();
                    OnnxTextTokenDecision tokenDecision = tokenAcceptor.accept(next);
                    boolean finished = progress.apply(tokenDecision);
                    profile.recordSampling(samplingStart);
                    if (finished)
                        break;
                }
                profile.recordInputTensorCache(workspace.inputTensorCacheStats());
                long finalDecodeStart = profile.mark();
                content = finalContent(setup, generatedTokenIds);
                profile.recordFinalDecode(finalDecodeStart);
                outputTokenCount = generatedTokenIds.size();
            }
            profile.recordWorkspaceLease(
                    lease.reused(),
                    lease.evicted(),
                    lease.requestedCapacity(),
                    lease.workspaceCapacity());

            long dur = System.currentTimeMillis() - t0;
            totalLatencyMs.addAndGet(dur);

            return new OnnxTextGenerationResult(
                    setup.requestId(),
                    content,
                    setup.promptLength(),
                    outputTokenCount,
                    dur,
                    profile,
                    false,
                    progress.finishReason());

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[ONNX] " + e.getMessage(), e);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            try {
                OnnxStreamingTextChunks chunks = OnnxStreamingTextChunks.create(
                        request.getRequestId(),
                        OnnxStreamingTokenDecoder.create(tokenizer, this::detokenize),
                        emitter::emit,
                        emitter::isCancelled);
                OnnxGeneratedTokenObserver observer = new OnnxGeneratedTokenObserver() {
                    @Override
                    public String onToken(int tokenId, int tokenIndex) {
                        return onToken(tokenId, tokenIndex, true);
                    }

                    @Override
                    public String onToken(int tokenId, int tokenIndex, boolean currentTextNeeded) {
                        chunks.emitToken(tokenId);
                        return currentTextNeeded ? chunks.currentText() : null;
                    }
                };
                OnnxTextGenerationResult generation = generateText(request,
                        observer,
                        emitter::isCancelled,
                        chunks::currentText);
                if (chunks.emitFinal(generation)) {
                    emitter.complete();
                }
            } catch (Exception e) {
                if (!emitter.isCancelled()) {
                    emitter.fail(e);
                }
            }
        });
    }

    @Override
    protected int[] tokenize(InferenceRequest request) {
        if (tokenizer == null) {
            return super.tokenize(request);
        }
        String promptText = request == null ? null : request.getPrompt();
        if (promptText == null || promptText.isBlank()) {
            promptText = request == null || request.getMessages() == null
                    ? ""
                    : request.getMessages().stream()
                            .map(message -> message.getContent() == null ? "" : message.getContent())
                            .reduce("", (left, right) -> left + right);
        }
        long[] ids = tokenizer.encode(promptText, EncodeOptions.defaultOptions());
        if (ids.length == 0) {
            return new int[] { tokenizer.padTokenId() >= 0 ? tokenizer.padTokenId() : 0 };
        }
        int[] values = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            values[i] = Math.toIntExact(ids[i]);
        }
        return values;
    }

    @Override
    protected String detokenize(int tokenId) {
        if (tokenizer == null) {
            return super.detokenize(tokenId);
        }
        long[] scratch = singleTokenDecodeScratch.get();
        scratch[0] = tokenId;
        return tokenizer.decode(scratch, 0, 1, DecodeOptions.defaultOptions());
    }

    private String decodeGenerated(OnnxGeneratedTokens generatedTokens) {
        if (generatedTokens == null || generatedTokens.isEmpty()) {
            return "";
        }
        if (tokenizer == null) {
            return generatedTokens.decodeEach(this::detokenize);
        }
        return generatedTokens.decodeWith(tokenizer, DecodeOptions.defaultOptions());
    }

    private String finalContent(OnnxTextGenerationSetup setup, OnnxGeneratedTokens generatedTokens) {
        String observed = setup.finalContentOrNull();
        if (observed != null) {
            return observed;
        }
        return decodeGenerated(generatedTokens);
    }

    @Override
    protected boolean isEos(int tokenId) {
        return eosTokenIds.contains(tokenId);
    }

    private void applyStandaloneDefaults(RunnerConfiguration config) {
        if (config == null) {
            config = new RunnerConfiguration();
        }
        if (libraryPath == null || libraryPath.isBlank()) {
            libraryPath = "/usr/lib/libonnxruntime.so";
        }
        if (executionProvider == null || executionProvider.isBlank()) {
            executionProvider = "auto";
        }
        if (intraOpThreads <= 0) {
            intraOpThreads = 4;
        }
        if (interOpThreads <= 0) {
            interOpThreads = 1;
        }
        if (graphOptLevel <= 0) {
            graphOptLevel = OnnxRuntimeBinding.GRAPH_OPT_LEVEL_ENABLE_ALL;
        }
        if (!config.getParameters().containsKey("enabled")) {
            enabled = true;
        }
        if (!config.getParameters().containsKey("use_past_kv_cache")) {
            usePastKvCache = true;
        }
    }

    private Path resolveSessionModelPath(Path sourceModelPath, RunnerConfiguration config)
            throws RunnerInitializationException {
        optimizedModelPath = null;
        optimizedModelCacheState = "disabled";
        if (config == null) {
            return sourceModelPath;
        }
        String cachePath = config.getStringParameter("optimized_model_file_path",
                config.getStringParameter("optimized_model_path", null));
        if (cachePath == null || cachePath.isBlank()) {
            return sourceModelPath;
        }
        optimizedModelPath = Path.of(cachePath).toAbsolutePath().normalize();
        if (Files.isRegularFile(optimizedModelPath)) {
            optimizedModelCacheState = "hit";
            return optimizedModelPath;
        }
        Path parent = optimizedModelPath.getParent();
        if (parent == null) {
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "Optimized ONNX cache path has no parent directory: " + optimizedModelPath);
        }
        optimizedModelCacheState = "write_requested";
        return sourceModelPath;
    }

    private void loadGenAiConfig(Path dir) {
        if (dir == null) {
            return;
        }
        Path configPath = dir.resolve("genai_config.json");
        if (!Files.isRegularFile(configPath)) {
            return;
        }
        try {
            JsonNode root = JSON.readTree(configPath.toFile());
            JsonNode model = root.path("model");
            JsonNode decoder = model.path("decoder");
            JsonNode decoderInputs = decoder.path("inputs");
            JsonNode decoderOutputs = decoder.path("outputs");
            vocabSize = positiveInt(model.path("vocab_size"), vocabSize);
            kvLayers = positiveInt(decoder.path("num_hidden_layers"), kvLayers);
            kvHeads = positiveInt(decoder.path("num_key_value_heads"), kvHeads);
            kvHeadSize = positiveInt(decoder.path("head_size"), kvHeadSize);
            inputIdsName = text(decoderInputs.path("input_ids"), inputIdsName);
            attentionMaskName = text(decoderInputs.path("attention_mask"), attentionMaskName);
            positionIdsName = text(decoderInputs.path("position_ids"), positionIdsName);
            pastKeyNameTemplate = text(decoderInputs.path("past_key_names"), pastKeyNameTemplate);
            pastValueNameTemplate = text(decoderInputs.path("past_value_names"), pastValueNameTemplate);
            logitsName = text(decoderOutputs.path("logits"), logitsName);
            presentKeyNameTemplate = text(decoderOutputs.path("present_key_names"), presentKeyNameTemplate);
            presentValueNameTemplate = text(decoderOutputs.path("present_value_names"), presentValueNameTemplate);
            OnnxStopTokens.Builder stops = defaultStopTokenBuilder();
            collectInts(model.path("eos_token_id"), stops);
            collectInts(model.path("pad_token_id"), stops);
            eosTokenIds = stops.build();
            genAiConfigLoaded = true;
        } catch (Exception e) {
            log.debugf(e, "[ONNX] Failed to read genai_config.json from %s", configPath);
        }
    }

    private static OnnxStopTokens.Builder defaultStopTokenBuilder() {
        return OnnxStopTokens.builder(DEFAULT_EOS_TOKEN_IDS);
    }

    private void loadTokenizer(Path dir) {
        if (dir == null) {
            return;
        }
        try {
            tokenizer = TokenizerFactory.load(dir, null);
            if (tokenizer.vocabSize() > 0) {
                vocabSize = tokenizer.vocabSize();
            }
            OnnxStopTokens.Builder stops = OnnxStopTokens.builder(eosTokenIds);
            if (tokenizer.eosTokenId() >= 0) {
                stops.add(tokenizer.eosTokenId());
            }
            if (tokenizer.padTokenId() >= 0) {
                stops.add(tokenizer.padTokenId());
            }
            stops.addAll(tokenizer.allStopTokenIds());
            eosTokenIds = stops.build();
            log.infof("[ONNX] Loaded tokenizer from %s (vocab=%d)", dir, tokenizer.vocabSize());
        } catch (Exception e) {
            tokenizer = null;
            log.debugf(e, "[ONNX] No tokenizer loaded from %s; using fallback token stubs", dir);
        }
    }

    private static int positiveInt(JsonNode node, int fallback) {
        if (node == null || !node.isNumber()) {
            return fallback;
        }
        int value = node.asInt();
        return value > 0 ? value : fallback;
    }

    private static String text(JsonNode node, String fallback) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return fallback;
        }
        String value = node.asText(null);
        return value == null || value.isBlank() ? fallback : value.trim();
    }

    private static void collectInts(JsonNode node, OnnxStopTokens.Builder target) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return;
        }
        if (node.isArray()) {
            for (JsonNode child : node) {
                collectInts(child, target);
            }
        } else if (node.isNumber()) {
            target.add(node.asInt());
        }
    }

    private OnnxTextSessionContract buildTextRuntimePlan() {
        return OnnxTextSessionContract.create(
                inputNames,
                outputNames,
                new OnnxTextSessionContract.TensorNames(
                        inputIdsName,
                        attentionMaskName,
                        positionIdsName,
                        logitsName,
                        pastKeyNameTemplate,
                        pastValueNameTemplate,
                        presentKeyNameTemplate,
                        presentValueNameTemplate),
                usePastKvCache,
                kvLayers);
    }

    // ── EP resolution ─────────────────────────────────────────────────────────

    private String resolveAndAttachEp(MemorySegment opts) {
        switch (executionProvider.toLowerCase()) {
            case "cuda" -> {
                ort.appendCudaProvider(opts, deviceId);
                return "CUDAExecutionProvider";
            }
            case "rocm" -> {
                ort.appendRocmProvider(opts, deviceId);
                return "ROCMExecutionProvider";
            }
            case "coreml" -> {
                if (appendCoreMlIfAvailable(opts)) {
                    return "CoreMLExecutionProvider";
                }
                return "CPUExecutionProvider (coreml-unavailable)";
            }
            case "metal", "mps", "apple", "apple-metal", "coreml-metal" -> {
                if (appendCoreMlIfAvailable(opts)) {
                    return "CoreMLExecutionProvider (metal)";
                }
                return "CPUExecutionProvider (metal-unavailable)";
            }
            case "cpu" -> {
                return "CPUExecutionProvider";
            }
            default -> {
                // auto: try CoreML (Apple Silicon) → CUDA → ROCm → CPU
                if (appendCoreMlIfAvailable(opts)) {
                    return "CoreMLExecutionProvider (auto)";
                }
                try {
                    ort.appendCudaProvider(opts, deviceId);
                    return "CUDAExecutionProvider (auto)";
                } catch (Exception e1) {
                    try {
                        ort.appendRocmProvider(opts, deviceId);
                        return "ROCMExecutionProvider (auto)";
                    } catch (Exception e2) {
                        return "CPUExecutionProvider (auto)";
                    }
                }
            }
        }
    }

    private boolean appendCoreMlIfAvailable(MemorySegment opts) {
        if (!isAppleSilicon()) {
            return false;
        }
        if (!metalAllowedByGlobalConfig()) {
            return false;
        }
        if (!coreMlAvailable) {
            return false;
        }
        return ort.appendCoreMlProvider(opts, 0);
    }

    private List<DeviceType> resolveSupportedDevices() {
        java.util.LinkedHashSet<DeviceType> devices = new java.util.LinkedHashSet<>();
        devices.add(DeviceType.CPU);
        devices.add(DeviceType.CUDA);
        devices.add(DeviceType.ROCM);
        if (coreMlAvailable && isAppleSilicon()) {
            devices.add(DeviceType.METAL);
        }
        return java.util.List.copyOf(devices);
    }

    private boolean isAppleSilicon() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        if (!osName.contains("mac")) {
            return false;
        }
        String arch = System.getProperty("os.arch", "").toLowerCase();
        return arch.contains("aarch64") || arch.contains("arm64");
    }

    private Path resolveLibraryPath(String configuredPath) {
        Path configured = Path.of(configuredPath);
        if (Files.exists(configured) && Files.isReadable(configured)) {
            return configured;
        }

        // Check environment variable override
        String envPath = System.getenv("GOLLEK_ONNX_LIB_PATH");
        if (envPath != null && !envPath.isBlank()) {
            Path envLibPath = Path.of(envPath);
            if (Files.exists(envLibPath) && Files.isReadable(envLibPath)) {
                log.infof("[ONNX] Using library from GOLLEK_ONNX_LIB_PATH: %s", envLibPath);
                return envLibPath;
            }
        }

        String osName = System.getProperty("os.name", "").toLowerCase();
        boolean isMac = osName.contains("mac");
        String libName = isMac ? "libonnxruntime.dylib" : "libonnxruntime.so";

        // Standard Gollek native library location: ~/.gollek/libs/onnxruntime/
        Path standardLibPath = Path.of(System.getProperty("user.home"), ".gollek", "libs", "onnxruntime", libName);
        if (Files.exists(standardLibPath)) {
            log.infof("[ONNX] Using standard library path: %s", standardLibPath);
            return standardLibPath;
        }

        // Legacy fallback: ~/.gollek/libs/libonnxruntime.dylib
        Path fallback = Path.of(System.getProperty("user.home"), ".gollek", "libs", libName);
        if (Files.exists(fallback)) {
            log.infof("[ONNX] Using fallback library path %s", fallback);
            return fallback;
        }

        return configured;
    }

    private boolean metalAllowedByGlobalConfig() {
        String metalEnabled = firstNonBlank(
                System.getProperty("gollek.runners.metal.enabled"),
                System.getenv("GOLLEK_METAL_ENABLED"));
        if (metalEnabled != null && metalEnabled.equalsIgnoreCase("false")) {
            return false;
        }
        String metalMode = firstNonBlank(
                System.getProperty("gollek.runners.metal.mode"),
                System.getenv("GOLLEK_METAL_MODE"));
        return metalMode == null || !metalMode.equalsIgnoreCase("disabled");
    }

    private String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return null;
    }

    private DeviceType resolveDeviceType() {
        String ep = executionProvider.toLowerCase();
        if (ep.contains("cuda") || ep.equals("tensorrt"))
            return DeviceType.CUDA;
        if (ep.contains("rocm"))
            return DeviceType.ROCM;
        if (ep.contains("coreml")
                || ep.equals("metal")
                || ep.equals("mps")
                || ep.equals("apple")
                || ep.equals("apple-metal"))
            return DeviceType.METAL;
        return DeviceType.CPU;
    }

    // ── Health & close ────────────────────────────────────────────────────────

    @Override
    public boolean health() {
        return initialized;
    }

    @Override
    public void close() {
        initialized = false;
        closeTextWorkspacePool();
        closeTextSessionResources();
        if (ort != null && ort.isNativeAvailable()) {
            if (!memInfo.equals(MemorySegment.NULL))
                ort.releaseMemoryInfo(memInfo);
            if (!ortSession.equals(MemorySegment.NULL))
                ort.releaseSession(ortSession);
            if (!ortEnv.equals(MemorySegment.NULL))
                ort.releaseEnv(ortEnv);
        }
        log.info("[ONNX] Closed");
    }

    private void closeTextWorkspacePool() {
        if (textWorkspacePool == null) {
            return;
        }
        textWorkspacePool.close();
        textWorkspacePool = null;
    }

    private void closeTextSessionResources() {
        if (textSessionResources == null) {
            return;
        }
        textSessionResources.close();
        textSessionResources = OnnxTextSessionResources.empty();
    }
}
