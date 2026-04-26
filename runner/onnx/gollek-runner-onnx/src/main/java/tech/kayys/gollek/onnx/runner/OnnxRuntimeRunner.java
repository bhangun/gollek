package tech.kayys.gollek.onnx.runner;

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
import tech.kayys.gollek.onnx.binding.OnnxRuntimeCpuFallback;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.RunnerMetadata;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
 * <td>{@code coreml}</td>
 * <td>Apple ANE/Metal</td>
 * <td>macOS only</td>
 * </tr>
 * <tr>
 * <td>{@code auto}</td>
 * <td>best available</td>
 * <td>tries CUDA → ROCm → CPU</td>
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

        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "ONNX Runtime runner disabled (gollek.runners.onnx.enabled=false)");

        executionProvider = config.getStringParameter("execution_provider", executionProvider);
        deviceId = config.getIntParameter("device_id", deviceId);
        intraOpThreads = config.getIntParameter("intra_op_threads", intraOpThreads);
        vocabSize = config.getIntParameter("vocab_size", vocabSize);

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

        if (!Files.exists(modelPath)) {
            log.errorf("[ONNX] Model file DOES NOT EXIST: %s", modelPath);
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED, "Model file not found: " + modelPath);
        }

        System.err.println("[OnnxRuntimeRunner] Loading model from: " + modelPath);

        // Create ORT environment
        ortEnv = ort.createEnv("gollek-ort");

        // Configure session options
        MemorySegment opts = ort.createSessionOptions();
        ort.setIntraOpNumThreads(opts, intraOpThreads);
        ort.setInterOpNumThreads(opts, interOpThreads);
        ort.setGraphOptimizationLevel(opts, graphOptLevel);

        // Attach execution provider
        resolvedEp = resolveAndAttachEp(opts);

        // Load model
        System.err.println("[OnnxRuntimeRunner] Creating ORT session...");
        ortSession = ort.createSession(ortEnv, modelPath.toString(), opts);
        System.err.println("[OnnxRuntimeRunner] ORT session created successfully: " + ortSession);
        ort.releaseSessionOptions(opts);

        // Discover tensor names from loaded model
        int numInputs = (int) ort.getInputCount(ortSession);
        int numOutputs = (int) ort.getOutputCount(ortSession);
        inputNames = new String[numInputs];
        outputNames = new String[numOutputs];
        for (int i = 0; i < numInputs; i++)
            inputNames[i] = ort.getInputName(ortSession, i);
        for (int i = 0; i < numOutputs; i++)
            outputNames[i] = ort.getOutputName(ortSession, i);

        // Pre-create CPU memory info (reused for all tensor allocations)
        memInfo = ort.createCpuMemoryInfo();

        this.manifest = modelManifest;
        this.initialized = true;

        System.err.println("[OnnxRuntimeRunner] Initialization complete. Set initialized = true");
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
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "ONNX Runtime runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTok = getMaxTokens(request);
        int[] prompt = tokenize(request);
        totalRequests.incrementAndGet();

        if (!ort.isNativeAvailable()) {
            // CPU fallback: return dummy response
            float[] logits = OnnxRuntimeCpuFallback.run(MemorySegment.NULL, vocabSize);
            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(detokenize(sampleGreedy(logits)))
                    .model(manifest.modelId())
                    .durationMs(System.currentTimeMillis() - t0)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("fallback", true)
                    .build();
        }

        try (Arena arena = Arena.ofAuto()) {
            StringBuilder sb = new StringBuilder();
            List<Integer> tokenIds = new ArrayList<>();
            for (int t : prompt)
                tokenIds.add(t);

            // Past KV cache tensors from previous steps (null on first step)
            MemorySegment[] pastKv = null;
            String[] pastKvNames = buildPastKvInputNames();
            String[] presentNames = buildPresentKvOutputNames();

            for (int step = 0; step <= maxTok; step++) {
                boolean isPrefill = (step == 0);

                // Build input_ids tensor — shape [1, seqLen]
                int[] inputIds = isPrefill
                        ? prompt
                        : new int[] { tokenIds.get(tokenIds.size() - 1) };
                long seqLen = inputIds.length;

                MemorySegment idsSeg = arena.allocate((long) inputIds.length * 4L, 4);
                for (int i = 0; i < inputIds.length; i++)
                    idsSeg.setAtIndex(ValueLayout.JAVA_INT, i, inputIds[i]);

                // Convert to int64 (ONNX expects int64 for token ids)
                MemorySegment ids64 = arena.allocate((long) inputIds.length * 8L, 8);
                for (int i = 0; i < inputIds.length; i++)
                    ids64.setAtIndex(ValueLayout.JAVA_LONG, i, inputIds[i]);

                MemorySegment idsValue = ort.createTensorWithData(
                        memInfo, ids64,
                        new long[] { 1, seqLen },
                        OnnxRuntimeBinding.ONNX_TENSOR_INT64);

                // Attention mask tensor — shape [1, seqLen], all ones
                MemorySegment maskSeg = arena.allocate(seqLen * 8L, 8);
                for (long i = 0; i < seqLen; i++)
                    maskSeg.setAtIndex(ValueLayout.JAVA_LONG, i, 1L);
                MemorySegment maskValue = ort.createTensorWithData(
                        memInfo, maskSeg,
                        new long[] { 1, seqLen },
                        OnnxRuntimeBinding.ONNX_TENSOR_INT64);

                // Assemble full input list (ids + mask + optional past KV)
                List<String> allInputNames = new ArrayList<>(List.of("input_ids", "attention_mask"));
                List<MemorySegment> allInputVals = new ArrayList<>(List.of(idsValue, maskValue));

                if (usePastKvCache && pastKv != null && pastKvNames.length > 0) {
                    for (int k = 0; k < pastKvNames.length; k++) {
                        allInputNames.add(pastKvNames[k]);
                        allInputVals.add(pastKv[k]);
                    }
                }

                // Resolve output names: logits + present KV cache
                String[] runOutputNames = usePastKvCache && presentNames.length > 0
                        ? concatArrays(new String[] { "logits" }, presentNames)
                        : new String[] { "logits" };

                // Run inference
                MemorySegment[] outputs = ort.run(
                        ortSession, MemorySegment.NULL,
                        allInputNames.toArray(String[]::new),
                        allInputVals.toArray(MemorySegment[]::new),
                        runOutputNames);

                // Extract logits from output[0] — shape [1, seqLen, vocabSize]
                // For decode, only the last token's logits matter
                float[] logits = ort.getTensorDataFloat(outputs[0], vocabSize);

                // Release input tensors (outputs will be reused as past KV)
                ort.releaseValue(idsValue);
                ort.releaseValue(maskValue);

                // Save present KV cache for next step
                if (usePastKvCache && outputs.length > 1) {
                    if (pastKv != null)
                        for (MemorySegment v : pastKv)
                            ort.releaseValue(v);
                    pastKv = new MemorySegment[outputs.length - 1];
                    System.arraycopy(outputs, 1, pastKv, 0, pastKv.length);
                }
                ort.releaseValue(outputs[0]);

                if (isPrefill)
                    continue; // prefill done, start decode

                int next = sampleGreedy(logits);
                if (isEos(next))
                    break;
                sb.append(detokenize(next));
                tokenIds.add(next);
            }

            // Release remaining past KV tensors
            if (pastKv != null)
                for (MemorySegment v : pastKv)
                    ort.releaseValue(v);

            long dur = System.currentTimeMillis() - t0;
            totalLatencyMs.addAndGet(dur);

            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(sb.toString())
                    .model(manifest.modelId())
                    .durationMs(dur)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("ep", resolvedEp)
                    .metadata("past_kv", usePastKvCache)
                    .metadata("prompt_len", prompt.length)
                    .metadata("output_len", sb.length())
                    .build();

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
                InferenceResponse r = infer(request);
                emitter.emit(new StreamingInferenceChunk(
                        request.getRequestId(),
                        0,
                        tech.kayys.gollek.spi.model.ModalityType.TEXT,
                        RUNNER_NAME,
                        r.getContent(),
                        true,
                        "stop",
                        null,
                        java.time.Instant.now(),
                        null));
                emitter.complete();
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
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
        if (ep.contains("coreml"))
            return DeviceType.METAL;
        return DeviceType.CPU;
    }

    // ── KV cache tensor naming ────────────────────────────────────────────────

    /**
     * Build the standard past_key_values input names used by HuggingFace
     * ONNX exports: {@code past_key_values.0.key}, {@code past_key_values.0.value},
     * …
     * for 32 layers.
     */
    private String[] buildPastKvInputNames() {
        int layers = 0;
        for (String name : inputNames) {
            if (name.startsWith("past_key_values.") && name.endsWith(".key")) {
                layers++;
            }
        }
        if (layers == 0) layers = 32; // Fallback if no KV inputs found (e.g. non-causal LM)
        String[] names = new String[layers * 2];
        for (int l = 0; l < layers; l++) {
            names[l * 2] = "past_key_values." + l + ".key";
            names[l * 2 + 1] = "past_key_values." + l + ".value";
        }
        return names;
    }

    private String[] buildPresentKvOutputNames() {
        int layers = 0;
        for (String name : inputNames) {
            if (name.startsWith("past_key_values.") && name.endsWith(".key")) {
                layers++;
            }
        }
        if (layers == 0) layers = 32;
        String[] names = new String[layers * 2];
        for (int l = 0; l < layers; l++) {
            names[l * 2] = "present." + l + ".key";
            names[l * 2 + 1] = "present." + l + ".value";
        }
        return names;
    }

    private String[] concatArrays(String[] a, String[] b) {
        String[] result = new String[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }

    // ── Health & close ────────────────────────────────────────────────────────

    @Override
    public boolean health() {
        return initialized;
    }

    @Override
    public void close() {
        initialized = false;
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
}
