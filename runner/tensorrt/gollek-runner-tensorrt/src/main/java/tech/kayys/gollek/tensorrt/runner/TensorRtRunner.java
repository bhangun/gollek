package tech.kayys.gollek.tensorrt.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.tensorrt.binding.TensorRtBinding;
import tech.kayys.gollek.tensorrt.binding.TensorRtCpuFallback;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;

import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.model.RunnerMetadata;

import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * TensorRT ModelRunner for Gollek.
 *
 * <p>
 * Runs pre-compiled TensorRT {@code .engine} / {@code .trt} files
 * ({@link ModelFormat#TENSORRT}) in-process via {@link TensorRtBinding},
 * which wraps the TRT 10 C API ({@code nvinfer_c.h}) through Java FFM.
 *
 * <h2>Why TensorRT?</h2>
 * <p>
 * TensorRT compiles an ONNX or custom model graph for a specific GPU at
 * build time, producing an optimised {@code .engine} binary that:
 * <ul>
 * <li>Fuses adjacent operators into single CUDA kernels (e.g. LayerNorm +
 * GEMM + Activation in one pass)</li>
 * <li>Auto-tunes kernel tile sizes and memory layouts for the target GPU</li>
 * <li>Applies INT8/FP8/FP16 quantisation with built-in calibration</li>
 * <li>Uses FlashAttention-style fused attention via {@code cuDNN 9.x}</li>
 * </ul>
 * Result: 2–4× lower latency vs ONNX Runtime CUDA EP for INT8 batch=1;
 * up to 8× throughput for large-batch FP16.
 *
 * <h2>Engine lifecycle</h2>
 * <p>
 * A TRT engine is compiled <em>once</em> (offline via {@code trtexec} or
 * the TRT Python API) and saved to disk. This runner deserialises it at
 * startup. Engine files are <em>GPU-specific</em> — an engine compiled for
 * an A100 will not run on an H100 without recompilation.
 *
 * <h2>Dynamic shapes (LLM decode)</h2>
 * <p>
 * For autoregressive decode the engine must be compiled with dynamic input
 * shapes:
 * 
 * <pre>
 *   trtexec --onnx=model.onnx \
 *           --saveEngine=model.engine \
 *           --fp16 \
 *           --minShapes=input_ids:1x1,attention_mask:1x1 \
 *           --optShapes=input_ids:1x512,attention_mask:1x512 \
 *           --maxShapes=input_ids:1x2048,attention_mask:1x2048
 * </pre>
 * 
 * At decode time this runner sets the actual input shape via
 * {@code IExecutionContext::setInputShape} (accessed through the tensor
 * address API) before each {@link TensorRtBinding#enqueueV3} call.
 *
 * <h2>KV cache</h2>
 * <p>
 * TRT engines compiled with {@code --plugin} support the KV-cache plugin
 * ({@code TrtLLM} / {@code tensorrt_llm}). This runner also supports the
 * simpler static-context approach: the full token sequence is fed on each step,
 * controlled by {@code gollek.runners.tensorrt.stateless-context}.
 * For paged KV cache integration with Gollek's {@link PagedKVCacheManager}
 * use the CUDA runner or the ONNX runner with past_key_values.
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.runners.tensorrt.enabled=false
 *   gollek.runners.tensorrt.library-path=/usr/lib/x86_64-linux-gnu/libnvinfer_10.so
 *   gollek.runners.tensorrt.engine-path=/models/model.engine
 *   gollek.runners.tensorrt.vocab-size=32000
 *   gollek.runners.tensorrt.max-seq-len=2048
 *   gollek.runners.tensorrt.input-tensor=input_ids
 *   gollek.runners.tensorrt.output-tensor=logits
 *   gollek.runners.tensorrt.stateless-context=false
 * </pre>
 *
 * <h3>Build the engine</h3>
 * 
 * <pre>
 *   # FP16:
 *   trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 \
 *           --minShapes=input_ids:1x1 --optShapes=input_ids:1x512 \
 *           --maxShapes=input_ids:1x2048
 *
 *   # INT8 with calibration data:
 *   trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8 \
 *           --calib=calib_cache.bin
 * </pre>
 */
@ApplicationScoped
public class TensorRtRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "tensorrt";

    @ConfigProperty(name = "gollek.runners.tensorrt.enabled", defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.tensorrt.library-path", defaultValue = "/usr/lib/x86_64-linux-gnu/libnvinfer_10.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.tensorrt.engine-path", defaultValue = "")
    String enginePath;

    @ConfigProperty(name = "gollek.runners.tensorrt.vocab-size", defaultValue = "32000")
    int vocabSize;

    @ConfigProperty(name = "gollek.runners.tensorrt.max-seq-len", defaultValue = "2048")
    int maxSeqLen;

    @ConfigProperty(name = "gollek.runners.tensorrt.input-tensor", defaultValue = "input_ids")
    String inputTensorName;

    @ConfigProperty(name = "gollek.runners.tensorrt.output-tensor", defaultValue = "logits")
    String outputTensorName;

    @ConfigProperty(name = "gollek.runners.tensorrt.stateless-context", defaultValue = "false")
    boolean statelessContext;

    @Inject
    PagedKVCacheManager kvCacheManager;

    private TensorRtBinding trt;
    private ModelManifest manifest;

    // TRT session objects — created once at startup
    private MemorySegment trtRuntime = MemorySegment.NULL;
    private MemorySegment trtEngine = MemorySegment.NULL;
    private MemorySegment trtCtx = MemorySegment.NULL;

    // CUDA device buffers allocated via cudaMalloc (or Arena.ofShared for UVM)
    private MemorySegment deviceInputBuf = MemorySegment.NULL;
    private MemorySegment deviceOutputBuf = MemorySegment.NULL;
    private Arena deviceArena;

    // Discovered tensor metadata
    private int nbIOTensors = 0;
    private long inputBufBytes = 0;
    private long outputBufBytes = 0;

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "tensorrt";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA;
    }

    @Override
    public RunnerMetadata metadata() {
        return new RunnerMetadata(RUNNER_NAME, "10.0",
                List.of(ModelFormat.TENSORRT),
                List.of(DeviceType.CUDA),
                Map.of("trt_version", "10.x",
                        "formats", "FP32/FP16/BF16/INT8/FP8",
                        "dynamic_shapes", "true",
                        "kv_cache_plugin", "optional (TrtLLM)",
                        "project",
                        "https://developer.nvidia.com/tensorrt"));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true)
                .maxBatchSize(128)
                .supportedDataTypes(new String[] { "fp32", "fp16", "bf16", "int8", "fp8" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "TensorRT runner disabled (gollek.runners.tensorrt.enabled=false)");

        vocabSize = config.getIntParameter("vocab_size", vocabSize);
        maxSeqLen = config.getIntParameter("max_seq_len", maxSeqLen);
        inputTensorName = config.getStringParameter("input_tensor", inputTensorName);
        outputTensorName = config.getStringParameter("output_tensor", outputTensorName);

        TensorRtBinding.initialize(Path.of(libraryPath));
        trt = TensorRtBinding.getInstance();

        if (!trt.isNativeAvailable()) {
            log.warn("[TRT] Native library not available — CPU fallback active");
            this.manifest = modelManifest;
            this.initialized = true;
            return;
        }

        // Resolve engine path: prefer RunnerConfiguration override, then manifest
        // artifact
        String ep = config.getStringParameter("engine_path", enginePath);
        if (ep.isBlank()) {
            ep = modelManifest.artifacts().values().stream()
                    .findFirst()
                    .map(loc -> loc.uri())
                    .orElseThrow(() -> new RunnerInitializationException(
                            ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                            "No .engine artifact in manifest and gollek.runners.tensorrt.engine-path not set"));
        }
        final String resolvedEnginePath = ep;

        // Create TRT runtime
        trtRuntime = trt.createRuntime();
        if (trtRuntime.equals(MemorySegment.NULL))
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED, "TRT createRuntime returned NULL");

        // Load and deserialise engine from disk — mmap for zero-copy
        deviceArena = Arena.ofShared();
        MemorySegment engineBlob = mmapEngine(Path.of(resolvedEnginePath), deviceArena);
        trtEngine = trt.deserializeEngineSegment(trtRuntime, engineBlob);
        if (trtEngine.equals(MemorySegment.NULL))
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "TRT deserializeEngine returned NULL for: " + resolvedEnginePath);

        // Create execution context
        trtCtx = trt.createExecutionContext(trtEngine);
        if (trtCtx.equals(MemorySegment.NULL))
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED, "TRT createExecutionContext returned NULL");

        // Discover IO tensors
        nbIOTensors = trt.getNbIOTensors(trtEngine);
        logTensorInfo();

        // Allocate device buffers via Arena.ofShared (CUDA UVM path — zero copy on
        // systems with HMM; on CUDA-only systems these would be cuMemAlloc'd).
        // Input: [1, maxSeqLen] int64 → maxSeqLen * 8 bytes
        // Output: [1, 1, vocabSize] float32 → vocabSize * 4 bytes
        inputBufBytes = (long) maxSeqLen * 8L;
        outputBufBytes = (long) vocabSize * 4L;
        deviceInputBuf = deviceArena.allocate(inputBufBytes, 16);
        deviceOutputBuf = deviceArena.allocate(outputBufBytes, 16);

        // Pre-bind IO tensor addresses to execution context
        trt.setTensorAddress(trtCtx, inputTensorName, deviceInputBuf);
        trt.setTensorAddress(trtCtx, outputTensorName, deviceOutputBuf);

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[TRT] Ready — engine=%s tensors=%d vocab=%d maxSeq=%d",
                Path.of(resolvedEnginePath).getFileName(),
                nbIOTensors, vocabSize, maxSeqLen);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode via TensorRT {@code enqueueV3}.
     *
     * <p>
     * Flow per token:
     * <ol>
     * <li>Write the current token sequence as int64 into the pinned
     * input buffer {@code deviceInputBuf}.</li>
     * <li>Call {@link TensorRtBinding#enqueueV3} — TRT executes fused
     * kernels on the default CUDA stream (stream = NULL).</li>
     * <li>Read the logits from {@code deviceOutputBuf}. On UVM-capable GPUs
     * this is a direct CPU read without {@code cudaMemcpy}.</li>
     * <li>Sample the next token and repeat.</li>
     * </ol>
     *
     * <p>
     * When {@code statelessContext=false} (default), the engine must be
     * compiled with {@code past_key_values} tensors to avoid O(T²) compute
     * per step. See class-level Javadoc for {@code trtexec} flags.
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "TensorRT runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTok = getMaxTokens(request);
        int[] prompt = tokenize(request);
        totalRequests.incrementAndGet();

        // CPU fallback path
        if (!trt.isNativeAvailable()) {
            float[] logits = TensorRtCpuFallback.run(vocabSize);
            long dur = System.currentTimeMillis() - t0;
            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(detokenize(sampleGreedy(logits)))
                    .model(manifest.modelId())
                    .durationMs(dur)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("fallback", true)
                    .build();
        }

        try {
            List<Integer> tokenIds = new ArrayList<>();
            for (int t : prompt)
                tokenIds.add(t);
            StringBuilder sb = new StringBuilder();

            // Prefill — run once for the full prompt
            enqueueStep(tokenIds);

            // Decode loop
            for (int step = 0; step < maxTok; step++) {
                // Read logits for the last token position
                int vocabSlice = Math.min(vocabSize, (int) (deviceOutputBuf.byteSize() / 4L));
                float[] logits = new float[vocabSlice];
                for (int i = 0; i < vocabSlice; i++)
                    logits[i] = deviceOutputBuf.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);

                int next = sampleGreedy(logits);
                if (isEos(next))
                    break;
                sb.append(detokenize(next));
                tokenIds.add(next);

                // Feed the new token (or full context if stateless)
                if (statelessContext) {
                    enqueueStep(tokenIds);
                } else {
                    enqueueStep(List.of(next)); // decode: single new token
                }
            }

            long dur = System.currentTimeMillis() - t0;
            totalLatencyMs.addAndGet(dur);

            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(sb.toString())
                    .model(manifest.modelId())
                    .durationMs(dur)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("prompt_tokens", prompt.length)
                    .metadata("output_tokens", sb.length())
                    .metadata("nb_io_tensors", nbIOTensors)
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[TRT] " + e.getMessage(), e);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            String reqId = request.getRequestId();
            int maxTok = getMaxTokens(request);
            int[] prompt = tokenize(request);
            int seq = 0;

            if (!trt.isNativeAvailable()) {
                float[] logits = TensorRtCpuFallback.run(vocabSize);
                emitter.emit(StreamingInferenceChunk.finalChunk(reqId, seq, detokenize(sampleGreedy(logits))));
                emitter.complete();
                return;
            }

            try {
                List<Integer> tokenIds = new ArrayList<>();
                for (int t : prompt)
                    tokenIds.add(t);
                enqueueStep(tokenIds);

                for (int step = 0; step < maxTok; step++) {
                    int vocabSlice = Math.min(vocabSize, (int) (deviceOutputBuf.byteSize() / 4L));
                    float[] logits = new float[vocabSlice];
                    for (int i = 0; i < vocabSlice; i++)
                        logits[i] = deviceOutputBuf.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i);

                    int next = sampleGreedy(logits);
                    boolean fin = isEos(next) || step == maxTok - 1;

                    if (fin) {
                        emitter.emit(StreamingInferenceChunk.finalChunk(reqId, seq++, detokenize(next)));
                    } else {
                        emitter.emit(StreamingInferenceChunk.of(reqId, seq++, detokenize(next)));
                    }

                    if (fin)
                        break;
                    tokenIds.add(next);
                    enqueueStep(statelessContext ? tokenIds : List.of(next));
                }
                emitter.complete();
            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    // ── TRT execution step ────────────────────────────────────────────────────

    /**
     * Write token ids into the device input buffer, then call
     * {@link TensorRtBinding#enqueueV3} on the default CUDA stream.
     *
     * <p>
     * On CUDA UVM / HMM-capable systems the CPU writes to the device
     * buffer directly (page-fault mechanism migrates pages to GPU before
     * the kernel reads them). On PCIe-discrete GPUs the programmer must
     * {@code cudaMemcpy} first — this runner uses Arena.ofShared() which
     * maps to pinned host memory and is accessible by the GPU as
     * {@code cudaMemcpyHostToDevice} staging.
     */
    private void enqueueStep(List<Integer> tokenIds) {
        if (tokenIds.size() > maxSeqLen) {
            throw new IllegalArgumentException("Input sequence length " + tokenIds.size() + " exceeds maximum allowed " + maxSeqLen);
        }
        int seqLen = tokenIds.size();
        for (int i = 0; i < seqLen; i++)
            deviceInputBuf.setAtIndex(ValueLayout.JAVA_LONG, i, tokenIds.get(i));

        boolean ok = trt.enqueueV3(trtCtx, MemorySegment.NULL); // NULL = default stream
        if (!ok)
            throw new RuntimeException("TRT enqueueV3 returned false");
        // Note: for async execution synchronise the CUDA stream here.
        // With NULL stream TRT runs synchronously on the calling thread.
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void logTensorInfo() {
        log.infof("[TRT] Engine has %d IO tensors:", nbIOTensors);
        for (int i = 0; i < nbIOTensors; i++) {
            String name = trt.getIOTensorName(trtEngine, i);
            int mode = trt.getTensorIOMode(trtEngine, name);
            long[] dims = trt.getTensorDimensions(trtEngine, name);
            long bpc = trt.getBytesPerComponent(trtEngine, name);
            String modeStr = (mode == TensorRtBinding.TRT_IO_MODE_INPUT) ? "INPUT" : "OUTPUT";
            log.infof("[TRT]   [%d] %-20s %s  shape=%s  bpc=%d",
                    i, name, modeStr, java.util.Arrays.toString(dims), bpc);
        }
    }

    private MemorySegment mmapEngine(Path engineFile, Arena arena) {
        try (RandomAccessFile raf = new RandomAccessFile(engineFile.toFile(), "r");
                FileChannel ch = raf.getChannel()) {
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            log.infof("[TRT] Engine mmap'd: %s (%.1f MB)",
                    engineFile.getFileName(), ch.size() / 1e6);
            return seg;
        } catch (Exception e) {
            throw new RuntimeException("Cannot mmap TRT engine: " + engineFile, e);
        }
    }

    @Override
    public boolean health() {
        return initialized;
    }

    @Override
    public void close() {
        initialized = false;
        if (trt != null && trt.isNativeAvailable()) {
            try { trt.destroyExecutionContext(trtCtx); } catch (Exception e) { log.warn("[TRT] Failed to destroy context", e); }
            try { trt.destroyEngine(trtEngine); } catch (Exception e) { log.warn("[TRT] Failed to destroy engine", e); }
            try { trt.destroyRuntime(trtRuntime); } catch (Exception e) { log.warn("[TRT] Failed to destroy runtime", e); }
        }
        if (deviceArena != null)
            try {
                deviceArena.close();
            } catch (Exception ignored) {
            }
        log.info("[TRT] Closed");
    }
}
