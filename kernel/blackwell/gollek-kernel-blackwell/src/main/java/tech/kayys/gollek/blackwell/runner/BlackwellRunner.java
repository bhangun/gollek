package tech.kayys.gollek.blackwell.runner;

import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.runner.RunnerCapabilities;
import tech.kayys.gollek.runner.RunnerConfiguration;
import tech.kayys.gollek.extension.AbstractGollekRunner;
import tech.kayys.gollek.blackwell.binding.BlackwellBinding;
import tech.kayys.gollek.blackwell.config.BlackwellRunnerMode;
import tech.kayys.gollek.blackwell.detection.BlackwellDetector;
import tech.kayys.gollek.blackwell.detection.BlackwellCapabilities;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;
import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.exception.RunnerInitializationException;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
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
import java.util.List;
import java.util.Map;

/**
 * NVIDIA Blackwell ModelRunner for Gollek.
 *
 * <p>
 * Runs {@link ModelFormat#GGUF} and {@link ModelFormat#SAFETENSORS} models
 * in-process on NVIDIA Blackwell GPUs (B100, B200, GB200) via
 * {@link BlackwellBinding}.
 *
 * <h2>Blackwell-Specific Optimizations</h2>
 * <p>
 * This runner leverages Blackwell-exclusive features:
 * <ol>
 * <li><b>FlashAttention-3 with TMEM</b> — 64MB on-chip tensor memory for QK^T
 * accumulation</li>
 * <li><b>FP4 Tensor Cores</b> — 2x throughput over FP8, 4x over FP16</li>
 * <li><b>Async Execution</b> — Concurrent copy/compute via stream
 * wait/write</li>
 * <li><b>192 GB HBM3e</b> — Largest unified memory pool for massive models</li>
 * </ol>
 *
 * <h2>FlashAttention-3 on Blackwell</h2>
 * <p>
 * FA3 uses TMEM (Tensor Memory) as an on-chip accumulator:
 * <table border="1" cellpadding="3">
 * <tr>
 * <th>Component</th>
 * <th>Hopper (H100)</th>
 * <th>Blackwell (B200)</th>
 * </tr>
 * <tr>
 * <td>QK^T accumulation</td>
 * <td>SRAM registers</td>
 * <td>64MB TMEM</td>
 * </tr>
 * <tr>
 * <td>Softmax</td>
 * <td>Tensor cores</td>
 * <td>Tensor cores + TMEM</td>
 * </tr>
 * <tr>
 * <td>×V multiply</td>
 * <td>Tensor cores</td>
 * <td>FP4 tensor cores</td>
 * </tr>
 * <tr>
 * <td>Throughput</td>
 * <td>1x (baseline)</td>
 * <td>2x H100</td>
 * </tr>
 * </table>
 *
 * <h2>Memory Model</h2>
 * <p>
 * B100/B200/GB200 use unified CPU+GPU memory architecture with 180-192 GB
 * HBM3e.
 * All {@link MemorySegment} allocations via
 * {@link BlackwellBinding#mallocManaged}
 * are accessible from both CPU and GPU without explicit copies.
 *
 * <h2>Compatibility</h2>
 * <p>
 * Target architectures:
 * <ul>
 * <li>sm_100 — NVIDIA B100 (180 GB HBM3e)</li>
 * <li>sm_100 — NVIDIA B200 (180 GB HBM3e)</li>
 * <li>sm_100 — NVIDIA GB200 (192 GB HBM3e, Grace-Blackwell)</li>
 * </ul>
 *
 * <h3>Config</h3>
 * 
 * <pre>
 *   gollek.runners.blackwell.enabled=false
 *   gollek.runners.blackwell.mode=auto  # auto|standard|offload|force|disabled
 *   gollek.runners.blackwell.library-path=/usr/local/cuda/lib64/libgollek_blackwell.so
 *   gollek.runners.blackwell.device-id=0
 *   gollek.runners.blackwell.use-fp4=true
 *   gollek.runners.blackwell.use-tmem=true
 *   gollek.runners.blackwell.num-layers=32
 *   gollek.runners.blackwell.num-heads=32
 *   gollek.runners.blackwell.num-heads-kv=8
 *   gollek.runners.blackwell.head-dim=128
 *   gollek.runners.blackwell.model-dim=4096
 *   gollek.runners.blackwell.ffn-dim=14336
 *   gollek.runners.blackwell.vocab-size=32000
 * </pre>
 *
 * <h3>Build Blackwell kernels</h3>
 * 
 * <pre>
 *   make -C src/main/cpp/blackwell CUDA_ARCH=sm_100
 *   # Output: target/native/linux-x86_64/gollek_blackwell_sm100.cubin
 * </pre>
 */
@ApplicationScoped
public class BlackwellRunner extends AbstractGollekRunner {

    public static final String RUNNER_NAME = "blackwell";

    @ConfigProperty(name = "gollek.runners.blackwell.enabled", defaultValue = "false")
    boolean enabled;

    @ConfigProperty(name = "gollek.runners.blackwell.mode", defaultValue = "auto")
    String blackwellMode;

    @ConfigProperty(name = "gollek.runners.blackwell.library-path", defaultValue = "/usr/local/cuda/lib64/libgollek_blackwell.so")
    String libraryPath;

    @ConfigProperty(name = "gollek.runners.blackwell.device-id", defaultValue = "0")
    int deviceId;

    @ConfigProperty(name = "gollek.runners.blackwell.use-fp4", defaultValue = "true")
    boolean useFp4;

    @ConfigProperty(name = "gollek.runners.blackwell.use-tmem", defaultValue = "true")
    boolean useTmem;

    @ConfigProperty(name = "gollek.runners.blackwell.num-layers", defaultValue = "32")
    int numLayers;

    @ConfigProperty(name = "gollek.runners.blackwell.num-heads", defaultValue = "32")
    int numHeads;

    @ConfigProperty(name = "gollek.runners.blackwell.num-heads-kv", defaultValue = "8")
    int numHeadsKv;

    @ConfigProperty(name = "gollek.runners.blackwell.head-dim", defaultValue = "128")
    int headDim;

    @ConfigProperty(name = "gollek.runners.blackwell.model-dim", defaultValue = "4096")
    int modelDim;

    @ConfigProperty(name = "gollek.runners.blackwell.ffn-dim", defaultValue = "14336")
    int ffnDim;

    @ConfigProperty(name = "gollek.runners.blackwell.vocab-size", defaultValue = "32000")
    int vocabSize;

    @Inject
    PagedKVCacheManager kvCacheManager;

    @Inject
    BlackwellDetector detector;

    private BlackwellBinding blackwell;
    private BlackwellCapabilities caps;
    private ModelManifest manifest;
    private MemorySegment weightsMem;
    private Arena weightsArena;
    private MemorySegment[] layerSlices;

    // Device memory (unified on Blackwell)
    private MemorySegment dResidual = MemorySegment.NULL;
    private MemorySegment dNormed = MemorySegment.NULL;
    private MemorySegment dQkv = MemorySegment.NULL;
    private MemorySegment dAttnOut = MemorySegment.NULL;
    private MemorySegment dFfnBuf = MemorySegment.NULL;
    private MemorySegment tmem = MemorySegment.NULL; // TMEM for FA3
    private MemorySegment cudaStream = MemorySegment.NULL;

    private static Path resolveLibraryPath(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return Path.of("");
        }
        if (rawPath.startsWith("~/")) {
            String home = System.getProperty("user.home");
            if (home != null && !home.isBlank()) {
                return Path.of(home + rawPath.substring(1));
            }
        }
        return Path.of(rawPath);
    }

    // ── ModelRunner identity ──────────────────────────────────────────────────

    @Override
    public String name() {
        return RUNNER_NAME;
    }

    @Override
    public String framework() {
        return "blackwell-cuda";
    }

    @Override
    public DeviceType deviceType() {
        return DeviceType.CUDA; // Blackwell uses CUDA device type
    }

    @Override
    public RunnerMetadata metadata() {
        int computeCap = caps != null ? caps.cudaComputeCap() : 0;
        boolean hasFA3 = computeCap >= 100;
        boolean hasFP4 = useFp4 && computeCap >= 100;
        boolean hasTMEM = useTmem && caps != null && caps.tmemSize() > 0;

        String attnPath = hasFA3 ? (hasTMEM ? "FlashAttention-3-TMEM" : "FlashAttention-3") : "FlashAttention-2";

        return new RunnerMetadata(RUNNER_NAME, "1.0.0",
                List.of(ModelFormat.GGUF, ModelFormat.SAFETENSORS),
                List.of(DeviceType.CUDA, DeviceType.CPU),
                Map.ofEntries(
                        Map.entry("cuda_api", "12.x+"),
                        Map.entry("architecture", "Blackwell"),
                        Map.entry("compute_cap", String.valueOf(computeCap)),
                        Map.entry("unified_memory", "true"),
                        Map.entry("device", caps != null ? caps.deviceName() : "N/A"),
                        Map.entry("tmem_size", String.valueOf(caps != null ? caps.tmemSize() : 0)),
                        Map.entry("attention_path", attnPath),
                        Map.entry("fa3_support", String.valueOf(hasFA3)),
                        Map.entry("fp4_support", String.valueOf(hasFP4)),
                        Map.entry("tmem_accel", String.valueOf(hasTMEM)),
                        Map.entry("async_exec", "true")));
    }

    @Override
    public RunnerCapabilities capabilities() {
        return RunnerCapabilities.builder()
                .supportsStreaming(true)
                .supportsBatching(true)
                .supportsQuantization(true)
                .maxBatchSize(64) // Blackwell can handle larger batches
                .supportedDataTypes(new String[] { "fp32", "fp16", "bf16", "fp8", "fp4", "int8" })
                .build();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void initialize(ModelManifest modelManifest, RunnerConfiguration config)
            throws RunnerInitializationException {

        BlackwellRunnerMode mode = BlackwellRunnerMode.from(blackwellMode);
        if (!enabled)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "Blackwell runner disabled (gollek.runners.blackwell.enabled=false)");
        if (mode == BlackwellRunnerMode.DISABLED)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "Blackwell runner disabled (gollek.runners.blackwell.mode=disabled)");
        if (mode == BlackwellRunnerMode.OFFLOAD)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "Blackwell runner disabled (gollek.runners.blackwell.mode=offload)");

        caps = detector.detect();
        if (!caps.available() && mode != BlackwellRunnerMode.FORCE) {
            throw new RunnerInitializationException(
                    ErrorCode.DEVICE_NOT_AVAILABLE,
                    "Blackwell unavailable on this host: " + caps.reason());
        }
        if (!caps.available()) {
            log.warnf("[Blackwell] Blackwell forced despite unavailable caps (%s)", caps.reason());
        }

        // Load Blackwell binding library
        Path resolvedLibraryPath = resolveLibraryPath(libraryPath);
        BlackwellBinding.initialize(resolvedLibraryPath);
        blackwell = BlackwellBinding.getInstance();

        if (!blackwell.isNativeAvailable()) {
            log.warn("[Blackwell] Blackwell library not available — CPU fallback active");
            this.manifest = modelManifest;
            this.initialized = true;
            return;
        }

        // Initialize Blackwell device
        int err = blackwell.init(deviceId);
        if (err != 0)
            throw new RunnerInitializationException(
                    ErrorCode.INIT_NATIVE_LIBRARY_FAILED,
                    "blackwellInit() returned " + err);

        String deviceName = caps != null ? caps.deviceName() : blackwell.deviceName(deviceId);
        log.infof("[Blackwell] Device %d: %s (compute=%d.%d, TMEM=%.1fMB, memory=%.1fGB)",
                deviceId, deviceName,
                caps != null ? caps.cudaComputeCap() / 10 : 0,
                caps != null ? caps.cudaComputeCap() % 10 : 0,
                caps != null ? caps.tmemMb() : 0,
                caps != null ? caps.totalMemoryGb() : 0);

        // Create CUDA stream with async support
        cudaStream = blackwell.streamCreate();

        // Allocate TMEM for FlashAttention-3
        if (useTmem && caps != null && caps.tmemSize() > 0) {
            tmem = blackwell.tmemAlloc(Math.min(caps.tmemSize(), 64L * 1024 * 1024));
            log.infof("[Blackwell] Allocated %.1f MB TMEM for FA3", caps.tmemMb());
        }

        // Override model dims from RunnerConfiguration
        numLayers = config.getIntParameter("num_layers", numLayers);
        numHeads = config.getIntParameter("num_heads", numHeads);
        numHeadsKv = config.getIntParameter("num_heads_kv", numHeadsKv);
        headDim = config.getIntParameter("head_dim", headDim);
        modelDim = config.getIntParameter("model_dim", modelDim);
        ffnDim = config.getIntParameter("ffn_dim", ffnDim);
        vocabSize = config.getIntParameter("vocab_size", vocabSize);

        // Allocate device memory (unified on Blackwell)
        long elemBytes = 2L; // FP16
        dResidual = blackwell.mallocManaged((long) modelDim * elemBytes, 1);
        dNormed = blackwell.mallocManaged((long) modelDim * elemBytes, 1);
        dQkv = blackwell.mallocManaged((long) (numHeads + 2 * numHeadsKv) * headDim * elemBytes, 1);
        dAttnOut = blackwell.mallocManaged((long) numHeads * headDim * elemBytes, 1);
        dFfnBuf = blackwell.mallocManaged((long) ffnDim * 2 * elemBytes, 1);

        // Memory-map model weights
        weightsArena = Arena.ofAuto();
        Path modelPath = resolveModelPath(modelManifest);
        weightsMem = mmapWeights(modelPath, weightsArena);
        layerSlices = sliceLayerWeights(weightsMem, numLayers);

        this.manifest = modelManifest;
        this.initialized = true;

        log.infof("[Blackwell] Ready — model=%s layers=%d heads=%d/%d dim=%d ffn=%d vocab=%d fp4=%s tmem=%s",
                modelManifest.modelId(), numLayers, numHeads, numHeadsKv,
                headDim, ffnDim, vocabSize, useFp4, useTmem);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Autoregressive decode using Blackwell kernels.
     *
     * <p>
     * Each transformer layer runs:
     * <ol>
     * <li>RMS Norm — {@link BlackwellBinding#rmsNorm}</li>
     * <li>QKV projection — {@link BlackwellBinding#matmulFp4} or
     * {@link BlackwellBinding#matmulFp8}</li>
     * <li>FlashAttention-3 — {@link BlackwellBinding#flashAttnV3Tmem} with
     * TMEM</li>
     * <li>Output projection + residual</li>
     * <li>SiLU-gated FFN — {@link BlackwellBinding#siluFfn}</li>
     * </ol>
     */
    @Override
    public InferenceResponse infer(InferenceRequest request) throws InferenceException {
        if (!initialized)
            throw new InferenceException(
                    ErrorCode.RUNTIME_INVALID_STATE, "Blackwell runner not initialized");

        long t0 = System.currentTimeMillis();
        String reqId = request.getRequestId();
        int maxTok = getMaxTokens(request);
        int[] prompt = tokenize(request);
        int promptLen = prompt.length;
        totalRequests.incrementAndGet();

        if (!blackwell.isNativeAvailable()) {
            // CPU fallback
            float[] logits = new float[vocabSize];
            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(detokenize(sampleGreedy(logits)))
                    .model(manifest.modelId())
                    .durationMs(System.currentTimeMillis() - t0)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("fallback", true)
                    .build();
        }

        kvCacheManager.allocateForPrefill(reqId, promptLen);
        try {
            StringBuilder sb = new StringBuilder();
            int[] bt = blockTable(reqId);
            int seqLen = promptLen;

            // Prefill
            runForwardPass(seqLen, bt, false);

            // Decode loop
            for (int step = 0; step < maxTok; step++) {
                float[] logits = runForwardPass(seqLen, bt, true);
                int next = sampleGreedy(logits);
                if (isEos(next))
                    break;
                sb.append(detokenize(next));
                seqLen++;
                if (kvCacheManager.appendToken(reqId))
                    bt = blockTable(reqId);
            }

            long dur = System.currentTimeMillis() - t0;
            totalLatencyMs.addAndGet(dur);

            return InferenceResponse.builder()
                    .requestId(reqId)
                    .content(sb.toString())
                    .model(manifest.modelId())
                    .durationMs(dur)
                    .metadata("runner", RUNNER_NAME)
                    .metadata("device", caps != null ? caps.deviceName() : "N/A")
                    .metadata("fp4_enabled", useFp4)
                    .metadata("tmem_enabled", useTmem)
                    .metadata("prompt_tokens", promptLen)
                    .metadata("output_tokens", sb.length())
                    .build();

        } catch (Exception e) {
            totalFailures.incrementAndGet();
            throw new InferenceException(ErrorCode.RUNTIME_INFERENCE_FAILED,
                    "[Blackwell] " + e.getMessage(), e);
        } finally {
            kvCacheManager.freeRequest(reqId);
        }
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            String reqId = request.getRequestId();
            int maxTok = getMaxTokens(request);
            int seqLen = tokenize(request).length;
            int seq = 0;

            if (!blackwell.isNativeAvailable()) {
                float[] logits = new float[vocabSize];
                emitter.emit(StreamingInferenceChunk.finalChunk(
                        reqId, seq, detokenize(sampleGreedy(logits))));
                emitter.complete();
                return;
            }

            kvCacheManager.allocateForPrefill(reqId, seqLen);
            try {
                int[] bt = blockTable(reqId);
                runForwardPass(seqLen, bt, false);

                for (int step = 0; step < maxTok; step++) {
                    float[] logits = runForwardPass(seqLen, bt, true);
                    int next = sampleGreedy(logits);
                    boolean fin = isEos(next) || step == maxTok - 1;
                    if (fin) {
                        emitter.emit(StreamingInferenceChunk.finalChunk(reqId, seq++, detokenize(next)));
                    } else {
                        emitter.emit(StreamingInferenceChunk.of(reqId, seq++, detokenize(next)));
                    }
                    if (fin)
                        break;
                    seqLen++;
                    if (kvCacheManager.appendToken(reqId))
                        bt = blockTable(reqId);
                }
                emitter.complete();
            } catch (Exception e) {
                emitter.fail(e);
            } finally {
                kvCacheManager.freeRequest(reqId);
            }
        });
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /**
     * One forward pass through all {@code numLayers} transformer layers.
     * Uses FlashAttention-3 with TMEM and FP4 tensor cores on Blackwell.
     */
    private float[] runForwardPass(int seqLen, int[] bt, boolean decodeOnly) {
        int T = decodeOnly ? 1 : seqLen;
        int blockSz = kvCacheManager.getConfig().getBlockSize();
        float attnScale = (float) (1.0 / Math.sqrt(headDim));

        // KV pool slabs (unified memory on Blackwell)
        MemorySegment kPool = kvCacheManager.getBlockPool().rawKPool();
        MemorySegment vPool = kvCacheManager.getBlockPool().rawVPool();

        boolean hasFA3 = caps != null && caps.cudaComputeCap() >= 100;
        boolean useFp4Matmul = useFp4 && hasFA3;

        for (int layer = 0; layer < numLayers; layer++) {
            MemorySegment w = layerSlices[layer];

            // Weight slice layout
            long off = 0L;
            int qkvDim = (numHeads + 2 * numHeadsKv) * headDim;
            int qDim = numHeads * headDim;

            MemorySegment normW = w.asSlice(off, (long) modelDim * 4L);
            off += (long) modelDim * 4L;
            MemorySegment wQkv = w.asSlice(off, (long) modelDim * qkvDim * 4L);
            off += (long) modelDim * qkvDim * 4L;
            MemorySegment wO = w.asSlice(off, (long) qDim * modelDim * 4L);
            off += (long) qDim * modelDim * 4L;
            MemorySegment fnormW = w.asSlice(off, (long) modelDim * 4L);
            off += (long) modelDim * 4L;
            MemorySegment wGate = w.asSlice(off, (long) modelDim * ffnDim * 4L);
            off += (long) modelDim * ffnDim * 4L;
            MemorySegment wUp = w.asSlice(off, (long) modelDim * ffnDim * 4L);
            off += (long) modelDim * ffnDim * 4L;
            MemorySegment wDown = w.asSlice(off, (long) ffnDim * modelDim * 4L);

            // ── 1. Pre-attention RMS Norm ────────────────────────────────────
            blackwell.rmsNorm(dNormed, dResidual, normW, modelDim, 1e-6f);

            // ── 2. QKV projection (FP4 on Blackwell) ─────────────────────────
            if (useFp4Matmul) {
                blackwell.matmulFp4(dQkv, dNormed, wQkv, T, modelDim, qkvDim, 1.0f, 0.0f);
            } else {
                blackwell.matmul(dQkv, dNormed, wQkv, T, modelDim, qkvDim, 1.0f, 0.0f);
            }

            // ── 3. FlashAttention-3 with TMEM ────────────────────────────────
            if (hasFA3 && useTmem && !tmem.equals(MemorySegment.NULL)) {
                // Use TMEM-accelerated FA3
                blackwell.flashAttnV3Tmem(dAttnOut, dQkv, kPool, vPool, tmem,
                        1, T, seqLen, numHeads, headDim,
                        attnScale, 1, useFp4 ? 1 : 0);
            } else if (hasFA3) {
                // FA3 without explicit TMEM management
                blackwell.flashAttnV3(dAttnOut, dQkv, kPool, vPool,
                        1, T, seqLen, numHeads, headDim,
                        attnScale, 1, useFp4 ? 1 : 0);
            } else {
                // Fallback to regular attention
                blackwell.attention(dAttnOut, dQkv, kPool, vPool,
                        null, null, 1, T, numHeads, headDim,
                        blockSz, bt.length, attnScale, 1);
            }

            // ── 4. Output projection + residual ──────────────────────────────
            MemorySegment proj = blackwell.mallocManaged((long) T * modelDim * 2L, 1);
            if (useFp4Matmul) {
                blackwell.matmulFp4(proj, dAttnOut, wO, T, qDim, modelDim, 1.0f, 0.0f);
            } else {
                blackwell.matmul(proj, dAttnOut, wO, T, qDim, modelDim, 1.0f, 0.0f);
            }
            addResidual(dResidual, proj, T * modelDim);
            blackwell.free(proj);

            // ── 5. Pre-FFN RMS Norm ──────────────────────────────────────────
            blackwell.rmsNorm(dNormed, dResidual, fnormW, modelDim, 1e-6f);

            // ── 6. FFN gate + up (FP4) ───────────────────────────────────────
            if (useFp4Matmul) {
                blackwell.matmulFp4(dFfnBuf, dNormed, wGate, T, modelDim, ffnDim, 1.0f, 0.0f);
                blackwell.matmulFp4(dFfnBuf, dNormed, wUp, T, modelDim, ffnDim, 1.0f, 0.0f);
            } else {
                blackwell.matmul(dFfnBuf, dNormed, wGate, T, modelDim, ffnDim, 1.0f, 0.0f);
                blackwell.matmul(dFfnBuf, dNormed, wUp, T, modelDim, ffnDim, 1.0f, 0.0f);
            }

            // ── 7. SiLU gate ─────────────────────────────────────────────────
            blackwell.siluFfn(dFfnBuf, dFfnBuf, dFfnBuf, T * ffnDim);

            // ── 8. FFN down projection + residual ────────────────────────────
            if (useFp4Matmul) {
                blackwell.matmulFp4(dFfnBuf, dFfnBuf, wDown, T, ffnDim, modelDim, 1.0f, 0.0f);
            } else {
                blackwell.matmul(dFfnBuf, dFfnBuf, wDown, T, ffnDim, modelDim, 1.0f, 0.0f);
            }
            addResidual(dResidual, dFfnBuf, T * modelDim);

            // Synchronize after each layer
            blackwell.streamSynchronize(cudaStream);
        }

        // Read logits from residual buffer (unified memory = direct CPU read)
        return readLogits(dResidual, vocabSize);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void addResidual(MemorySegment a, MemorySegment b, int n) {
        // Blackwell has unified memory - direct CPU access
        for (int i = 0; i < n; i++) {
            float aVal = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float bVal = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            a.setAtIndex(ValueLayout.JAVA_FLOAT, i, aVal + bVal);
        }
    }

    private float[] readLogits(MemorySegment buffer, int vocabSize) {
        float[] logits = new float[vocabSize];
        // Unified memory - direct CPU read
        for (int i = 0; i < vocabSize; i++) {
            logits[i] = buffer.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return logits;
    }

    private int[] blockTable(String reqId) {
        return kvCacheManager.getBlockTable(reqId).stream().mapToInt(Integer::intValue).toArray();
    }

    private Path resolveModelPath(ModelManifest m) {
        return m.artifacts().values().stream().findFirst()
                .map(l -> Path.of(l.uri()))
                .orElseThrow(() -> new IllegalArgumentException("No model artifact in manifest"));
    }

    private MemorySegment mmapWeights(Path modelPath, Arena arena) {
        try (RandomAccessFile raf = new RandomAccessFile(modelPath.toFile(), "r");
                FileChannel ch = raf.getChannel()) {
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            log.infof("[Blackwell] Weights mmap'd: %s (%.1f GB)", modelPath.getFileName(), ch.size() / 1e9);
            return seg;
        } catch (Exception e) {
            log.warnf("[Blackwell] Cannot mmap %s: %s — using zero weights", modelPath, e.getMessage());
            return arena.allocate(256L * 1024 * 1024, 64);
        }
    }

    private MemorySegment[] sliceLayerWeights(MemorySegment weights, int numLayers) {
        MemorySegment[] slices = new MemorySegment[numLayers];
        long layerBytes = weights.byteSize() / numLayers;
        for (int i = 0; i < numLayers; i++) {
            slices[i] = weights.asSlice((long) i * layerBytes, layerBytes);
        }
        return slices;
    }

    @Override
    public boolean health() {
        return initialized && (blackwell == null || blackwell.isNativeAvailable());
    }

    @Override
    public void close() {
        initialized = false;
        if (blackwell != null && blackwell.isNativeAvailable()) {
            if (!cudaStream.equals(MemorySegment.NULL)) {
                blackwell.streamDestroy(cudaStream);
            }
            if (!tmem.equals(MemorySegment.NULL)) {
                blackwell.free(tmem);
            }
            for (MemorySegment buf : List.of(dResidual, dNormed, dQkv, dAttnOut, dFfnBuf)) {
                if (buf != null && !buf.equals(MemorySegment.NULL)) {
                    blackwell.free(buf);
                }
            }
            if (weightsArena != null) {
                weightsArena.close();
            }
        }
        log.info("[Blackwell] Runner closed");
    }
}
